"""Eager (offline, query-agnostic) baseline: canonicalize EVERY candidate text column of every DB up front with the
same blocking + LLM select verifier, write a cleaned copy of the root, and record the LLM cost.
python eager_clean.py --root DIRTY_ROOT --out CLEAN_ROOT [--llm PATH]
Columns: text, not PK/FK, not identifier-like, average length <= 60.
"""
import argparse, json, os, shutil, sqlite3, time, collections, random
import varfix as V
from rapidfuzz import fuzz, process
from run_varfix import LLMJudge

CLEAN = os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--llm', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--max_block_n', type=int, default=20000)
    ap.add_argument('--formulation', default='select_nodist', help='same verifier as CANDOR')
    ap.add_argument('--stats_rules', type=int, default=1, help='same statistics rules as CANDOR v3')
    ap.add_argument('--max_block', type=int, default=256, help='split larger blocks into chunks (bounded rounds)')
    ap.add_argument('--prior_sim', type=int, default=85, help='same v4 frequency-prior rule as CANDOR')
    a = ap.parse_args()
    V.PRIOR_SIM[0] = a.prior_sim
    tables = {d['db_id']: d for d in json.load(open(f'{CLEAN}/dev_tables.json'))}
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.root, a.out, ignore=shutil.ignore_patterns('db_contents_index', 'pred_*', 'dev_bird.json', '*.pkl'))
    judge = LLMJudge(a.llm)
    t0 = time.time()
    stats = collections.Counter()
    percol = {}
    states = []
    for db, info in sorted(tables.items()):
        dbo = V.DB(f'{a.root}/dev_databases/{db}/{db}.sqlite')
        keys = V.key_columns(info)
        tn = info['table_names_original']
        for (ti, c), ty in zip(info['column_names_original'], info['column_types']):
            if ti < 0:
                continue
            tc = (tn[ti], c)
            dom = dbo.domain(*tc)
            if len(dom) < 2 or V.is_key_or_idlike(keys, tc, dom):
                continue
            if sum(len(v) for v, _ in dom) / len(dom) > 60:
                continue
            stats['columns'] += 1
            stats['values'] += len(dom)
            cnt = dict(dom)
            vals = sorted(cnt)
            blocks = V.group_blocks(vals, max_pairs_n=a.max_block_n)
            for b in blocks:
                # fuzzy chaining can produce giant blocks (e.g., thousands of display names); split them with
                # sorted-neighborhood windows over the normalized key so that variants stay in the same chunk,
                # then order each chunk by frequency for the greedy rounds
                if len(b) > a.max_block:
                    b = sorted(b, key=lambda v: (V.norm(v) or v, v.lower()))
                    chunks = [b[j:j + a.max_block] for j in range(0, len(b), a.max_block)]
                else:
                    chunks = [b]
                for ch in chunks:
                    ch = sorted(ch, key=lambda v: (-cnt[v], v))
                    if len(ch) < 2:
                        continue
                    if len(b) > a.max_block:
                        stats['split_blocks'] += 1
                    states.append({'db': db, 'tc': tc, 'canon': [ch[0]], 'mem': {ch[0]: [ch[0]]}, 'queue': ch[1:], 'dom': dom})
        print(db, dict(stats), flush=True)
    rounds = 0
    while any(st['queue'] for st in states):
        reqs, refs = [], []
        for st in states:
            while st['queue']:
                v = st['queue'].pop(0)
                hit = [c for c in st['canon'] if V.norm(c) == V.norm(v) and V.norm(v)]
                if hit:
                    st['mem'][hit[0]].append(v)
                    continue
                leaders = st['canon']
                if a.stats_rules:
                    fr = dict(st['dom'])
                    leaders = [u for u in leaders if not (V.short_codes(u, v) or V.freq_distinct(fr.get(u, 0), fr.get(v, 0), a=u, b=v))]
                    if not leaders:
                        stats['stat_rejects'] += 1
                        st['canon'].append(v)
                        st['mem'][v] = [v]
                        continue
                cands = [x[0] for x in process.extract(v, leaders, scorer=fuzz.WRatio, limit=4, processor=str.lower)]
                random.Random(f"{st['db']}|{st['tc']}|{v}").shuffle(cands)
                reqs.append(V.select_prompt(st['db'], st['tc'], v, cands, st['dom']))
                refs.append((st, v, cands))
                break
        if not reqs:
            break
        rounds += 1
        res = judge.choice(reqs, [len(r[2]) + 1 for r in refs])
        tent = []
        for (st, v, cands), (pr, pt) in zip(refs, res):
            stats['llm_calls'] += 1
            stats['ptok'] += pt
            j = max(range(len(cands) + 1), key=lambda k: pr[V.LET[k]])
            tent.append(cands[j] if (j < len(cands) and pr[V.LET[j]] >= a.tau) else None)
        # bidirectional check (same as CANDOR): does the chosen leader denote v?
        rq, rj = [], []
        for k, ((st, v, cands), best) in enumerate(zip(refs, tent)):
            if best is not None:
                if a.stats_rules:
                    fr = dict(st['dom'])
                    if V.typo_gate(best, v, fr.get(best, 0), fr.get(v, 0)):
                        stats['typo_gated'] += 1
                        continue
                opts = [v] + (V.distractors(best, v, st['dom']) if a.formulation == 'select' else [])
                random.Random(f"{st['db']}|{st['tc']}|{best}|{v}").shuffle(opts)
                rq.append((V.select_prompt(st['db'], st['tc'], best, opts, st['dom']), opts, v))
                rj.append(k)
        if rq:
            back = judge.choice([x[0] for x in rq], [len(x[1]) + 1 for x in rq])
            for k, (pr, pt), (_, opts, v) in zip(rj, back, rq):
                stats['llm_calls'] += 1
                stats['ptok'] += pt
                lv = V.LET[opts.index(v)]
                if not (pr[lv] >= a.tau and pr[lv] >= max(pr.values())):
                    tent[k] = None
        for (st, v, cands), best in zip(refs, tent):
            if best is not None:
                st['mem'][best].append(v)
            else:
                st['canon'].append(v)
                st['mem'][v] = [v]
    # apply mapping (variant -> canonical = most frequent member)
    emap = {}
    for db in tables:
        con = sqlite3.connect(f'{a.out}/dev_databases/{db}/{db}.sqlite')
        for st in [s for s in states if s['db'] == db]:
            cnt = dict(st['dom'])
            for c, m in st['mem'].items():
                if len(m) < 2:
                    continue
                m = sorted(set(m), key=lambda v: (-cnt.get(v, 0), v))
                t, col = st['tc']
                con.executemany(f'UPDATE `{t}` SET `{col}`=? WHERE `{col}`=?', [(m[0], v) for v in m[1:]])
                stats['merged_values'] += len(m) - 1
                for v in m[1:]:
                    emap.setdefault(db, {}).setdefault(f'{t}.{col}', {})[v] = m[0]
        con.commit()
        con.close()
    json.dump(emap, open(f'{a.out}/eager_mapping.json', 'w'), ensure_ascii=False)
    meta = {'args': vars(a), 'time_s': round(time.time() - t0, 1), 'rounds': rounds, 'stats': dict(stats)}
    json.dump(meta, open(f'{a.out}/eager_meta.json', 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
