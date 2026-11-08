"""Driver: rewrite predicted SQL of a text-to-SQL run for a (dirty) BIRD root, then save rewritten SQL + cost stats.

python run_varfix.py --root ROOT --pred PRED.json --mode ours|norm|fuzzy|oracle|none --out OUT.json [--llm PATH]
PRED.json: list aligned with dev.json, items with 'pred_sql' or 'pred_sqls'.

ours: blocking -> pairwise LLM verification (P(Yes) >= tau) ; GROUP sites: star clustering around the most frequent
      value of each block (two rounds). Normalization-identical variants are auto-accepted (auto_norm=1).
"""
import argparse, json, os, time, collections, math
import varfix as V
from rapidfuzz import fuzz, process

CLEAN = os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627')


class LLMJudge:
    def __init__(self, path, gpu_util=0.85, max_len=4096):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.llm = LLM(model=path, dtype='float16' if 'AWQ' in path else 'bfloat16', max_model_len=max_len,
                       gpu_memory_utilization=gpu_util, enable_prefix_caching=True)
        self.SP = SamplingParams
        self.max_len = max_len
        self.too_long = 0  # prompts that exceed the context are not sent: the verifier abstains (no rewrite)
        yes = [t for t in ['Yes', ' Yes', 'yes', 'YES'] if len(self.tok.encode(t, add_special_tokens=False)) == 1]
        no = [t for t in ['No', ' No', 'no', 'NO'] if len(self.tok.encode(t, add_special_tokens=False)) == 1]
        self.yes_ids = {self.tok.encode(t, add_special_tokens=False)[0] for t in yes}
        self.no_ids = {self.tok.encode(t, add_special_tokens=False)[0] for t in no}

    def choice(self, prompts, nopts):
        """multiple-choice scoring: returns list of (probs over letters A.., prompt_tokens)"""
        if not prompts:
            return []
        chat = [self.tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
                for p in prompts]
        res = [None] * len(chat)
        fits = self._fits(chat, 1)
        for i in range(len(chat)):
            if not fits[i]:
                res[i] = (dict.fromkeys(V.LET[:nopts[i]], 0.0), 0)
        order = sorted([i for i in range(len(chat)) if fits[i]], key=lambda i: chat[i])
        outs = self.llm.generate([chat[i] for i in order], self.SP(temperature=0.0, max_tokens=1, logprobs=20)) if order else []
        for i, o in zip(order, outs):
            lets = V.LET[:nopts[i]]
            d = dict.fromkeys(lets, 0.0)
            for k, x in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).items():
                t = (x.decoded_token or '').strip().strip('.').strip(')')
                if t in d:
                    d[t] += math.exp(x.logprob)
            z = sum(d.values())
            res[i] = ({k: (v / z if z else 0.0) for k, v in d.items()}, len(o.prompt_token_ids))
        return res

    def gen(self, prompts, max_tokens=64):
        """free-form generation: returns list of (text, prompt_tokens, completion_tokens)"""
        if not prompts:
            return []
        chat = [self.tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
                for p in prompts]
        fits = self._fits(chat, max_tokens)
        idx = [i for i in range(len(chat)) if fits[i]]
        outs = self.llm.generate([chat[i] for i in idx], self.SP(temperature=0.0, max_tokens=max_tokens)) if idx else []
        res = [('', 0, 0)] * len(chat)
        for i, o in zip(idx, outs):
            res[i] = (o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids))
        return res

    def _fits(self, chat, max_tokens):
        ok = [len(self.tok.encode(c, add_special_tokens=False)) + max_tokens <= self.max_len for c in chat]
        if not all(ok):
            self.too_long += ok.count(False)
            print(f'[LLMJudge] {ok.count(False)} prompt(s) exceed max_len={self.max_len}: abstained (total {self.too_long})', flush=True)
        return ok

    def pyes(self, prompts):
        """returns list of (p_yes, prompt_tokens)"""
        if not prompts:
            return []
        chat = [self.tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
                for p in prompts]
        order = sorted(range(len(chat)), key=lambda i: chat[i])  # adjacent shared prefixes
        outs = self.llm.generate([chat[i] for i in order], self.SP(temperature=0.0, max_tokens=1, logprobs=20))
        res = [None] * len(chat)
        for i, o in zip(order, outs):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = sum(math.exp(v.logprob) for k, v in lp.items() if k in self.yes_ids)
            pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in self.no_ids)
            res[i] = (py / (py + pn) if py + pn > 0 else 0.0, len(o.prompt_token_ids))
        return res


def load_preds(p):
    preds = json.load(open(p))
    out = []
    for x in preds:
        s = x.get('pred_sql')
        if s is None:
            s = (x.get('pred_sqls') or [''])[0]
        out.append(s or '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--mode', default='ours')
    ap.add_argument('--out', required=True)
    ap.add_argument('--llm', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--tau', type=float, default=0.5, help='P(Yes) acceptance threshold')
    ap.add_argument('--auto_norm', type=int, default=1)
    ap.add_argument('--restrict', type=int, default=1, help='query-driven domain restriction for group sites')
    ap.add_argument('--small_dom', type=int, default=30, help='domains up to this size: every value is a candidate')
    ap.add_argument('--K', type=int, default=20, help='max blocking candidates per constant')
    ap.add_argument('--fuzzy_tau', type=float, default=90)
    ap.add_argument('--no_group', type=int, default=0)
    ap.add_argument('--no_eq', type=int, default=0)
    ap.add_argument('--keep_keys', type=int, default=0, help='ablation: also canonicalize key/id-like columns')
    ap.add_argument('--gpu_util', type=float, default=0.85)
    ap.add_argument('--twin_rule', type=int, default=0, help='(superseded by class-level verification)')
    ap.add_argument('--mutual', type=int, default=1, help='bidirectional selection test before accepting a variant')
    # v3 (frozen 2026-11-06): the three options below are ON by default; set to 0 to obtain v2
    ap.add_argument('--stats_rules', type=int, default=1, help='v3: short-code / frequency-prior rejects + typo gate')
    ap.add_argument('--expand', type=int, default=1, help='v3: LLM query-expansion blocking for selection constants')
    ap.add_argument('--wrapped', type=int, default=1, help='v3: treat LOWER/UPPER/TRIM(col) = literal as a selection site')
    ap.add_argument('--typo_gate', type=int, default=1, help='ablation only: 0 disables the typo gate inside the statistics rules')
    ap.add_argument('--freq_prior', type=int, default=1, help='ablation only: 0 disables the frequency prior inside the statistics rules')
    ap.add_argument('--prior_sim', type=int, default=85, help='v4 (adopted): frequency prior skipped for pairs with similarity >= this (0 = v3)')
    ap.add_argument('--formulation', default='select_nodist', choices=['select', 'select_nodist', 'pair'],
                    help='v2 default select_nodist (target + none); select = with in-domain distractors (v1)')
    ap.add_argument('--dump_pairs', default='')
    a = ap.parse_args()
    V.WRAPPED[0] = bool(a.wrapped)
    V.PRIOR_SIM[0] = a.prior_sim

    dev = json.load(open(f'{CLEAN}/dev.json'))
    tables = {d['db_id']: d for d in json.load(open(f'{CLEAN}/dev_tables.json'))}
    sqls = load_preds(a.pred)
    assert len(sqls) == len(dev)
    cmap = json.load(open(f'{a.root}/canon_map.json')) if os.path.exists(f'{a.root}/canon_map.json') else {}
    schemas = {db: V.Schema(info) for db, info in tables.items()}
    keys = {db: V.key_columns(info) for db, info in tables.items()}
    dbs = {db: V.DB(f'{a.root}/dev_databases/{db}/{db}.sqlite') for db in tables}
    judge = LLMJudge(a.llm, gpu_util=a.gpu_util) if a.mode == 'ours' else None
    t0 = time.time()
    t_llm = 0.0

    def canon_of(db, tc, v):
        return cmap.get(db, {}).get(f'{tc[0]}.{tc[1]}', {}).get(v, v)

    def skip_col(db, tc, dom):
        return (not a.keep_keys) and V.is_key_or_idlike(keys[db], tc, dom)

    pair_log = []

    def run_select(reqs):
        """reqs: list of (item, db, tc, v, opts) -> list of {opt: prob, None: prob_none}"""
        nonlocal t_llm
        prompts = [V.select_prompt(db, tc, v, opts, dbs[db].domain(*tc)) for (_, db, tc, v, opts) in reqs]
        t1 = time.time()
        res = judge.choice(prompts, [len(r[4]) + 1 for r in reqs])
        t_llm += time.time() - t1
        out = []
        for (it, db, tc, v, opts), (pr, pt) in zip(reqs, res):
            it['stats']['llm_calls'] += 1
            it['stats']['ptok'] += pt
            d = {o: pr[V.LET[j]] for j, o in enumerate(opts)}
            d[None] = pr[V.LET[len(opts)]]
            out.append(d)
        return out

    def shuffled(xs, seed):
        import random as _r
        xs = list(xs)
        _r.Random(seed).shuffle(xs)
        return xs

    def run_pairs(reqs):
        """eq verification. reqs: list of (item, db, tc, target, v, exclude) -> P(v denotes target)"""
        nonlocal t_llm
        if a.formulation == 'pair':  # ablation: pairwise yes/no judgment
            prompts = [V.pair_prompt(db, tc, x, y, dbs[db].domain(*tc)) for (_, db, tc, x, y, _) in reqs]
            t1 = time.time()
            res = judge.pyes(prompts)
            t_llm += time.time() - t1
            out = []
            for (it, db, tc, x, y, _), (p, pt) in zip(reqs, res):
                it['stats']['llm_calls'] += 1
                it['stats']['ptok'] += pt
                out.append(p)
                if a.dump_pairs:
                    pair_log.append({'i': it['i'], 'db': db, 'tc': tc, 'a': x, 'b': y, 'p': p,
                                     'truth': canon_of(db, tc, x) == canon_of(db, tc, y)})
            return out
        sreq = []
        for (it, db, tc, x, y, exc) in reqs:
            dom = dbs[db].domain(*tc)
            dis = [] if a.formulation == 'select_nodist' else V.distractors(y, x, dom, exclude=exc)
            opts = shuffled([x] + dis, f'{db}|{tc}|{x}|{y}')
            sreq.append((it, db, tc, y, opts))
        out = []
        for (it, db, tc, x, y, exc), d in zip(reqs, run_select(sreq) if sreq else []):
            p = d[x]
            out.append(p if p >= max(d.values()) else min(p, a.tau - 1e-6))  # must also be the argmax
            if a.dump_pairs:
                pair_log.append({'i': it['i'], 'db': db, 'tc': tc, 'a': x, 'b': y, 'p': p,
                                 'truth': canon_of(db, tc, x) == canon_of(db, tc, y)})
        return out

    # ---------- phase 1: analysis + eq candidates
    items = []
    for i, (e, sql) in enumerate(zip(dev, sqls)):
        db = e['db_id']
        tree, eqs, groups = V.analyze(sql, schemas[db]) if (sql and a.mode != 'none') else (None, [], [])
        if a.no_eq:
            eqs = []
        if a.no_group:
            groups = []
        items.append({'i': i, 'db': db, 'q': e['question'], 'sql': sql, 'tree': tree, 'eqs': eqs, 'groups': groups,
                      'stats': collections.Counter()})
    # v3 (optional): LLM query expansion as an extra blocking source -- propose alternative forms of each constant and
    # look them up in the attribute domain by normalization key (aliases with no lexical overlap: 'cl' -> 'chlorine').
    expansions = {}
    if a.expand and judge:
        ereq = []
        for it in items:
            for s in it['eqs']:
                dom = dbs[it['db']].domain(*s['tc'])
                if skip_col(it['db'], s['tc'], dom) or len(dom) <= a.small_dom:
                    continue
                for c in s['consts']:
                    ereq.append((it, it['db'], s['tc'], c))
        t1 = time.time()
        outs = judge.gen([V.expand_prompt(db, tc, c, dbs[db].domain(*tc)) for (_, db, tc, c) in ereq], max_tokens=60)
        t_llm += time.time() - t1
        for (it, db, tc, c), (txt, pt, ct) in zip(ereq, outs):
            it['stats']['llm_calls'] += 1
            it['stats']['llm_expand'] += 1
            it['stats']['ptok'] += pt
            it['stats']['ctok'] += ct
            nidx = dbs[db].norm_index(*tc)
            hits = []
            for g in V.parse_strlist(txt):
                hits += [x for x in nidx.get(V.norm(g) or g, []) if x != c]
            expansions[(it['i'], db, tuple(tc), c)] = list(dict.fromkeys(hits))
    reqs, refs = [], []
    for it in items:
        db = it['db']
        for s in it['eqs']:
            dom = dbs[db].domain(*s['tc'])
            s['varsets'] = [[c] for c in s['consts']]
            if a.mode != 'oracle' and skip_col(db, s['tc'], dom):
                it['stats']['skipped_key'] += 1
                continue
            for k, c in enumerate(s['consts']):
                vs = s['varsets'][k]
                if a.mode == 'oracle':
                    cc = canon_of(db, s['tc'], c)
                    vs += [v for v, _ in dom if v != c and canon_of(db, s['tc'], v) == cc]
                    continue
                if len(dom) <= a.small_dom and a.mode == 'ours':
                    sc = {v: (sco, t) for v, sco, t in V.eq_candidates(c, dom, K=len(dom), tau=0)}
                    cands = [(v,) + sc.get(v, (0.0, 'all')) for v, _ in dom if v != c]
                else:
                    cands = V.eq_candidates(c, dom, K=a.K)
                if expansions:
                    have = {x[0] for x in cands}
                    cands = cands + [(x, 0.0, 'expand') for x in expansions.get((it['i'], db, tuple(s['tc']), c), [])
                                     if x not in have]
                if a.mode == 'like':  # common hack: A LIKE '%c%' (case-insensitive substring), as an explicit IN-list
                    cl = c.lower().strip()
                    vs += [v for v, _ in dom if v != c and cl and cl in v.lower()]
                    continue
                if a.mode == 'norm':
                    vs += [v for v, _, t in cands if t == 'norm']
                elif a.mode == 'fuzzy':
                    vs += [v for v, sc, t in cands if t in ('norm', 'abbr') or sc >= a.fuzzy_tau]
                elif a.mode == 'ours':
                    # normalization classes: the target's class is auto-accepted; every other candidate class is
                    # verified once through its most frequent member and accepted/rejected as a whole.
                    cnt = dict(dom)
                    classes = collections.OrderedDict()
                    nc = V.norm(c)
                    for v, sc, t in cands:
                        classes.setdefault(V.norm(v) or v, []).append(v)
                    auto = classes.pop(nc, []) if (a.auto_norm and nc) else []
                    if auto:
                        auto = [x for x in dbs[db].norm_index(*s['tc']).get(nc, auto) if x != c]
                    vs += auto
                    for key, mem in classes.items():
                        full_mem = dbs[db].norm_index(*s['tc']).get(key, mem)
                        rep = max(full_mem, key=lambda x: (cnt.get(x, 0), x))
                        if a.twin_rule and V.has_competing_twin(rep, c, dom):
                            it['stats']['twin_rejects'] += 1
                            continue
                        if a.stats_rules and (V.short_codes(c, rep) or (a.freq_prior and V.freq_distinct(cnt.get(c, 0), cnt.get(rep, 0), a=c, b=rep))):
                            it['stats']['stat_rejects'] += 1
                            continue
                        reqs.append((it, db, s['tc'], c, rep, tuple(auto) + tuple(full_mem)))
                        refs.append((s, k, full_mem))
    # ---------- phase 2: LLM verification of eq candidates
    if judge and reqs:
        acc = [(r, q, ref) for r, q, ref in zip(reqs, run_pairs(reqs), refs) if q >= a.tau]
        if a.mutual and acc:
            # reverse direction: does the target constant denote the accepted representative? (bidirectional test)
            gated, need = [], []
            for x in acc:
                (it, db, tc, c, rep, exc) = x[0]
                fr = dict(dbs[db].domain(*tc))
                if a.stats_rules and a.typo_gate and V.typo_gate(c, rep, fr.get(c, 0), fr.get(rep, 0)):
                    it['stats']['typo_gated'] += 1
                    gated.append(x)
                else:
                    need.append(x)
            rev = [(it, db, tc, rep, c, tuple(exc)) for (it, db, tc, c, rep, exc), _, _ in need]
            back = run_pairs(rev) if rev else []
            acc = gated + [x for x, q in zip(need, back) if q >= a.tau]
            for (it, *_), _, _ in acc:
                pass
        for (it, db, tc, c, rep, exc), q, (s, k, mem) in acc:
            s['varsets'][k] += [v for v in mem if v not in s['varsets'][k]]
    for it in items:
        for s in it['eqs']:
            if cmap:  # variant-discovery quality vs ground truth (selection sites)
                dom = dbs[it['db']].domain(*s['tc'])
                for c, x in zip(s['consts'], s['varsets']):
                    cc = canon_of(it['db'], s['tc'], c)
                    T = {v for v, _ in dom if v != c and canon_of(it['db'], s['tc'], v) == cc}
                    P = set(x) - {c}
                    it['stats']['var_tp'] += len(P & T)
                    it['stats']['var_fp'] += len(P - T)
                    it['stats']['var_fn'] += len(T - P)
                    it['stats']['var_sites_exact'] += int(P == T)
                    it['stats']['var_sites'] += 1
            vs = []
            for x in s['varsets']:
                vs += [v for v in x if v not in vs]
            if len(vs) > len(s['consts']):
                V.apply_eq(s, vs)
                it['stats']['eq_rewritten'] += 1
        it['stats']['eq_sites'] = len(it['eqs'])
        it['stats']['group_sites'] = len(it['groups'])
    # ---------- phase 3: group sites (domain restriction, blocking, star clustering)
    pending = []  # (item, site, block-list-of-values)
    for it in items:
        db = it['db']
        for s in it['groups']:
            full = dbs[db].domain(*s['tc'])
            cnt = dict(full)
            s['groups_out'] = []
            if a.mode == 'oracle':
                by = collections.defaultdict(list)
                for v in cnt:
                    by[canon_of(db, s['tc'], v)].append(v)
                s['groups_out'] = [vs for vs in by.values() if len(vs) > 1]
                continue
            if skip_col(db, s['tc'], full):
                it['stats']['skipped_key'] += 1
                continue
            vals = None
            if a.restrict:
                try:
                    rs = dbs[db].query(V.restricted_domain_sql(s))
                    if rs is not None:
                        vals = sorted(set(r[0] for r in rs if isinstance(r[0], str)))
                except Exception:
                    vals = None
            if vals is None:
                vals = sorted(cnt)
            it['stats']['group_dom'] += len(vals)
            if a.mode == 'like':
                continue
            if a.mode == 'norm':
                by = collections.defaultdict(list)
                for v in vals:
                    by[V.norm(v)].append(v)
                s['groups_out'] = [vs for k, vs in by.items() if len(vs) > 1 and k]
                continue
            blocks = V.group_blocks(vals)
            if a.mode == 'fuzzy':
                s['groups_out'] = blocks
                continue
            for b in blocks:
                pending.append((it, s, sorted(b, key=lambda v: (-cnt.get(v, 0), v))))
    # greedy attach-to-canonical: values in descending frequency either attach to an existing canonical of
    # their block (LLM select among the most similar canonicals) or become a new canonical. One LLM batch per round.
    states = []
    for it, s, b in pending:
        states.append({'it': it, 's': s, 'canon': [b[0]], 'mem': {b[0]: [b[0]]}, 'queue': list(b[1:])})
    while a.mode == 'ours' and any(st['queue'] for st in states):
        sreq, sref = [], []
        for st in states:
            while st['queue']:
                v = st['queue'].pop(0)
                hit = [c for c in st['canon'] if a.auto_norm and V.norm(c) == V.norm(v) and V.norm(v)]
                if hit:
                    st['mem'][hit[0]].append(v)
                    continue
                db, tc = st['it']['db'], st['s']['tc']
                leaders = st['canon']
                if a.stats_rules:
                    fr = dict(dbs[db].domain(*tc))
                    leaders = [u for u in leaders if not (V.short_codes(u, v) or (a.freq_prior and V.freq_distinct(fr.get(u, 0), fr.get(v, 0), a=u, b=v)))]
                    if not leaders:  # statistics alone say v is a value of its own
                        st['it']['stats']['stat_rejects'] += 1
                        st['canon'].append(v)
                        st['mem'][v] = [v]
                        continue
                cands = [x[0] for x in process.extract(v, leaders, scorer=fuzz.WRatio, limit=4, processor=str.lower)]
                sreq.append((st['it'], db, tc, v, shuffled(cands, f'{db}|{tc}|{v}')))
                sref.append((st, v))
                break
        if not sreq:
            break
        res = run_select(sreq)
        ok = {}
        if a.mutual:
            # reverse check for tentative attachments: does the canonical denote v?
            rq, rr = [], []
            for j, ((st, v), d) in enumerate(zip(sref, res)):
                best = max(d, key=d.get)
                if best is not None and d[best] >= a.tau:
                    db, tc = st['it']['db'], st['s']['tc']
                    if a.stats_rules:
                        fr = dict(dbs[db].domain(*tc))
                        if a.typo_gate and V.typo_gate(best, v, fr.get(best, 0), fr.get(v, 0)):
                            st['it']['stats']['typo_gated'] += 1
                            continue
                    rq.append((st['it'], db, tc, v, best, ()))
                    rr.append(j)
            for j, q in zip(rr, run_pairs(rq) if rq else []):
                ok[j] = q >= a.tau
        for j, ((st, v), d) in enumerate(zip(sref, res)):
            best = max(d, key=d.get)
            if best is not None and d[best] >= a.tau and ok.get(j, True):
                st['mem'][best].append(v)
                if a.dump_pairs:
                    db, tc = st['it']['db'], st['s']['tc']
                    pair_log.append({'i': st['it']['i'], 'db': db, 'tc': tc, 'a': best, 'b': v, 'p': d[best],
                                     'truth': canon_of(db, tc, best) == canon_of(db, tc, v), 'kind': 'group'})
            else:
                st['canon'].append(v)
                st['mem'][v] = [v]
    for st in states:
        for c, m in st['mem'].items():
            if len(m) > 1:
                st['s']['groups_out'].append(m)
    # ---------- phase 4: apply group rewrites, emit
    out = []
    for it in items:
        db = it['db']
        for s in it['groups']:
            full = dict(dbs[db].domain(*s['tc']))
            gl = []
            for g in s.get('groups_out', []):
                g = sorted(set(g), key=lambda v: (-full.get(v, 0), v))
                if len(g) > 1:
                    gl.append((g[0], g[1:]))
            if gl:
                n = V.apply_group(s, gl, schemas[db])
                it['stats']['group_rewritten'] += 1 if n else 0
        changed = it['tree'] is not None and (it['stats']['eq_rewritten'] or it['stats']['group_rewritten'])
        new_sql = it['tree'].sql(dialect='sqlite') if changed else it['sql']
        out.append({'i': it['i'], 'pred_sql': new_sql, 'orig_sql': it['sql'], 'stats': dict(it['stats'])})
    tot = collections.Counter()
    for o in out:
        tot.update(o['stats'])
    meta = {'args': vars(a), 'time_s': round(time.time() - t0, 1), 'llm_time_s': round(t_llm, 1), 'totals': dict(tot)}
    json.dump(out, open(a.out, 'w'), ensure_ascii=False)
    json.dump(meta, open(a.out.replace('.json', '.meta.json'), 'w'), indent=1)
    if a.dump_pairs:
        json.dump(pair_log, open(a.dump_pairs, 'w'), ensure_ascii=False)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
