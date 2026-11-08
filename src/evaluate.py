"""Evaluate predicted SQL on a (dirty) root against clean truth = gold SQL on the clean BIRD DB.
pred file: json list aligned with dev.json; each item has 'pred_sql' (str) or 'pred_sqls' (list, first used).
Outputs per-question records + summary. Result cells are mapped through the injected canonical map.
"""
import argparse, json, os, pickle, collections
from sqlexec import run_many, load_global_map, canon_rows

CLEAN = os.environ.get('CANDOR_BIRD_DEV', 'data/bird/dev_20240627')


def truth_cache():
    p = os.environ.get('CANDOR_TRUTH', 'data/truth_clean.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    dev = json.load(open(f'{CLEAN}/dev.json'))
    jobs = [(f"{CLEAN}/dev_databases/{e['db_id']}/{e['db_id']}.sqlite", e['SQL']) for e in dev]
    res = run_many(jobs, timeout=120)
    out = [set(r) if ok else None for ok, r in res]
    pickle.dump(out, open(p, 'wb'))
    return out


def gold_on_root(root):
    p = f'{root}/gold_on_root.pkl'
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    dev = json.load(open(f'{CLEAN}/dev.json'))
    jobs = [(f"{root}/dev_databases/{e['db_id']}/{e['db_id']}.sqlite", e['SQL']) for e in dev]
    res = run_many(jobs, timeout=120)
    gm = load_global_map(root)
    out = [canon_rows(r, gm.get(e['db_id'])) if ok else None for (ok, r), e in zip(res, dev)]
    pickle.dump(out, open(p, 'wb'))
    return out


def evaluate(root, preds, timeout=30):
    dev = json.load(open(f'{CLEAN}/dev.json'))
    truth = truth_cache()
    affected = [t is not None and g != t for t, g in zip(truth, gold_on_root(root))]
    gm = load_global_map(root)
    sqls = []
    for p in preds:
        s = p.get('pred_sql')
        if s is None:
            s = (p.get('pred_sqls') or [''])[0]
        sqls.append(s or '')
    jobs = [(f"{root}/dev_databases/{e['db_id']}/{e['db_id']}.sqlite", s) for e, s in zip(dev, sqls)]
    res = run_many(jobs, timeout=timeout)
    recs = []
    for i, (e, (ok, r)) in enumerate(zip(dev, res)):
        corr = bool(ok and truth[i] is not None and canon_rows(r, gm.get(e['db_id'])) == truth[i])
        recs.append({'i': i, 'db': e['db_id'], 'diff': e.get('difficulty'), 'ok': ok, 'correct': corr,
                     'affected': affected[i], 'err': None if ok else r})
    return recs


def summarize(recs):
    n = len(recs)
    s = {'n': n, 'EX': round(100 * sum(r['correct'] for r in recs) / n, 2)}
    aff = [r for r in recs if r['affected']]
    un = [r for r in recs if not r['affected']]
    s['n_aff'] = len(aff)
    s['EX_aff'] = round(100 * sum(r['correct'] for r in aff) / max(1, len(aff)), 2)
    s['EX_unaff'] = round(100 * sum(r['correct'] for r in un) / max(1, len(un)), 2)
    s['exec_err'] = sum(not r['ok'] for r in recs)
    return s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    preds = json.load(open(a.pred))
    recs = evaluate(a.root, preds)
    s = summarize(recs)
    print(json.dumps(s))
    if a.out:
        json.dump({'summary': s, 'records': recs}, open(a.out, 'w'))
