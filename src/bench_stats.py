"""Summary statistics of every BIRD-Dirty root: injected columns/cells by type, distinct variants, and gold-SQL damage."""
import json, os, sys, collections
from evaluate import evaluate, summarize, CLEAN
B = sys.argv[1] if len(sys.argv) > 1 else 'data/bird_dirty'
OUTF = sys.argv[2] if len(sys.argv) > 2 else 'runs/bench_stats.json'
dev = json.load(open(f'{CLEAN}/dev.json'))
rows = {}
for d in sorted(os.listdir(B)):
    r = f'{B}/{d}'
    if not os.path.exists(f'{r}/inject_stats.json'):
        continue
    st = json.load(open(f'{r}/inject_stats.json'))
    cm = json.load(open(f'{r}/canon_map.json'))
    tot = collections.Counter()
    for db, s in st['stats'].items():
        tot.update({k: v for k, v in s.items()})
    nvar = sum(len(vm) for cols in cm.values() for vm in cols.values())
    recs = evaluate(r, [{'pred_sql': e['SQL']} for e in dev])
    s = summarize(recs)
    by_op = collections.Counter()
    for e, rec in zip(dev, recs):
        if rec['affected']:
            q = e['SQL'].upper()
            by_op['groupby' if 'GROUP BY' in q else ('distinct' if 'DISTINCT' in q else 'filter_only')] += 1
    rows[d] = {'rate': st['args']['rate'], 'types': st['args']['types'], 'seed': st['args']['seed'],
               'cols': tot['cols'], 'eligible_cols': tot['eligible_cols'], 'cells': tot['cells'], 'distinct_variants': nvar,
               'cells_by_type': {k: tot[k] for k in ['case', 'punct', 'typo', 'abbrev', 'alias'] if tot[k]},
               'gold_EX': s['EX'], 'n_aff': s['n_aff'], 'aff_by_op': dict(by_op)}
    print(d, json.dumps(rows[d]), flush=True)
json.dump(rows, open(OUTF, 'w'), indent=1)
