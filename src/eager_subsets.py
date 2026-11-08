"""Re-summarize the eager baseline with the affected/unaffected split of the DIRTY root (comparable to other methods).
python eager_subsets.py ROOTNAME"""
import os
import json, sys
rn = sys.argv[1]
names = sys.argv[2:] or ['eager_q7']
base = json.load(open(f'{os.environ.get("CANDOR_RUNS", "runs")}/{rn}/omni7b_none.eval.json'))['records']
aff = [r['affected'] for r in base]
for tag, name in [(t, n) for t in ('omni7b', 'qwen7b', 'qwen32', 'omni14b', 'llama8') for n in names]:
    f = f'{os.environ.get("CANDOR_RUNS", "runs")}/{rn}/{tag}_{name}.eval.json'
    try:
        d = json.load(open(f))
    except FileNotFoundError:
        continue
    recs = d['records'] if 'records' in d else None
    if recs is None:
        continue
    for r in recs:
        r['affected'] = aff[r['i']]
    n = len(recs)
    A = [r for r in recs if r['affected']]
    U = [r for r in recs if not r['affected']]
    s = {'n': n, 'EX': round(100 * sum(r['correct'] for r in recs) / n, 2), 'n_aff': len(A),
         'EX_aff': round(100 * sum(r['correct'] for r in A) / len(A), 2),
         'EX_unaff': round(100 * sum(r['correct'] for r in U) / len(U), 2), 'exec_err': sum(not r['ok'] for r in recs)}
    json.dump({'summary': s, 'records': recs}, open(f, 'w'))
    print(tag, name, s)
