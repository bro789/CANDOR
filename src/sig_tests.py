"""Paired significance of CANDOR v4 vs each baseline: exact McNemar test and paired bootstrap 95% CI of the EX difference.
python sig_tests.py OUT.json ROOT [ROOT...]   (roots under runs/, e.g. main_r20_s3)
Per root and generator: every baseline that exists; 'strongest' = the non-oracle baseline with the highest EX."""
import json, os, sys, random
from math import comb
R = os.environ.get('CANDOR_RUNS', 'runs')
TAGS = ['omni7b', 'qwen7b', 'qwen32', 'omni14b', 'llama8']
BASES = ['none', 'like', 'norm', 'fuzzy', 'repair_q7', 'repair_q32', 'eagerv4_q7', 'eagerv4_norm_q7']
OURS = 'ours_v4'
B = 10000


def recs(rt, tag, m):
    f = f'{R}/{rt}/{tag}_{m}.eval.json'
    if not os.path.exists(f):
        return None
    return [bool(r['correct']) for r in sorted(json.load(open(f))['records'], key=lambda r: r['i'])]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def boot(x, y, seed=0):
    n = len(x)
    d = [int(a) - int(b) for a, b in zip(x, y)]
    rng = random.Random(seed)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) * 100 / n for _ in range(B))
    return s[int(0.025 * B)], s[int(0.975 * B) - 1]


out = []
for rt in sys.argv[2:]:
    for tag in TAGS:
        o = recs(rt, tag, OURS)
        if o is None:
            continue
        rows = []
        for m in BASES:
            b = recs(rt, tag, m)
            if b is None:
                continue
            assert len(b) == len(o), (rt, tag, m)
            n01 = sum(1 for x, y in zip(o, b) if x and not y)   # CANDOR right, baseline wrong
            n10 = sum(1 for x, y in zip(o, b) if y and not x)   # baseline right, CANDOR wrong
            lo, hi = boot(o, b)
            rows.append({'root': rt, 'gen': tag, 'baseline': m, 'n': len(o), 'EX_ours': round(100 * sum(o) / len(o), 2),
                         'EX_base': round(100 * sum(b) / len(b), 2), 'diff': round(100 * (sum(o) - sum(b)) / len(o), 2),
                         'ci95': [round(lo, 2), round(hi, 2)], 'wins': n01, 'losses': n10, 'p_mcnemar': mcnemar_exact(n01, n10)})
        nonor = [r for r in rows if r['baseline'] != 'none']
        if nonor:
            s = max(nonor, key=lambda r: r['EX_base'])
            for r in rows:
                r['strongest'] = r is s
        out += rows
        for r in rows:
            print(f"{rt} {tag:8s} vs {r['baseline']:16s} {r['EX_ours']:6.2f} - {r['EX_base']:6.2f} = {r['diff']:+6.2f} "
                  f"CI[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] W/L {r['wins']}/{r['losses']} p={r['p_mcnemar']:.2e}"
                  f"{'  <- strongest' if r.get('strongest') else ''}", flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
