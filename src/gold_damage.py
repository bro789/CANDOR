"""How much does value-level dirt hurt even a perfect SQL? EX of gold SQL on dirty root vs clean truth."""
import json, sys
from evaluate import evaluate, summarize, CLEAN
root = sys.argv[1]
dev = json.load(open(f'{CLEAN}/dev.json'))
recs = evaluate(root, [{'pred_sql': e['SQL']} for e in dev])
s = summarize(recs)
print('gold-on-dirty', json.dumps(s))
import collections
by = collections.defaultdict(lambda: [0, 0])
for r in recs:
    by[r['db']][0] += r['affected']
    by[r['db']][1] += 1
print({k: f'{a}/{b}' for k, (a, b) in sorted(by.items())})
json.dump(recs, open(f'{root}/gold_damage.json', 'w'))
