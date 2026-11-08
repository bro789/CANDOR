"""Per-query cost of a CANDOR run: LLM calls, prompt tokens, share of queries needing no call, wall time per query."""
import json, sys, statistics as st
f = sys.argv[1]
runs = json.load(open(f))
meta = json.load(open(f.replace('.json', '.meta.json')))
calls = [r['stats'].get('llm_calls', 0) for r in runs]
ptok = [r['stats'].get('ptok', 0) for r in runs]
n = len(runs)
q = sorted(calls)
out = {'n': n, 'calls_mean': round(sum(calls) / n, 2), 'calls_median': st.median(calls), 'calls_p90': q[int(0.9 * n)],
       'calls_max': q[-1], 'zero_call_share': round(sum(c == 0 for c in calls) / n, 3),
       'ptok_mean': round(sum(ptok) / n, 1), 'time_per_q_s': round(meta['time_s'] / n, 3),
       'llm_time_per_q_s': round(meta.get('llm_time_s', 0) / n, 3),
       'eq_sites': meta['totals'].get('eq_sites'), 'group_sites': meta['totals'].get('group_sites'),
       'rewritten_q': sum(1 for r in runs if r['pred_sql'] != r['orig_sql'])}
print(json.dumps(out))
