"""Baseline: LLM self-repair with database value hints (agentic-style, one generation per query with sites).
The same site analysis and blocking as ours supply hints; the LLM rewrites the SQL itself.
python llm_repair.py --root ROOT --pred PRED.json --out OUT.json [--llm PATH]
"""
import argparse, json, os, re, time, collections
import varfix as V
from run_varfix import load_preds, CLEAN

TEMPLATE = """You are given a question, a SQLite query generated for it, and hints about the stored values of the columns the
query compares or groups on. The database stores some values inconsistently: the same real-world value may appear with
different letter case, spacing, punctuation, typos, abbreviations or codes.
Rewrite the query so that it returns the answer it would return if every value were stored consistently: filters must
match every stored variant of the intended value (e.g., use IN lists) and grouping / DISTINCT / COUNT(DISTINCT) must treat
variants of the same value as one (e.g., use CASE to map variants to one representative). Do not include values that
denote something different. Keep the query's intent and output columns unchanged. If nothing needs to change, return the
query unchanged.

Question: {q}
SQL:
```sql
{sql}
```
Value hints:
{hints}

Return only the final SQL in a ```sql``` block."""


def parse_sql(t):
    b = re.findall(r"```sql\s*(.*?)\s*```", t, re.S)
    return b[-1].strip() if b else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--llm', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--gpu_util', type=float, default=0.85)
    a = ap.parse_args()
    dev = json.load(open(f'{CLEAN}/dev.json'))
    tables = {d['db_id']: d for d in json.load(open(f'{CLEAN}/dev_tables.json'))}
    sqls = load_preds(a.pred)
    schemas = {db: V.Schema(info) for db, info in tables.items()}
    keys = {db: V.key_columns(info) for db, info in tables.items()}
    dbs = {db: V.DB(f'{a.root}/dev_databases/{db}/{db}.sqlite') for db in tables}
    prompts, idx = [], []
    for i, (e, sql) in enumerate(zip(dev, sqls)):
        db = e['db_id']
        if not sql:
            continue
        tree, eqs, groups = V.analyze(sql, schemas[db])
        hints = []
        for s in eqs:
            dom = dbs[db].domain(*s['tc'])
            if V.is_key_or_idlike(keys[db], s['tc'], dom):
                continue
            for c in s['consts']:
                cands = [v for v, _, _ in V.eq_candidates(c, dom, K=15)]
                if cands:
                    hints.append(f"- `{s['tc'][0]}`.`{s['tc'][1]}`: stored values similar to {json.dumps(c, ensure_ascii=False)}: "
                                 + ', '.join(json.dumps(v, ensure_ascii=False) for v in cands))
        for s in groups:
            full = dbs[db].domain(*s['tc'])
            if V.is_key_or_idlike(keys[db], s['tc'], full):
                continue
            vals = None
            try:
                rs = dbs[db].query(V.restricted_domain_sql(s))
                if rs is not None:
                    vals = sorted(set(r[0] for r in rs if isinstance(r[0], str)))
            except Exception:
                pass
            vals = vals if vals is not None else sorted(dict(full))
            blocks = V.group_blocks(vals)[:15]
            if blocks:
                hints.append(f"- `{s['tc'][0]}`.`{s['tc'][1]}` (grouped/distinct): groups of similar stored values: "
                             + '; '.join('[' + ', '.join(json.dumps(v, ensure_ascii=False) for v in b[:6]) + ']' for b in blocks))
        if hints:
            prompts.append(TEMPLATE.format(q=e['question'], sql=sql, hints='\n'.join(hints)[:6000]))
            idx.append(i)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.llm)
    llm = LLM(model=a.llm, dtype='bfloat16' if 'AWQ' not in a.llm else 'float16', max_model_len=8192,
              gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True)
    chat = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    t0 = time.time()
    outs = llm.generate(chat, SamplingParams(temperature=0, max_tokens=1500))
    out = [{'i': i, 'pred_sql': s, 'orig_sql': s, 'stats': {}} for i, s in enumerate(sqls)]
    ptok = ctok = 0
    for i, o in zip(idx, outs):
        new = parse_sql(o.outputs[0].text)
        ptok += len(o.prompt_token_ids)
        ctok += len(o.outputs[0].token_ids)
        out[i]['stats'] = {'llm_calls': 1, 'ptok': len(o.prompt_token_ids), 'ctok': len(o.outputs[0].token_ids)}
        if new:
            out[i]['pred_sql'] = new
    json.dump(out, open(a.out, 'w'), ensure_ascii=False)
    meta = {'args': vars(a), 'llm_time_s': round(time.time() - t0, 1), 'totals': {'llm_calls': len(idx), 'ptok': ptok, 'ctok': ctok}}
    json.dump(meta, open(a.out.replace('.json', '.meta.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
