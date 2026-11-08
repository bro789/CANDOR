# CANDOR: Query-Time Value Canonicalization for LLM Text-to-SQL over Dirty Databases

This repository contains the code and data for the paper
*CANDOR: Query-Time Value Canonicalization for LLM Text-to-SQL over Dirty Databases*:

- **CANDOR**, a training-free layer that runs after any text-to-SQL system. It finds the value-sensitive sites of the
  generated query, collects candidate variants of each constant and grouped domain by blocking, verifies them with a
  small open LLM (selection with abstention plus a bidirectional test, and simple rules over value frequencies), and
  rewrites the query into portable SQL (`IN`-lists and `CASE` expressions).
- **BIRD-Dirty**, a text-to-SQL benchmark with paired dirty and clean databases. It injects value-level representation
  variants (case, punctuation/spacing, typos, abbreviations, knowledge aliases) into the 11 BIRD development databases
  and keeps the ground-truth map from every variant to its canonical value.
- The baselines (Norm, Fuzzy, LIKE substring, LLM-Repair, eager cleaning, oracle variant sets), the evaluation under
  clean-answer semantics, the significance tests, and the scripts that produce every table and figure of the paper.

## Repository layout

```
src/                     all Python code (flat; run from src/ or with src/ on PYTHONPATH)
  varfix.py              CANDOR core: site analysis, blocking, verifier prompts, statistics rules, rewriting
  run_varfix.py          rewrite a file of predicted queries with one method (--mode none|like|norm|fuzzy|oracle|ours)
  run_varfix_online.py   the same, one query at a time, for latency measurements
  evaluate.py, sqlexec.py  execution accuracy under clean-answer semantics (affected / unaffected split)
  inject.py              BIRD-Dirty generator;  bench_delta.py  release format (cell-level deltas)
  gen_aliases_kb.py      builds data/aliases_kb.json from standard code lists
  prep_omnisql.py, infer_t2s.py   OmniSQL prompts (with BM25 value retrieval) and vLLM text-to-SQL inference
  llm_repair.py, eager_clean.py   LLM baselines
  scale_domains.py       synthetic domain growth for the scalability study
  sig_tests.py, make_tables.py, paper_figs.py   significance tests, LaTeX tables, figures
scripts/                 end-to-end pipeline (01 ... 08), configured by scripts/env.sh
data/aliases_kb.json     alias lists used by the generator
data/deltas/<config>/    BIRD-Dirty: delta.jsonl.gz (changed cells) + canon_map.json.gz (variant -> canonical)
results/                 the paper's tables as generated (main, seeds, robust, ablation) and significance tests
```

## BIRD-Dirty configurations

| Directory | Rate r | Variant types | Seed | Use in the paper |
|---|---|---|---|---|
| `main_r20_s0` | 0.2 | all five | 0 | development |
| `main_r20_s1` | 0.2 | all five | 1 | validation, component analysis |
| `main_r20_s2`, `main_r20_s3`, `main_r20_s4` | 0.2 | all five | 2, 3, 4 | held-out test sets |
| `rate_r05_s4`, `rate_r10_s4`, `rate_r30_s4` | 0.05 / 0.1 / 0.3 | all five | 4 | rate sweep |
| `type_{case,punct,typo,abbrev,alias}_s4` | 0.2 | one type | 4 | per-type robustness |

The databases are distributed as deltas against the original BIRD databases: `scripts/01_build_benchmark.sh` copies the
BIRD development databases and applies the changed cells, which reproduces the databases used in the paper exactly (the
script prints a content hash per configuration). `inject.py` generates new configurations from scratch
(`python src/inject.py --out DIR --rate 0.2 --types case,punct,typo,abbrev,alias --seed N`).

## Setup

1. Python 3.10/3.11 and Java 11 (for pyserini). Install the requirements:
   `pip install -r requirements.txt`. The LLM parts (text-to-SQL generators and the verifier) run with vLLM on one
   48 GB GPU; the rule-based baselines and the evaluation need no GPU.
2. Download the BIRD development set (https://bird-bench.github.io/) and unpack it so that
   `data/bird/dev_20240627/{dev.json,dev_tables.json,dev_databases/}` exist, or point `CANDOR_BIRD_DEV` to it.
3. For prompt construction, clone OmniSQL (https://github.com/RUCKBReasoning/OmniSQL, commit `a66d732`) into
   `third_party/OmniSQL`, or point `OMNISQL_DIR` to its `train_and_evaluate` directory.
4. Model names in `scripts/env.sh` are Hugging Face ids; set the variables there to local paths if needed.

## Reproducing the paper

```bash
bash scripts/01_build_benchmark.sh    # BIRD-Dirty databases from the deltas
bash scripts/02_text_to_sql.sh        # prompts + SQL from the five generators
bash scripts/03_main_experiments.sh   # all methods x generators x held-out seeds, and on the clean databases
bash scripts/04_llm_baselines.sh      # LLM-Repair and eager cleaning (seed 3)
bash scripts/05_robustness.sh         # gold SQL on the rate sweep and the per-type configurations (seed 4)
bash scripts/06_components.sh         # component analysis (seed 1) and verifier models (seed 3)
bash scripts/07_cost_scale.sh         # latency and domain-scaling study
bash scripts/08_tables.sh             # significance tests and LaTeX tables
```

Outputs go to `runs/<config>/<generator>_<method>.{json,eval.json,meta.json}`; `make_tables.py` reads only these files.
All parameters of CANDOR are the defaults of `run_varfix.py` (K = 20, tau = 0.5, small domains <= 30 values,
frequency prior 5 / 0.3 with the similarity exception at 85, typo gate at similarity 90).
Use the pinned `sqlglot` version: other versions parse a few queries differently, which changes a handful of rewrites.
BIRD question 1118 compares a birthday with `JULIANDAY('now')`, so its answer, and therefore the affected / unaffected
split, can differ by one question depending on the date on which the gold results are computed.

To apply CANDOR to your own predictions, write a JSON list aligned with a BIRD-format `dev.json`
(`[{"pred_sql": "..."}, ...]`) and run

```bash
cd src
python run_varfix.py --root PATH/TO/ROOT --pred preds.json --mode ours --llm Qwen/Qwen2.5-7B-Instruct --out rewritten.json
python evaluate.py --root PATH/TO/ROOT --pred rewritten.json --out rewritten.eval.json   # needs canon_map.json in ROOT
```

## License

Code: MIT License (see `LICENSE`). The BIRD-Dirty deltas and canonical maps are derived from the BIRD benchmark
(CC BY-SA 4.0) and are released under the same license.
