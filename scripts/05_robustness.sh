#!/bin/bash
# Robustness with the gold SQL as input (Table "robust", Figure "rate"): every matcher on the rate sweep and on each
# variant type in isolation, all on held-out seed 4.
source "$(dirname "$0")/env.sh"
GOLD=$CANDOR_RUNS/gold_pred.json
[ -f "$GOLD" ] || python -c "import json,sys; json.dump([{'pred_sql': e['SQL']} for e in json.load(open(sys.argv[1]))], open(sys.argv[2], 'w'))" \
  "$CANDOR_BIRD_DEV/dev.json" "$GOLD"
for RN in rate_r05_s4 rate_r10_s4 main_r20_s4 rate_r30_s4 type_case_s4 type_punct_s4 type_typo_s4 type_abbrev_s4 type_alias_s4; do
  R=$(root_of $RN); O=$CANDOR_RUNS/gold_$RN; mkdir -p "$O"
  for M in none norm fuzzy like oracle ours; do
    N=$M; X=(); [ $M = ours ] && { N=ours_v4; X=(--llm "$VERIFIER"); }
    f=$O/gold_$N.json
    [ -f "$f" ] || python "$SRC/run_varfix.py" --root "$R" --pred "$GOLD" --mode $M --out "$f" "${X[@]}" > "$O/gold_$N.log" 2>&1
    python "$SRC/evaluate.py" --root "$R" --pred "$f" --out "$O/gold_$N.eval.json" | tail -n 1 | sed "s/^/$RN $N /"
  done
done
