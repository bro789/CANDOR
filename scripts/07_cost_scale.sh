#!/bin/bash
# Per-query latency (warm / cold domain caches) on seed 3, and scalability with value domains grown x10 and x100.
source "$(dirname "$0")/env.sh"
R=$(root_of main_r20_s3)
O=$CANDOR_RUNS/latency; mkdir -p "$O"
for C in 0 1; do
  python "$SRC/run_varfix_online.py" --root "$R" --pred "$R/pred_omni7b.json" --mode ours --llm "$VERIFIER" \
    --online 200 --cold $C --out "$O/cold$C.json" > "$O/cold$C.log" 2>&1
  tail -n 1 "$O/cold$C.log"
done

DBS="card_games superhero thrombosis_prediction financial"
for K in 10 100; do
  [ -d "$CANDOR_DIRTY/scale_x$K" ] || (cd "$SRC" && python scale_domains.py "$R" "$CANDOR_DIRTY/scale_x$K" $K $DBS)
done
O=$CANDOR_RUNS/scale; mkdir -p "$O"
for RN in main_r20_s3 scale_x10 scale_x100; do
  for C in 0 1; do
    python "$SRC/run_varfix_online.py" --root "$(root_of $RN)" --pred "$(root_of $RN)/pred_omni7b.json" --mode ours \
      --llm "$VERIFIER" --online 100 --online_dbs ${DBS// /,} --cold $C --out "$O/${RN}_cold$C.json" > "$O/${RN}_cold$C.log" 2>&1
    echo "$RN cold=$C $(tail -n 1 "$O/${RN}_cold$C.log")"
  done
done
