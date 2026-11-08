#!/bin/bash
# LLM baselines on held-out seed 3: LLM-Repair (7B / 32B) and eager cleaning (+ query-time Norm).
source "$(dirname "$0")/env.sh"
RN=main_r20_s3; R=$(root_of $RN); O=$CANDOR_RUNS/$RN; mkdir -p "$O"

# LLM-Repair: the LLM rewrites the query given value hints for every site
for T in omni7b qwen7b qwen32 omni14b llama8; do
  for V in q7 q32; do
    L=$VERIFIER; [ $V = q32 ] && L=$VERIFIER_32B
    f=$O/${T}_repair_$V.json
    [ -f "$f" ] || python "$SRC/llm_repair.py" --root "$R" --pred "$R/pred_$T.json" --out "$f" --llm "$L" > "$O/${T}_repair_$V.log" 2>&1
    python "$SRC/evaluate.py" --root "$R" --pred "$f" --out "$O/${T}_repair_$V.eval.json" | tail -n 1
  done
done

# Eager cleaning: canonicalize all eligible text attributes offline, rebuild prompts, regenerate SQL, evaluate
E=$CANDOR_DIRTY/eager_$RN
[ -f "$E/eager_meta.json" ] || python "$SRC/eager_clean.py" --root "$R" --out "$E" --llm "$VERIFIER" > "$O/eager_clean.log" 2>&1
[ -f "$E/dev_bird.json" ] || python "$SRC/prep_omnisql.py" "$E"
for T in omni7b qwen7b qwen32 omni14b llama8; do
  [ -f "$E/pred_$T.json" ] || python "$SRC/infer_t2s.py" --model "${GENERATOR[$T]}" --tag "$T" --roots "$E"
  python "$SRC/evaluate.py" --root "$E" --pred "$E/pred_$T.json" --out "$O/${T}_eagerv4_q7.eval.json" | tail -n 1
  f=$O/${T}_eagerv4_norm_q7.json
  [ -f "$f" ] || python "$SRC/run_varfix.py" --root "$E" --pred "$E/pred_$T.json" --mode norm --out "$f" > /dev/null 2>&1
  python "$SRC/evaluate.py" --root "$E" --pred "$f" --out "$O/${T}_eagerv4_norm_q7.eval.json" | tail -n 1
done
(cd "$SRC" && python eager_subsets.py $RN eagerv4_q7 eagerv4_norm_q7)
