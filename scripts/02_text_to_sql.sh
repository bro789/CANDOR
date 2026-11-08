#!/bin/bash
# Build OmniSQL prompts (schema + BM25 value retrieval over each database's own, possibly dirty, content) and run the
# five text-to-SQL generators. Writes ROOT/pred_TAG.json for every root.
# Needs the OmniSQL repository at $OMNISQL_DIR (see README), Java 11 for pyserini, and one GPU (vLLM).
set -e
source "$(dirname "$0")/env.sh"
MAIN="clean main_r20_s2 main_r20_s3 main_r20_s4"
roots() { for rn in "$@"; do root_of "$rn"; done; }

for rn in $MAIN main_r20_s1; do
  R=$(root_of "$rn"); [ -f "$R/dev_bird.json" ] || python "$SRC/prep_omnisql.py" "$R"
done

# all generators on the clean databases and the three held-out seeds
for T in omni7b qwen7b qwen32 omni14b llama8; do
  python "$SRC/infer_t2s.py" --model "${GENERATOR[$T]}" --tag "$T" --roots $(roots $MAIN)
done
# validation seed (component analysis): OmniSQL-7B only
python "$SRC/infer_t2s.py" --model "${GENERATOR[omni7b]}" --tag omni7b --roots "$(root_of main_r20_s1)"
