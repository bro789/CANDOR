#!/bin/bash
# Component analysis on the validation seed (Table "ablation") and the verifier-model comparison on seed 3.
source "$(dirname "$0")/env.sh"
S=main_r20_s1; T=omni7b
mkdir -p "$CANDOR_RUNS/$S"
rewrite $S $T ours v4      --llm "$VERIFIER"                     # full method
rewrite $S $T ours ours_v3 --llm "$VERIFIER" --prior_sim 0       # - similarity exception of the frequency prior
abl() { local N=$1; shift; rewrite $S $T ours v4abl_$N "$@"; }
abl pair       --llm "$VERIFIER" --formulation pair --mutual 0   # pairwise yes/no verifier
abl nomutual   --llm "$VERIFIER" --mutual 0                      # selection without the bidirectional test
abl dist       --llm "$VERIFIER" --formulation select            # + in-domain distractors
abl nostats    --llm "$VERIFIER" --stats_rules 0
abl noexpand   --llm "$VERIFIER" --expand 0
abl nowrapped  --llm "$VERIFIER" --wrapped 0
abl smalldom0  --llm "$VERIFIER" --small_dom 0
abl norestrict --llm "$VERIFIER" --restrict 0
for tau in 03 07 09; do abl tau$tau --llm "$VERIFIER" --tau 0.${tau#0}; done
abl q4         --llm "$VERIFIER_Q3_4B"
abl q32        --llm "$VERIFIER_32B"

# verifier models on held-out seed 3
for T in omni7b qwen7b; do
  rewrite main_r20_s3 $T ours ours_v4_q14      --llm "$VERIFIER_14B"
  rewrite main_r20_s3 $T ours ours_v4_mistral7 --llm "$VERIFIER_MISTRAL"
  rewrite main_r20_s3 $T ours ours_v4_llama8   --llm "$VERIFIER_LLAMA"
done
