#!/bin/bash
# Main comparison (Tables "main" and "seeds"): every generator x held-out seed, rewritten by No rewrite, LIKE substring,
# Norm, Fuzzy, CANDOR and oracle variant sets; the same methods (without the oracle) on the original databases.
source "$(dirname "$0")/env.sh"
for RN in main_r20_s2 main_r20_s3 main_r20_s4 clean; do
  for T in omni7b qwen7b qwen32 omni14b llama8; do
    for M in none like norm fuzzy; do rewrite $RN $T $M $M; done
    [ $RN = clean ] || rewrite $RN $T oracle oracle
    rewrite $RN $T ours ours_v4 --llm "$VERIFIER"
  done
done
# CANDOR with the 32B verifier
for RN in main_r20_s3 clean; do for T in omni7b qwen7b; do
  rewrite $RN $T ours ours_v4_q32 --llm "$VERIFIER_32B"
done; done
