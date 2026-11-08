# Paths and models used by all scripts. Source this file; override any variable by exporting it first.
CANDOR_HOME=${CANDOR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export CANDOR_HOME
export CANDOR_BIRD_DEV=${CANDOR_BIRD_DEV:-$CANDOR_HOME/data/bird/dev_20240627}   # original BIRD dev set
export CANDOR_DIRTY=${CANDOR_DIRTY:-$CANDOR_HOME/data/bird_dirty}                 # BIRD-Dirty roots
export CANDOR_RUNS=${CANDOR_RUNS:-$CANDOR_HOME/runs}                              # run outputs
export CANDOR_TRUTH=${CANDOR_TRUTH:-$CANDOR_HOME/data/truth_clean.pkl}            # cache of gold results on clean DBs
export CANDOR_TABLES=${CANDOR_TABLES:-$CANDOR_HOME/results/tables}
export OMNISQL_DIR=${OMNISQL_DIR:-$CANDOR_HOME/third_party/OmniSQL/train_and_evaluate}
export TOKENIZERS_PARALLELISM=false
SRC=$CANDOR_HOME/src

# verifier LLMs (HF ids or local paths)
VERIFIER=${VERIFIER:-Qwen/Qwen2.5-7B-Instruct}            # default verifier of CANDOR
VERIFIER_32B=${VERIFIER_32B:-Qwen/Qwen2.5-32B-Instruct-AWQ}
VERIFIER_14B=${VERIFIER_14B:-Qwen/Qwen2.5-14B-Instruct}
VERIFIER_MISTRAL=${VERIFIER_MISTRAL:-mistralai/Mistral-7B-Instruct-v0.3}
VERIFIER_LLAMA=${VERIFIER_LLAMA:-meta-llama/Llama-3.1-8B-Instruct}
VERIFIER_Q3_4B=${VERIFIER_Q3_4B:-Qwen/Qwen3-4B-Instruct-2507}

# text-to-SQL generators: tag -> model
declare -A GENERATOR=(
  [omni7b]=${GEN_OMNI7B:-seeklhy/OmniSQL-7B}
  [omni14b]=${GEN_OMNI14B:-seeklhy/OmniSQL-14B}
  [qwen7b]=${GEN_QWEN7B:-Qwen/Qwen2.5-7B-Instruct}
  [qwen32]=${GEN_QWEN32:-Qwen/Qwen2.5-32B-Instruct-AWQ}
  [llama8]=${GEN_LLAMA8:-meta-llama/Llama-3.1-8B-Instruct}
)

root_of() { if [ "$1" = clean ]; then echo "$CANDOR_BIRD_DEV"; else echo "$CANDOR_DIRTY/$1"; fi; }

# rewrite one prediction file with one method and evaluate it.
# usage: rewrite ROOTNAME TAG MODE NAME [extra run_varfix.py args]   -> $CANDOR_RUNS/ROOTNAME/TAG_NAME.{json,eval.json}
rewrite() {
  local RN=$1 T=$2 M=$3 N=$4; shift 4
  local R; R=$(root_of "$RN"); local O=$CANDOR_RUNS/$RN; mkdir -p "$O"
  local f=$O/${T}_$N.json
  [ -f "$f" ] || python "$SRC/run_varfix.py" --root "$R" --pred "$R/pred_$T.json" --mode "$M" --out "$f" "$@" > "$O/${T}_$N.log" 2>&1
  python "$SRC/evaluate.py" --root "$R" --pred "$f" --out "$O/${T}_$N.eval.json" | tail -n 1 | sed "s/^/$RN $T $N /"
}
