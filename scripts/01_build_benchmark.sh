#!/bin/bash
# Rebuild every BIRD-Dirty configuration used in the paper from the released cell-level deltas.
# Needs the original BIRD dev set (dev.json, dev_tables.json, dev_databases/) at $CANDOR_BIRD_DEV.
set -e
source "$(dirname "$0")/env.sh"
[ -f "$CANDOR_BIRD_DEV/dev.json" ] || { echo "BIRD dev set not found at $CANDOR_BIRD_DEV (see README)"; exit 1; }
for d in "$CANDOR_HOME"/data/deltas/*/; do
  name=$(basename "$d"); R=$CANDOR_DIRTY/$name
  [ "$name" = main_r20_s0 ] && R=$CANDOR_DIRTY/main_r20          # development seed keeps its original root name
  if [ -f "$R/canon_map.json" ]; then echo "exists: $R"; continue; fi
  mkdir -p "$R"
  python "$SRC/bench_delta.py" apply "$d/delta.jsonl.gz" "$R"
  gunzip -c "$d/canon_map.json.gz" > "$R/canon_map.json"
  cp "$CANDOR_BIRD_DEV/dev.json" "$CANDOR_BIRD_DEV/dev_tables.json" "$R/"
  python "$SRC/bench_delta.py" hash "$R" | sed "s|^|$(basename "$R") |"
done
python "$SRC/bench_stats.py" "$CANDOR_DIRTY" "$CANDOR_RUNS/bench_stats.json"
