#!/bin/bash
# Significance tests and LaTeX tables from the run outputs (no hand-typed numbers).
source "$(dirname "$0")/env.sh"
cd "$SRC"
python sig_tests.py "$CANDOR_RUNS/sig_all.json" main_r20_s2 main_r20_s3 main_r20_s4
python make_tables.py
