#!/usr/bin/env bash
# Reproduces every number, table and figure of the paper from the feature tables in data/ (about 1 minute).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
python src/unify.py data results/pooled.csv
python src/pooled.py results/pooled.csv results
python src/cost2.py results/pooled.json results/cost_per_bearing.json
python src/figs2.py results/pooled.csv results/pooled.json figures
echo "Done: results/ and figures/"
