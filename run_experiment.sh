#!/usr/bin/env bash
# One unseeded dream/fixed pair, the simplest possible comparison.
# This is how runs/dream_r4 and runs/fixed_r4 were produced. For anything new prefer run_matrix.sh, which
# runs several seeds per arm and is resumable.
#
#   ./run_experiment.sh [model]
#   TASK=binpack ./run_experiment.sh
set -uo pipefail
cd "$(dirname "$0")"

MODEL=${1:-${DREAM_RSI_MODEL:-qwen2.5-coder:7b}}
TASK=${TASK:-cache}

mkdir -p runs
for mode in dream fixed; do
  uv run python -m dream_rsi.run --mode "$mode" --task "$TASK" --rounds 4 \
    --model "$MODEL" --name "${mode}_r4" 2>&1 | tee "runs/${mode}_r4.out"
done
uv run python -m dream_rsi.report dream_r4 fixed_r4 > runs/report.md
echo "done: runs/report.md"
