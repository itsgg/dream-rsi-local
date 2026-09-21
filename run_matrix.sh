#!/usr/bin/env bash
# Seeded matrix: one run per (seed, mode) cell, then a report over all of them.
#
#   ./run_matrix.sh [model] [rounds] [seeds...]
#   TASK=binpack ./run_matrix.sh qwen2.5-coder:7b 4 1 2 3
#
# Defaults: model qwen2.5-coder:7b, 4 rounds, seeds 1 2 3, task cache.
# Resumable: a cell whose summary.json already has the requested number of rounds is skipped, so it is safe
# to re-run after an interruption. Progress goes to the console and to runs/<name>.out at the same time.
set -uo pipefail
cd "$(dirname "$0")"

MODEL=${1:-${DREAM_RSI_MODEL:-qwen2.5-coder:7b}}
ROUNDS=${2:-4}
if [ "$#" -gt 2 ]; then shift 2; SEEDS=("$@"); else SEEDS=(1 2 3); fi
TASK=${TASK:-cache}
TAG=$(printf '%s' "${TASK}_${MODEL}" | tr ':.' '__')

mkdir -p runs
for s in "${SEEDS[@]}"; do
  for mode in dream fixed; do
    name="${mode}_${TAG}_s${s}"
    done_rounds=0
    if [ -f "runs/$name/summary.json" ]; then
      done_rounds=$(uv run python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))['rounds']))" "runs/$name/summary.json")
    fi
    if [ "$done_rounds" = "$ROUNDS" ]; then
      echo "== skip $name (already has $ROUNDS rounds)"
      continue
    fi
    echo "== $name"
    uv run python -m dream_rsi.run --mode "$mode" --task "$TASK" --rounds "$ROUNDS" \
      --model "$MODEL" --seed "$s" --name "$name" 2>&1 | tee "runs/$name.out"
  done
done

reports=()
for s in "${SEEDS[@]}"; do reports+=("dream_${TAG}_s${s}" "fixed_${TAG}_s${s}"); done
# stderr deliberately stays on the console: a crashing report must not write its traceback into the report.
uv run python -m dream_rsi.report "${reports[@]}" > "runs/report_${TAG}.md"
echo ALL_DONE > "runs/ALL_DONE_${TAG}"
echo "== done: runs/report_${TAG}.md"
