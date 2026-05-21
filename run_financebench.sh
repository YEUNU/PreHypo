#!/bin/bash
#
# run_financebench.sh — run the FinanceBench dataset end to end.
#
# Thin wrapper over run_ocr.sh / run_index.sh / run_benchmark.sh that pins the
# FinanceBench corpus, queries, and per-strategy corpus tags (<strategy>_full,
# matching run_all_*.sh) so you don't repeat them. Each strategy is indexed and
# benchmarked under its own tag.
#
# Usage:
#   ./run_financebench.sh ocr                      # OCR the PDFs into a corpus
#   ./run_financebench.sh index                    # index all 4 strategies
#   ./run_financebench.sh benchmark                # benchmark all 4 strategies
#   ./run_financebench.sh all                      # index + benchmark (not OCR)
#   ./run_financebench.sh index --model prehypo    # one strategy only
#
# Options:
#   --model {all|prehypo|naive|hoprag|ms_graphrag}   default: all
# Any other flags are forwarded to the underlying run_*.sh (e.g. --sample, --n,
# --clear-graph, --skip-server).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"

QUERIES_FILE="data/financebench_queries.json"
STRATEGIES=(prehypo naive hoprag ms_graphrag)

STAGE="${1:-all}"; shift || true
MODEL="all"
PASS=()
while [ $# -gt 0 ]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

models_to_run() {
    if [ "$MODEL" = "all" ]; then printf '%s\n' "${STRATEGIES[@]}"; else echo "$MODEL"; fi
}

do_ocr() {
    echo ">>> [FinanceBench] OCR (tables -> Markdown)"
    ./run_ocr.sh --convert_tables "${PASS[@]}"
}

do_index() {
    for m in $(models_to_run); do
        echo ">>> [FinanceBench index] $m  (corpus-tag ${m}_full)"
        ./run_index.sh --model "$m" --corpus-tag "${m}_full" "${PASS[@]}"
    done
}

do_benchmark() {
    for m in $(models_to_run); do
        echo ">>> [FinanceBench benchmark] $m  (corpus-tag ${m}_full)"
        ./run_benchmark.sh --model "$m" --queries "$QUERIES_FILE" --corpus-tag "${m}_full" "${PASS[@]}"
    done
}

case "$STAGE" in
    ocr)             do_ocr ;;
    index)           do_index ;;
    benchmark|bench) do_benchmark ;;
    all)             do_index; do_benchmark ;;
    *) echo "Usage: $0 <ocr|index|benchmark|all> [--model all|<strategy>] [extra run_*.sh flags]"; exit 1 ;;
esac
