#!/bin/bash
#
# run_multihoprag.sh — run the MultiHop-RAG dataset end to end.
#
# Thin wrapper over run_index.sh / run_benchmark.sh that pins the MultiHop-RAG
# corpus, queries, and corpus tag. MultiHop-RAG articles are plain text, so
# there is no OCR stage.
#
# Tagging: all four strategies share one dataset-level corpus tag (`multihoprag`)
# because the strategy is already encoded in the Neo4j label prefix
# (PR_/NA_/HO_) and the ms_graphrag parquet path. Benchmarks are still run
# per strategy. (The --queries smoke set uses its own mini corpus + tag.)
#
# Usage:
#   ./run_multihoprag.sh all                          # index all 4 + benchmark (sample100)
#   ./run_multihoprag.sh index                        # index all 4 strategies
#   ./run_multihoprag.sh benchmark --queries full     # benchmark all on full 2556
#   ./run_multihoprag.sh index --model prehypo         # one strategy only
#
# Options:
#   --model   {all|prehypo|naive|hoprag|ms_graphrag}   default: all
#   --queries {smoke|sample100|full}                   default: sample100
# Any other flags are forwarded to the underlying run_*.sh (e.g. --clear-graph,
# --skip-server).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"

STRATEGIES=(prehypo naive hoprag ms_graphrag)

STAGE="${1:-all}"; shift || true
MODEL="all"
QUERIES="sample100"
PASS=()
while [ $# -gt 0 ]; do
    case $1 in
        --model)   MODEL="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

# Map the query-set selector to its corpus dir, corpus tag, and queries file.
case "$QUERIES" in
    smoke)     CORPUS_DIR="data/multihoprag_smoke_corpus"; CORPUS_TAG="multihoprag_smoke"; QUERIES_FILE="data/multihoprag_smoke_queries.json" ;;
    sample100) CORPUS_DIR="data/multihoprag_corpus";       CORPUS_TAG="multihoprag";       QUERIES_FILE="data/multihoprag_sample100_queries.json" ;;
    full)      CORPUS_DIR="data/multihoprag_corpus";       CORPUS_TAG="multihoprag";       QUERIES_FILE="data/multihoprag_queries.json" ;;
    *) echo "Unknown --queries '$QUERIES' (use smoke|sample100|full)"; exit 1 ;;
esac

models_to_run() {
    if [ "$MODEL" = "all" ]; then printf '%s\n' "${STRATEGIES[@]}"; else echo "$MODEL"; fi
}

do_index() {
    for m in $(models_to_run); do
        echo ">>> [MultiHop-RAG index] $m  (dataset $CORPUS_DIR, corpus-tag $CORPUS_TAG)"
        ./run_index.sh --model "$m" --dataset "$CORPUS_DIR" --corpus-tag "$CORPUS_TAG" "${PASS[@]}"
    done
}

do_benchmark() {
    for m in $(models_to_run); do
        echo ">>> [MultiHop-RAG benchmark] $m  (queries $QUERIES_FILE, corpus-tag $CORPUS_TAG)"
        ./run_benchmark.sh --model "$m" --queries "$QUERIES_FILE" --corpus-tag "$CORPUS_TAG" "${PASS[@]}"
    done
}

case "$STAGE" in
    index)           do_index ;;
    benchmark|bench) do_benchmark ;;
    all)             do_index; do_benchmark ;;
    *) echo "Usage: $0 <index|benchmark|all> [--model all|<strategy>] [--queries smoke|sample100|full] [extra run_*.sh flags]"; exit 1 ;;
esac
