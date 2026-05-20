# PreHypo: Indexing-Time Structure Preservation for Page-Grounded Financial QA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference implementation of **PreHypo**, a GraphRAG framework whose entire advantage on FinanceBench comes from indexing-time structure preservation. The query path is a thin two-stage hybrid retrieve over a graph built once offline — no agent loop, no reflection, no refinement.

---

## What this repository is

Four indexing-time designs, evaluated on [FinanceBench](https://github.com/patronus-ai/financebench):

1. **Topology-Preserving VLM OCR** — financial tables rendered as Markdown so row-column structure survives into the LLM context.
2. **Adaptive Context-Aware Chunking** — page-cluster boundaries respected, with a rolling header (anchor + milestone summaries + prev-group summary) attached to each chunk embedding.
3. **Predictive Knowledge Mapping** — every chunk receives dual hypothetical-query annotations ($Q^-$ for self-contained facts, $Q^+$ for outgoing dependencies), indexed separately.
4. **Rank-Based HOP Edges Pre-Built Offline** — chunk-to-chunk semantic edges are computed once via a cross-encoder reranker; the query path never expands the graph.

The query path is deliberately thin: two-stage hybrid retrieve (Q⁻/body, then Q⁺ expansion), rerank with top-up, deterministic 1-hop traversal over the pre-built NEXT/HOP edges, and a single LLM synthesis call with inline citations.

---

## Main Results (FinanceBench, n=150, 5-fold mean ± std)

| System | Judge ↑ | J\|Att ↑ | Halluc ↓ | H\|Att ↓ | **PgM ↑** | Att | Lat (s) |
|---|---|---|---|---|---|---|---|
| **PreHypo (ours)** | **0.32 ± 0.11** | **0.57 ± 0.09** | 0.17 ± 0.04 | 0.31 ± 0.08 | **0.19 ± 0.07** | 0.55 ± 0.10 | 6.81 |
| MS-GraphRAG | 0.19 ± 0.10 | 0.23 ± 0.10 | 0.32 ± 0.05 | 0.40 ± 0.11 | 0.03 ± 0.04 | 0.80 ± 0.08 | 26.4 |
| HopRAG | 0.19 ± 0.05 | 0.39 ± 0.05 | 0.13 ± 0.06 | 0.26 ± 0.04 | 0.04 ± 0.04 | 0.50 ± 0.11 | 163.2 |
| Naive | 0.05 ± 0.04 | 0.10 ± 0.05 | 0.04 ± 0.04 | 0.06 ± 0.04 | 0.00 ± 0.00 | 0.59 ± 0.08 | 2.1 |

`J|Att` / `H|Att` are conditional metrics over attempted queries (substantive response, not abstention). DocM (gold document in retrieved set) is ≥ 0.89 for all four systems — PreHypo/HopRAG 0.99, MS-GraphRAG 0.95, Naive 0.89 — and is omitted from the table.

Generator: Qwen3-4B-Instruct-2507 across every system. The defensible win is **PgM 0.19 — the largest gap relative to fold variance (vs. ≤ 0.04 for every baseline)**; the absolute level is low, since most correct answers are synthesized across sibling pages rather than from the single gold page. The Judge gap (13pp) overlaps with baselines per fold but is resolved under query-level paired bootstrap with Bonferroni correction (all three baseline pairings above zero). Naive's nominally low Halluc reflects an answered set of which only ~10% are correct, not faithfulness.

---

## Repository layout

```
prehypo/
├── main.py                          # single CLI entry point (--mode index|benchmark|ocr)
├── cli/
│   ├── index.py                     # indexing runner
│   └── benchmark.py                 # benchmark runner (single + multi-seed)
├── core/
│   ├── config.py                    # RAGConfig — env-driven thresholds
│   ├── neo4j_service.py             # async Neo4j driver lifecycle
│   ├── vllm_client.py               # vLLM + OpenAI routing
│   └── schemas.py
├── models/
│   ├── prehypo/                     # the paper's system
│   │   ├── graphrag.py              # GraphRAG facade; run_workflow() is the query entry point
│   │   ├── indexing/                # §3.1 — ocr, chunking, knowledge_mapping (Q-/Q+), hop_edges, graph_writer
│   │   ├── retrieval/               # §3.2 — hybrid (RRF), rerank, traversal, retrieve, rewrite, text_utils
│   │   └── schemas.py / state.py / trace.py
│   ├── naive/                       # baseline (sentence chunking + vector search)
│   ├── hoprag/                      # baseline (runtime hop traversal via official HopRAG)
│   └── ms_graphrag/                 # baseline (community-report retrieval via graphrag package)
├── utils/
│   ├── abstain.py                   # honest-abstain detection (3-way FinanceBench taxonomy)
│   ├── metrics.py                   # combined judge + hallucination call
│   ├── prompts/                     # indexing + retrieval + judge prompts
│   └── io.py / formatters.py / parsers.py / reporting.py / ocr_tools.py / tool_definitions.py
├── tools/
│   ├── benchmark_report.py
│   └── bootstrap_ci.py
├── scripts/                         # port-probe, env-check, helpers
├── tests/                           # chunking / retrieval / rolling-summary / live-integration
├── run_servers.sh                   # start Neo4j + vLLM (gen / gen2 / ocr / embed / rerank)
├── run_index.sh / run_benchmark.sh / run_ocr.sh
├── run_all_indexing_parallel.sh     # full paper indexing matrix
├── run_all_benchmark_parallel.sh    # full paper benchmark matrix
├── pyproject.toml / requirements.txt
└── README.md
```

---

## Installation

```bash
# Python 3.12+ recommended; see .python-version
pip install -e .         # installs vllm, torch, transformers, neo4j, ... per pyproject.toml
# or
uv pip install -e .

# Neo4j 5.x — Docker is simplest:
docker run -d --name prehypo-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> neo4j:5

# Configure env vars
cp .env.example .env
# Required: NEO4J_PASSWORD, OPENAI_API_KEY (for the LLM judge)
```

`requirements.txt` covers only the runtime client (Neo4j driver, OpenAI client, pytest); `pyproject.toml` is the canonical dependency list and includes the vLLM / Torch / transformers stack needed by `run_servers.sh` to serve the local generation, embedding, and reranker models.

vLLM servers are launched by `run_servers.sh` (separate processes for generation, embeddings, and reranker; ports listed in `scripts/probe_ports.py`).

---

## Quick start

```bash
# 0) Download FinanceBench (questions + document metadata + SEC PDFs).
#    Use --download-pdfs to fetch the 150 source PDFs into data/finance_pdfs/.
python3 data/prepare_financebench.py --download-pdfs
# Produces:
#   data/financebench_open_source.jsonl
#   data/financebench_document_information.jsonl
#   data/financebench_queries.json
#   data/finance_pdfs/*.pdf

# 1) Start services (Neo4j + vLLM gen / embed / rerank / ocr)
./run_servers.sh all

# 2) OCR the PDFs into a Markdown corpus
./run_ocr.sh --convert_tables

# 3) Build the PreHypo index
./run_index.sh --model prehypo --corpus-tag prehypo_full

# 4) Benchmark
./run_benchmark.sh --model prehypo --corpus-tag prehypo_full

# 5) Stop services
./stop_servers.sh all
```

Result JSON is written to `data/results/<timestamp>/prehypo/<corpus_tag>/*.json` and includes per-query details, category breakdowns, the FinanceBench 3-way label (Correct / Incorrect / Refusal), and aggregate metrics.

---

## Reproducing the paper experiments

```bash
# Index all four systems on the full corpus
./run_all_indexing_parallel.sh --full

# Run the benchmark matrix
./run_all_benchmark_parallel.sh --full
```

PreHypo ablations (single indexing component disabled per run) are driven by environment toggles read in `core/config.py`:

| Variable | Default | Effect when set to `false` |
|---|---|---|
| `RAG_ABLATION_TABLE` | `true` | Tables left as raw OCR text instead of Markdown |
| `RAG_ABLATION_CHUNKING` | `true` | Fixed-length chunking instead of adaptive |
| `RAG_ABLATION_SUMMARY` | `true` | Rolling context header removed |
| `RAG_ABLATION_Q_PLUS` | `true` | Stage 2 Q⁺ expansion disabled |
| `RAG_ABLATION_Q_MINUS` | `true` | Stage 1 Q⁻ channel disabled |

Each ablation lives under its own corpus tag (e.g. `prehypo_full_no_chunk`) so indexed graphs never collide.

---

## Key hyperparameters

Full list in the paper appendix; the most important:

| Parameter | Value | Where |
|---|---|---|
| `τ_page` | 0.5 | adaptive chunking (page-cluster threshold) |
| `τ_chunk` | 0.65 | adaptive chunking (sentence cohesion threshold) |
| `τ_r` | 0.4 | reranker threshold (used identically in HOP build and query rerank) |
| `L_hop` | 5 | max outgoing HOP edges per source chunk |
| `K_hop` | 15 | HOP candidate pool per source chunk |
| Stage 1 weights | 0.7 / 0.3 | $Q^-$ / body |
| Stage 2 weights | 0.6 / 0.4 | $Q^+$ / $Q^-$ support |
| RRF `k` | 60 | $w_v=1.3$, $w_t=1.0$ |
| Embedding dim | 1024 | Qwen3-Embedding-0.6B |

---

## What is intentionally **not** in this repository

The system the paper analyzes does not include a reflective agent loop. An earlier draft of this work explored a five-stage Perception/Planning/Execution/Reflection/Refinement loop on top of the same indexing pipeline; in our final measurements the loop was net-negative on PreHypo itself and at best baseline-dependent across the four GraphRAG systems we tried. The paper reports the retrieval-only configuration and the code here mirrors that decision.

---

## Citation

```bibtex
@article{prehypo2026,
  title   = {{PreHypo}: Indexing-Time Structure Preservation for Page-Grounded Financial QA},
  author  = {Anonymous},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
