# CLAUDE.md

Operational guide for running PreHypo locally and for the MultiHop-RAG benchmark
work. Architecture/results live in `README.md`; this file captures the
run-environment knowledge that isn't obvious from the code.

## Pipeline at a glance

```
python main.py --mode {index|benchmark|benchmark_all|ocr} \
  --strategy {prehypo|naive|hoprag|ms_graphrag} \
  --dataset <corpus_dir> --queries_file <queries.json> \
  --corpus-tag <tag> --model generation-model
```

- `benchmark_all` loops strategies `["naive","prehypo","hoprag","ms_graphrag"]`.
- prehypo / naive / hoprag store their graphs in **Neo4j** (label-prefixed by
  corpus tag). ms_graphrag writes **parquet** under
  `data/ms_graphrag_output/<tag>/` (not Neo4j).
- **Never pass `--clear-graph` on a resume/re-run** — it wipes ALL Neo4j data,
  including the other baselines' indexes.

## Setup (uv-managed, reproducible for a fresh clone)

The Python env is managed with **uv**. `pyproject.toml` is the canonical
dependency list (there is no `requirements.txt`).

```bash
uv venv --python 3.12 .venv               # .python-version pins 3.12
VIRTUAL_ENV=.venv uv pip install -e .      # installs vllm/torch/.../flashinfer
cp .env.example .env                        # set NEO4J_PASSWORD, OPENAI_API_KEY
.venv/bin/python -m spacy download en_core_web_sm   # required by the hoprag baseline
```

- This venv has **no `pip`** (uv-managed). Check installs with
  `VIRTUAL_ENV=.venv uv pip show <pkg>` or `.venv/bin/python -c "import pkg"` —
  `.venv/bin/pip` does not exist and gives false negatives.
- The flat layout needs `[tool.setuptools] py-modules = []` (already set) or the
  editable build aborts with "Multiple top-level packages".
- The run scripts do **not** require activating the venv: a shared
  `resolve_python` helper (`scripts/lib.sh`) finds `.venv/bin/python`
  automatically (override with `PYTHON_BIN`, falls back to system python). So a
  fresh clone runs `./run_*.sh` directly after `uv venv && uv pip install -e .`.

## Dataset run scripts (per-dataset entrypoints)

Two wrappers pin each dataset's corpus/queries/tag so you don't repeat long
arg strings. Both delegate to `run_index.sh` / `run_benchmark.sh` and run each
strategy separately (not `benchmark_all`); extra flags pass through.

```bash
# FinanceBench — per-strategy tags <strategy>_full (matches run_all_*.sh).
./run_financebench.sh ocr                      # OCR PDFs -> Markdown corpus
./run_financebench.sh index   [--model all|<strategy>]
./run_financebench.sh benchmark
./run_financebench.sh all                      # index + benchmark (OCR is separate)

# MultiHop-RAG — plain text (no OCR); shared dataset tag `multihoprag`
# (strategy is already separated by the PR_/NA_/HO_ label prefix).
./run_multihoprag.sh all                            # index all 4 + benchmark (sample100)
./run_multihoprag.sh index     [--model all|<strategy>]
./run_multihoprag.sh benchmark --queries full       # smoke|sample100|full
```

The legacy `run_all_indexing_parallel.sh` / `run_all_benchmark_parallel.sh`
remain for the FinanceBench paper matrix (ablation variants with their own
tags); the per-dataset scripts above are the simpler everyday entrypoints.

## Local single-GPU run environment (RTX 5000 Ada, 32 GB)

`run_servers.sh` defaults to a 2-GPU layout but the per-service GPU is now
env-configurable — on a single GPU run
`GEN_GPU=0 EMBED_GPU=0 RERANK_GPU=0 OCR_GPU=0 ./run_servers.sh all` (watch total
`--gpu-memory-utilization` when co-locating). The manual launch below is
equivalent and what was validated on this box. Attention backend:
`flashinfer-python` IS installed (transitive dep of
vllm, 0.6.8.post1), but vLLM defaults attention to FLASH_ATTN/FlashAttention 2
when `--attention-backend` is unset (FLASHINFER is listed as available; sampling
already uses FlashInfer). To use FlashInfer for attention too, pass
`--attention-backend FLASHINFER` (requires a server restart to take effect).
FLASH_ATTN works fine as-is. torch 2.11.0+cu130, system CUDA 13.2 — compatible.
Validated layout:

| Port  | Model                      | served-name      | util | max-len | role           |
|-------|----------------------------|------------------|------|---------|----------------|
| 28000 | Qwen/Qwen3-4B-Instruct-2507| generation-model | 0.45 | 16384   | generation     |
| 18082 | Qwen/Qwen3-Embedding-0.6B  | embedding-model  | 0.15 | 8192    | embeddings     |
| 18083 | Qwen/Qwen3-Reranker-0.6B   | reranker-model   | 0.20 | 4096    | reranking      |

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name generation-model --host 0.0.0.0 --port 28000 \
  --gpu-memory-utilization 0.45 --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code > logs/vllm_gen.log 2>&1 &
# embed (18082, util 0.15) and rerank (18083, util 0.20) similarly.
```

Notes:
- gen + embed + rerank ≈ 0.80 util fits in 32 GB (≈ 26.8 GiB observed). Start
  servers sequentially (KV-cache contention can kill a second small model that
  comes up simultaneously).
- **Indexing needs only gen + embed**; rerank is used by the *benchmark* path
  (and prehypo HOP-edge construction). For long indexes, leave rerank down to
  free GPU, then bring it up before benchmarking.
- `.env` has `VLLM_URL_2=http://localhost:28010/v1` (2nd gen) which doesn't
  exist on a single GPU → prefix commands with `VLLM_URL_2=` (empty) to disable
  round-robin to the dead endpoint. `load_dotenv` does not override existing
  env, so this works.
- venv: flat layout needs `[tool.setuptools] py-modules = []` in pyproject for
  the editable build to pass. `pdf2image`/`pillow` are required (OCR import).

## Neo4j

- `bash run_servers.sh neo4j` → docker `prehypo-neo4j` (neo4j:5-community, bolt
  7687, http 7474, creds `neo4j/1q2w3e4r`).
- **Data persists across reboots**: bind-mounted host dir `neo4j_data/` → /data.
  After a reboot just `docker start prehypo-neo4j` — prehypo/naive/hoprag data
  is intact. Verify before re-indexing.
- Query from CLI:
  `docker exec prehypo-neo4j cypher-shell -u neo4j -p 1q2w3e4r --format plain "<cypher>"`

## Reboot / interruption recovery

1. `docker start prehypo-neo4j`; verify labels survived:
   `CALL db.labels()` + per-label counts.
2. Relaunch vLLM gen + embed (see above).
3. Re-run any unfinished index. **prehypo/naive/hoprag** are durable in Neo4j —
   if a strategy's node counts look complete, skip it. **ms_graphrag** resumes
   from its own cache (`data/ms_graphrag_output/<tag>/_cache/extract_graph/`):
   re-run the identical index command and it replays cached chunks instantly,
   then continues from the break. No `--clear-graph`.

## MultiHop-RAG benchmark

Genuine dataset = yixuantt/MultiHop-RAG (609 articles, 2556 queries;
question_type: comparison 856 / inference 816 / temporal 583 / null 301).

- Prep: `python data/prepare_multihoprag.py`. **GitHub LFS gotcha**: must use
  `media.githubusercontent.com/media/...` (raw endpoint returns LFS pointer
  text). Titles carry HTML entities (`&#039;`) → `html.unescape` applied.
- Files: `data/multihoprag_corpus/` (txt input dir), `multihoprag_corpus.json`,
  `multihoprag_queries.json` (full 2556), `multihoprag_sample100_queries.json`
  (balanced 25×4). Dataset is detected via the per-query `dataset:"multihoprag"`
  marker (`cli/benchmark.py`), NOT the filename.
- **Domain branching**: `RAGConfig.DOMAIN` = financial|news; multihoprag→news.
  `main.py` auto-detects from `--dataset`/`--queries_file` marker and sets
  `RAG_DOMAIN` env *before* heavy imports (prompts pick their variant at import
  time). News variants exist for HOPRAG_PROMPT, QUERY_REWRITE_PROMPT,
  RERANKER_INSTRUCTION, SEARCH_CONTINUATION_PROMPT, and `answer_role()`.
- Metrics (MultiHop-RAG official): retrieval fact-level MRR@10/MAP@10/Hits@10/4
  (gold=`evidence_list[].fact`); generation = LLM-judge accuracy, broken out by
  question_type. null queries handled by abstain logic.
- Run: `VLLM_URL_2= python main.py --mode benchmark_all --queries_file
  data/multihoprag_sample100_queries.json --model generation-model
  --corpus-tag multihoprag`. Judge uses OpenAI `EVAL_MODEL` (key in `.env`).
- **Judge via OpenAI Batch API** (opt-in, 50% cheaper): set `RAG_JUDGE_BATCH=true`
  with an OpenAI `EVAL_MODEL`. The benchmark collects all judge prompts during
  the pass, submits ONE batch to `/v1/chat/completions`, polls until done (no
  client timeout — up to the 24h batch SLA), then patches scores and recomputes
  the 3-way labels. Any batch failure falls back to the synchronous per-query
  judge so scores are never silently dropped. Default off = synchronous judge.
  Code: `utils/batch_judge.py` (collector/runner), `utils/metrics.py`
  (`_resolve_judge_fields` shared by both paths), `cli/benchmark.py` (phase-2).

## Neo4j data layout & integrity checks

Corpus-tag prefixes labels so datasets coexist: `PR_<tag>_*` (prehypo),
`NA_<tag>_*` (naive), `HO_<tag>` (hoprag).

prehypo schema (written by `models/prehypo/indexing/graph_writer.py`):
- `PR_<tag>_Document {filename, corpus, title, summary, updated_at}`
- `PR_<tag>_Chunk {id, text, chunk_summary, q_minus_text, q_plus_text,
  embedding, body_embedding, q_minus_embedding, q_plus_embedding}`
- Relationships: `NEXT` (chunk sequence; edges = chunks − docs),
  `HOP` (rank-based, §3.1.4, MERGE-idempotent)
- Indexes: 3 vector (body/qminus/qplus) + 3 fulltext + range on id/filename

By-design partial coverage (NOT incomplete indexing):
- `q_minus_embedding` < chunks: empty Q⁻ falls back to body embedding for the
  primary `embedding` (graph_writer.py).
- `q_plus_embedding` < chunks: Q⁺ passes `_is_high_quality_q_plus` gating; the
  surviving count equals the "wrote N HOP edges over M Q+ chunks" log line.

Useful integrity probes:
```cypher
// no duplicate documents
MATCH (d:PR_<tag>_Document) RETURN count(d), count(DISTINCT d.title);
// hoprag edges are unique (start,end,question) — multi-question, not dupes
MATCH (a:HO_<tag>)-[r:HO_<tag>_p2a]->(b:HO_<tag>)
RETURN count(r), count(DISTINCT [elementId(a),elementId(b),r.question]);
// prehypo property coverage
MATCH (c:PR_<tag>_Chunk) RETURN count(c), count(c.embedding),
  count(c.q_minus_embedding), count(c.q_plus_embedding);
```
