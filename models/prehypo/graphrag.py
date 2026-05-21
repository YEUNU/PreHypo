"""GraphRAG facade composing the offline indexing pipeline (paper §3.1) and
the query-time retrieval pipeline (paper §3.2.3).

All Neo4j labels and index names are derived from (strategy, corpus_tag) so
multiple corpora and strategies coexist in the same database without
collision.
"""
import asyncio
import logging
import re
from typing import Any, Optional

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from models.prehypo.indexing import IndexingPipeline
from models.prehypo.retrieval import RetrievalPipeline
from utils.prompts.shared import answer_role


_ANSWER_PREFIX = "@@ANSWER:"


logger = logging.getLogger(__name__)


class GraphRAG(IndexingPipeline, RetrievalPipeline):
    def __init__(
        self,
        strategy: str = "prehypo",
        indexing_model_id: Optional[str] = None,
        corpus_tag: Optional[str] = None,
        save_intermediate: bool = False,
    ):
        self.strategy = strategy.lower()
        self.corpus_tag = corpus_tag or "default"
        self.prefix = self.strategy[:2].upper() + "_"
        self._safe_corpus = re.sub(r"[^A-Za-z0-9_]", "_", self.corpus_tag)

        self.chunk_label = f"{self.prefix}{self._safe_corpus}_Chunk"
        self.doc_label = f"{self.prefix}{self._safe_corpus}_Document"

        self.body_vector_index = f"{self.strategy}_{self._safe_corpus}_vector_idx"
        self.body_text_index = f"{self.strategy}_{self._safe_corpus}_text_idx"
        self.q_minus_vector_index = f"{self.strategy}_{self._safe_corpus}_qminus_vector_idx"
        self.q_plus_vector_index = f"{self.strategy}_{self._safe_corpus}_qplus_vector_idx"
        self.q_minus_text_index = f"{self.strategy}_{self._safe_corpus}_qminus_text_idx"
        self.q_plus_text_index = f"{self.strategy}_{self._safe_corpus}_qplus_text_idx"
        self.vector_index = self.body_vector_index
        self.text_index = self.body_text_index

        self.neo4j = Neo4jService()
        self.llm = VLLMClient()
        self._index_ready = False

        indexing_model_id = indexing_model_id or RAGConfig.DEFAULT_MODEL
        self.indexing_llm = get_llm_client(indexing_model_id)

        self.hop_threshold = RAGConfig.HOP_THRESHOLD
        self.similarity_threshold = RAGConfig.SIMILARITY_THRESHOLD
        self.max_retries = RAGConfig.RETRY_COUNT
        self.vector_dimensions = RAGConfig.EMBEDDING_DIMENSIONS
        self._pending_batch = []
        self._batch_lock = asyncio.Lock()
        self._index_setup_lock = asyncio.Lock()
        self.debug_output_dir = f"data/debug/{self.corpus_tag}"
        self.save_intermediate = save_intermediate

    # ---------- helpers ----------
    @classmethod
    def _ensure_answer_prefix(cls, answer: str) -> str:
        text = str(answer or "")
        if _ANSWER_PREFIX not in text:
            return f"{_ANSWER_PREFIX} {text}"
        return text

    @staticmethod
    def _strip_format_instruction(query: str) -> str:
        marker = "[Benchmark Output Format]"
        if marker in query:
            return query.split(marker, 1)[0].strip()
        return query

    @staticmethod
    def _build_unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen = set()
        for row in rows:
            doc = row.get("title") or row.get("doc") or "Unknown"
            page = row.get("page", 0)
            sent_id = row.get("sent_id", 0)
            key = (doc, page, sent_id)
            if key in seen:
                continue
            unique.append({
                "doc": doc,
                "page": page,
                "text": row.get("text", ""),
                "sent_id": sent_id,
            })
            seen.add(key)
        return unique

    @staticmethod
    def _build_answer_prompt(context: str, user_query: str) -> str:
        # Single-pass synthesis prompt (paper §3.2.6). Structurally identical
        # to the HopRAG / naive baseline prompts so any score gap traces back
        # to retrieval, not synthesis-prompt asymmetry. The analyst role is
        # domain-aware (RAGConfig.DOMAIN) so news/multi-hop corpora aren't
        # framed as financial filings; all baselines use the same role helper.
        return (
            f"You are {answer_role()}. Answer the question using only the provided context.\n"
            "If the context is insufficient, say you do not know.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question: {user_query}\n"
            "\n"
            "Answer:"
        )

    # ---------- main entry ----------
    async def run_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        """Retrieve-only query path (paper §3.2). Returns (answer, sources, trace).

        No agent loop, no reflection, no refinement. The path is:
          1. Two-stage hybrid retrieve (Q-/body, then Q+ expansion if needed).
          2. Cross-encoder rerank with top-up.
          3. Deterministic 1-hop NEXT/HOP traversal over pre-built edges
             (when RAG_AGENTIC_OFF_GRAPH_DEPTH > 0, default).
          4. Single LLM synthesis call.
        """
        _ = history
        retrieval_query = self._strip_format_instruction(user_query)
        graph_depth = RAGConfig.AGENTIC_OFF_GRAPH_DEPTH

        if graph_depth > 0:
            context, nodes = await self.graph_search(
                entities=[retrieval_query],
                depth=graph_depth,
                top_k=RAGConfig.DEFAULT_TOP_K,
                user_query=retrieval_query,
                force_expand=True,
            )
        else:
            context, nodes = await self.retrieve(retrieval_query, top_k=RAGConfig.DEFAULT_TOP_K)

        retrieved_nodes = nodes if isinstance(nodes, list) else []
        sources = self._build_unique_sources(retrieved_nodes)

        trace: list[dict[str, Any]] = [{
            "step": "retrieve",
            "input": {"query": user_query, "top_k": RAGConfig.DEFAULT_TOP_K, "graph_depth": graph_depth},
            "output": {"retrieved_sources": len(sources)},
        }]

        if not context:
            answer = self._ensure_answer_prefix("Insufficient evidence.")
            trace.append({
                "step": "synthesis",
                "output": {"answer": answer, "reason": "empty_context"},
            })
            return answer, sources, trace

        prompt = self._build_answer_prompt(context, user_query)
        messages = [{"role": "user", "content": prompt}]
        raw = await self.llm.generate_response(messages)
        answer = self._ensure_answer_prefix(str(raw or ""))
        trace.append({
            "step": "synthesis",
            "output": {"answer": answer},
        })
        return answer, sources, trace
