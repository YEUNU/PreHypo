"""Tests for the retrieval-only query path on GraphRAG.run_workflow().

The PreHypo query path is deliberately thin: retrieve -> single LLM call,
no agent loop, no reflection, no refinement. These tests pin the public
contract of run_workflow() and the small helpers it composes
(_ensure_answer_prefix, _strip_format_instruction, _build_unique_sources,
_build_answer_prompt).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.prehypo.graphrag import GraphRAG


# ---------------------------------------------------------------------------
# Helpers (no mocking needed — pure functions / classmethods)
# ---------------------------------------------------------------------------


def test_ensure_answer_prefix_adds_marker_when_missing():
    rag = GraphRAG(strategy="prehypo")
    out = rag._ensure_answer_prefix("Revenue was $394B in FY2022.")  # noqa: SLF001
    assert out.startswith("@@ANSWER:")
    assert "Revenue was $394B" in out


def test_ensure_answer_prefix_is_noop_when_marker_present():
    rag = GraphRAG(strategy="prehypo")
    raw = "@@ANSWER: Revenue was $394B in FY2022."
    assert rag._ensure_answer_prefix(raw) == raw  # noqa: SLF001


def test_ensure_answer_prefix_handles_empty_and_none():
    rag = GraphRAG(strategy="prehypo")
    assert rag._ensure_answer_prefix(None).startswith("@@ANSWER:")  # noqa: SLF001
    assert rag._ensure_answer_prefix("").startswith("@@ANSWER:")  # noqa: SLF001


def test_strip_format_instruction_drops_benchmark_suffix():
    rag = GraphRAG(strategy="prehypo")
    q = "What was Apple's FY2022 revenue? [Benchmark Output Format] respond with..."
    assert rag._strip_format_instruction(q) == "What was Apple's FY2022 revenue?"  # noqa: SLF001


def test_strip_format_instruction_passthrough_without_marker():
    rag = GraphRAG(strategy="prehypo")
    q = "What was Apple's FY2022 revenue?"
    assert rag._strip_format_instruction(q) == q  # noqa: SLF001


def test_build_unique_sources_dedups_by_doc_page_sent():
    rag = GraphRAG(strategy="prehypo")
    rows = [
        {"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "..."},
        {"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "..."},  # dup
        {"doc": "AAPL_10K", "page": 41, "sent_id": 4, "text": "..."},   # different sent
        {"title": "AAPL_10K", "page": 42, "sent_id": 3, "text": "..."},  # different page
    ]
    out = rag._build_unique_sources(rows)  # noqa: SLF001
    assert len(out) == 3
    keys = {(s["doc"], s["page"], s["sent_id"]) for s in out}
    assert keys == {("AAPL_10K", 41, 3), ("AAPL_10K", 41, 4), ("AAPL_10K", 42, 3)}


def test_build_unique_sources_uses_unknown_when_doc_missing():
    rag = GraphRAG(strategy="prehypo")
    out = rag._build_unique_sources([{"page": 1, "sent_id": 0, "text": "x"}])  # noqa: SLF001
    assert out[0]["doc"] == "Unknown"


def test_build_answer_prompt_contains_context_and_query():
    prompt = GraphRAG._build_answer_prompt("CTX_BLOCK", "QUESTION_TEXT")
    assert "CTX_BLOCK" in prompt
    assert "QUESTION_TEXT" in prompt
    # Voice-of-the-prompt: cite-only synthesis, abstain on insufficient.
    assert "only the provided context" in prompt
    assert "insufficient" in prompt.lower() or "do not know" in prompt.lower()


# ---------------------------------------------------------------------------
# run_workflow — full path with mocked retrieve + LLM
# ---------------------------------------------------------------------------


def _make_rag_with_mocks(
    *,
    nodes=None,
    context="Some retrieved context.",
    llm_answer="Apple's FY2022 revenue was $394B.",
    graph_depth=1,
):
    rag = GraphRAG(strategy="prehypo")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value=llm_answer)
    rag.graph_search = AsyncMock(return_value=(context, nodes or []))
    rag.retrieve = AsyncMock(return_value=(context, nodes or []))
    # Pin graph_depth at the config layer so we test both branches.
    rag_patch = patch("core.config.RAGConfig.GRAPH_HOP_DEPTH", graph_depth)
    rag_patch.start()
    return rag, rag_patch


@pytest.mark.asyncio
async def test_run_workflow_returns_answer_sources_trace_tuple():
    nodes = [{"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "Revenue $394B"}]
    rag, p = _make_rag_with_mocks(nodes=nodes)
    try:
        answer, sources, trace = await rag.run_workflow("What was Apple's FY2022 revenue?")
        assert answer.startswith("@@ANSWER:")
        assert "Apple's FY2022 revenue was $394B" in answer
        assert sources == [{"doc": "AAPL_10K", "page": 41, "sent_id": 3, "text": "Revenue $394B"}]
        assert isinstance(trace, list) and len(trace) == 2
        assert trace[0]["step"] == "retrieve"
        assert trace[1]["step"] == "synthesis"
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_uses_graph_search_when_depth_positive():
    rag, p = _make_rag_with_mocks(graph_depth=1)
    try:
        await rag.run_workflow("any query")
        rag.graph_search.assert_awaited_once()
        rag.retrieve.assert_not_awaited()
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_falls_back_to_retrieve_when_depth_zero():
    rag, p = _make_rag_with_mocks(graph_depth=0)
    try:
        await rag.run_workflow("any query")
        rag.retrieve.assert_awaited_once()
        rag.graph_search.assert_not_awaited()
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_abstains_on_empty_context():
    rag, p = _make_rag_with_mocks(context="", nodes=[])
    try:
        answer, sources, trace = await rag.run_workflow("query nobody can answer")
        assert "Insufficient evidence" in answer
        assert answer.startswith("@@ANSWER:")
        # Synthesis step should record the empty-context reason and the LLM
        # should NOT have been called.
        rag.llm.generate_response.assert_not_awaited()
        assert trace[1]["output"]["reason"] == "empty_context"
        assert sources == []
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_strips_benchmark_format_marker_before_retrieving():
    nodes = [{"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "Revenue $394B"}]
    rag, p = _make_rag_with_mocks(nodes=nodes, graph_depth=1)
    try:
        await rag.run_workflow("What was Apple's FY2022 revenue? [Benchmark Output Format] foo")
        # graph_search receives the stripped query, not the suffix-tainted one.
        call_kwargs = rag.graph_search.await_args.kwargs
        assert "[Benchmark Output Format]" not in (call_kwargs.get("user_query") or "")
        assert "[Benchmark Output Format]" not in (call_kwargs.get("entities") or [""])[0]
    finally:
        p.stop()
