"""Unit tests for the opt-in OpenAI Batch API judge path.

Mocks the OpenAI SDK so the collect -> submit -> poll -> parse -> map flow and
the shared score-resolution logic are exercised without a real API call.
"""
import json

import pytest

import openai

import utils.batch_judge as bj
from utils.batch_judge import OpenAIBatchJudge
from utils.metrics import _resolve_judge_fields


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Files:
    def __init__(self, out_text):
        self._out = out_text

    def create(self, file, purpose):  # noqa: A002 - mirrors SDK kwarg name
        return _Obj(id="file-in")

    def content(self, file_id):
        return _Obj(text=self._out)


class _Batches:
    """Returns 'in_progress' on create, 'completed' on the first retrieve, so
    the poll loop runs exactly one iteration."""

    def create(self, **kw):
        return _Obj(id="batch-1", status="in_progress", output_file_id="file-out", request_counts=None)

    def retrieve(self, batch_id):
        return _Obj(id="batch-1", status="completed", output_file_id="file-out", request_counts=None)


def _make_fake_openai(payloads_by_id):
    lines = []
    for cid, payload in payloads_by_id.items():
        lines.append(json.dumps({
            "custom_id": cid,
            "response": {"body": {"choices": [{"message": {"content": json.dumps(payload)}}]}},
        }))
    out_text = "\n".join(lines)

    class _FakeOpenAI:
        def __init__(self, api_key=None):
            self.files = _Files(out_text)
            self.batches = _Batches()

    return _FakeOpenAI


@pytest.mark.asyncio
async def test_batch_judge_collects_runs_and_maps(monkeypatch):
    monkeypatch.setattr(bj.time, "sleep", lambda *_a, **_k: None)  # no real polling delay
    monkeypatch.setattr(openai, "OpenAI", _make_fake_openai({
        "0": {"score": 1.0, "hallucination": 0.0, "reason": "ok"},
        "1": {"score": 0.0, "hallucination": 1.0, "reason": "bad"},
    }))

    judge = OpenAIBatchJudge(model="gpt-test", api_key="sk-test", poll_seconds=2)
    judge.register("0", "judge prompt for q0")
    judge.register("1", "judge prompt for q1")
    assert judge.count == 2

    results = await judge.run()
    assert set(results.keys()) == {"0", "1"}
    assert results["0"]["score"] == 1.0
    assert results["1"]["hallucination"] == 1.0


@pytest.mark.asyncio
async def test_batch_judge_empty_is_noop():
    judge = OpenAIBatchJudge(model="gpt-test", api_key="sk-test")
    assert await judge.run() == {}


def test_resolve_judge_fields_payload_then_unjudged():
    # Payload with a usable score is used as-is.
    fields = _resolve_judge_fields(
        {"score": 1.0, "hallucination": 0.0, "reason": "correct"},
        response="The answer is 42.",
        judge_model="gpt-test",
    )
    assert fields["llm_judge_score"] == 1.0
    assert fields["hallucination"] == 0.0
    assert fields["hallucination_model"] == "gpt-test"

    # No payload (judge failed, or batch not yet resolved) → UNJUDGED (-1),
    # never silently 0; hallucination is unjudged too (not fabricated).
    from utils.metrics import UNJUDGED_SCORE
    fb = _resolve_judge_fields(
        None,
        response="Some substantive answer.",
        judge_model="gpt-test",
    )
    assert fb["llm_judge_score"] == UNJUDGED_SCORE == -1.0
    assert fb["llm_judge_reason"] == "unjudged_no_score"
    assert fb["hallucination"] == UNJUDGED_SCORE


def test_resolve_judge_fields_abstain_is_not_hallucination():
    fields = _resolve_judge_fields(
        {"score": 0.0, "hallucination": 1.0, "reason": "wrong"},
        response="Insufficient evidence to answer.",
        judge_model="gpt-test",
    )
    # Honest abstain is resolved deterministically to non-hallucination.
    assert fields["hallucination"] == 0.0
    assert fields["hallucination_source"] == "rule_non_answer"


@pytest.mark.asyncio
async def test_reconcile_pending_judges_patches_and_cleans(tmp_path, monkeypatch):
    """A pending manifest + result JSON gets patched from the resolved batch,
    aggregates recomputed, summary sidecar refreshed, and the manifest removed."""
    import cli.benchmark as cb
    from core.config import RAGConfig

    run_dir = tmp_path / "run"
    strat_dir = run_dir / "naive" / "multihoprag"
    strat_dir.mkdir(parents=True)
    result_file = strat_dir / "naive_multihoprag.json"

    # Two deferred (UNJUDGED -1) rows tagged with judge_custom_id, out of order.
    result_json = {
        "strategy": "naive", "corpus_tag": "multihoprag", "dataset": "MultiHop-RAG",
        "total_queries": 2,
        "details": [
            {"category": "inference", "answer": "Paris is the capital.",
             "judge_custom_id": "1", "llm_judge_score": -1.0, "hallucination": -1.0,
             "doc_match": 1.0, "latency": 2.0},
            {"category": "comparison", "answer": "It is larger.",
             "judge_custom_id": "0", "llm_judge_score": -1.0, "hallucination": -1.0,
             "doc_match": 0.0, "latency": 3.0},
        ],
    }
    result_file.write_text(json.dumps(result_json))
    manifest = {
        "batch_id": "batch-XYZ", "judge_model": "gpt-test",
        "result_file": str(result_file), "strategy": "naive",
        "corpus_tag": "multihoprag", "dataset": "MultiHop-RAG",
        "is_financebench": False, "is_multihoprag": True, "submitted": 2,
    }
    pending = strat_dir / "naive_multihoprag.pending_judge.json"
    pending.write_text(json.dumps(manifest))

    monkeypatch.setattr(RAGConfig, "OPENAI_API_KEY", "sk-test")
    # custom_id "0" -> correct, "1" -> wrong; reconcile must map by judge_custom_id.
    monkeypatch.setattr(cb, "_recompute_aggregates", cb._recompute_aggregates)  # keep real
    monkeypatch.setattr(
        "utils.batch_judge.resolve_batches",
        lambda api_key, ids, poll: {"batch-XYZ": {
            "0": {"score": 1.0, "hallucination": 0.0, "reason": "ok"},
            "1": {"score": 0.0, "hallucination": 1.0, "reason": "wrong"},
        }},
    )

    patched = await cb.reconcile_pending_judges(run_dir)
    assert patched == 1
    assert not pending.exists()  # manifest removed

    out = json.loads(result_file.read_text())
    by_cat = {r["category"]: r for r in out["details"]}
    assert by_cat["comparison"]["llm_judge_score"] == 1.0   # custom_id 0
    assert by_cat["inference"]["llm_judge_score"] == 0.0     # custom_id 1
    # Aggregate over judged rows only.
    assert out["avg_llm_judge_score"] == pytest.approx(0.5)
    assert out["financebench_correct_count"] == 1
    assert out["financebench_incorrect_count"] == 1
    # Summary sidecar refreshed.
    sidecar = json.loads((strat_dir / "naive_multihoprag.summary.json").read_text())
    assert sidecar["avg_llm_judge_score"] == pytest.approx(0.5)
    assert "details" not in sidecar
