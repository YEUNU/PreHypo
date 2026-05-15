"""Smoke tests for the PreHypo package layout.

These tests pin the structural invariants of this repository:
- The package was renamed `models.hyporeflect` -> `models.prehypo`.
- The agentic 5-stage code paths (stages/, orchestrator, service,
  agentic_core) are gone.
- GraphRAG exposes a `run_workflow()` entry point (the retrieval-only path
  that the paper reports).

If any of these break, the README and paper claims drift away from reality.
"""
import importlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _all_python_sources():
    for path in REPO_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == "test_smoke.py":
            # Skip self — this file legitimately mentions hyporeflect /
            # AgentService while testing their absence.
            continue
        yield path


# ---------------------------------------------------------------------------
# Package rename / agentic strip
# ---------------------------------------------------------------------------


def test_models_prehypo_imports_cleanly():
    mod = importlib.import_module("models.prehypo.graphrag")
    assert hasattr(mod, "GraphRAG")


def test_models_hyporeflect_is_gone():
    try:
        importlib.import_module("models.hyporeflect")
    except ImportError:
        return
    raise AssertionError("models.hyporeflect should not exist in PreHypo")


def test_models_agentic_core_is_gone():
    try:
        importlib.import_module("models.agentic_core")
    except ImportError:
        return
    raise AssertionError("models.agentic_core should not exist in PreHypo")


def test_models_prehypo_stages_is_gone():
    try:
        importlib.import_module("models.prehypo.stages")
    except ImportError:
        return
    raise AssertionError("models.prehypo.stages should not exist in PreHypo")


def test_agent_service_class_does_not_exist():
    try:
        mod = importlib.import_module("models.prehypo.service")
    except ImportError:
        return
    assert not hasattr(mod, "AgentService"), "AgentService leaked back into PreHypo"


# ---------------------------------------------------------------------------
# Source-level sweeps
# ---------------------------------------------------------------------------


_FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"\bfrom\s+models\.hyporeflect\b"),
    re.compile(r"\bimport\s+models\.hyporeflect\b"),
    re.compile(r"\bfrom\s+models\.agentic_core\b"),
    re.compile(r"\bimport\s+models\.agentic_core\b"),
    re.compile(r"\bfrom\s+models\.prehypo\.stages\b"),
    re.compile(r"\bAgentService\b"),
)


def test_no_forbidden_imports_or_agent_service_refs():
    offenders = []
    for path in _all_python_sources():
        try:
            src = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in _FORBIDDEN_IMPORT_PATTERNS:
            if pat.search(src):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), pat.pattern))
    assert not offenders, f"forbidden refs found:\n" + "\n".join(
        f"  {p} matches {pat}" for p, pat in offenders
    )


def test_no_agentic_prompt_imports():
    # The three prompt files were deleted (utils/prompts/{execution,
    # planning,synthesis}.py). No source file should still import names
    # that lived only in those modules.
    forbidden_names = (
        "QUERY_STATE_PROMPT", "EVIDENCE_LEDGER_PROMPT",
        "CONTEXT_ATOMIZATION_PROMPT", "CONTEXT_PACKING_PROMPT",
        "PERCEPTION_PROMPT", "PLANNING_PROMPT", "PLANNING_MERGED_PROMPT",
        "REFLECTION_PROMPT", "RESPONSE_REFINEMENT_PROMPT",
    )
    offenders = []
    for path in _all_python_sources():
        try:
            src = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name in forbidden_names:
            if re.search(rf"\b{name}\b", src):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), name))
    assert not offenders, "agentic prompt names still referenced:\n" + "\n".join(
        f"  {p} mentions {n}" for p, n in offenders
    )


# ---------------------------------------------------------------------------
# GraphRAG public API
# ---------------------------------------------------------------------------


def test_graphrag_exposes_run_workflow():
    from models.prehypo.graphrag import GraphRAG
    assert hasattr(GraphRAG, "run_workflow"), \
        "GraphRAG.run_workflow() is the retrieval-only query entry; required by cli/benchmark.py"


def test_graphrag_inherits_indexing_and_retrieval():
    from models.prehypo.graphrag import GraphRAG
    from models.prehypo.indexing import IndexingPipeline
    from models.prehypo.retrieval import RetrievalPipeline
    assert issubclass(GraphRAG, IndexingPipeline)
    assert issubclass(GraphRAG, RetrievalPipeline)


def test_baselines_still_importable():
    # Naive / HopRAG / MS-GraphRAG are kept in this repo as comparison
    # baselines — they should import cleanly with the agentic code paths
    # removed.
    from models.naive.naive_rag import NaiveRAG  # noqa: F401
    # HopRAG / MS-GraphRAG adapters pull heavy optional deps at import time
    # (paddle/graphrag/etc.); we don't import them here so the smoke test
    # stays light and runnable without those installed.
