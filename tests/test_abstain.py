"""Tests for FinanceBench abstain detection and 3-way labeling.

The point of these tests is to pin the rule that judge score takes
precedence over abstain detection: if the ground truth itself is a
non-answer, an abstention IS the Correct Answer (judge score 1.0), and the
3-way label must not downgrade it to Refusal.
"""
import pytest

from utils.abstain import is_abstain, financebench_label, ABSTAIN_PHRASES


# ---------------------------------------------------------------------------
# is_abstain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Insufficient evidence to determine the answer.",
    "I do not know based on the provided context.",
    "I don't know.",
    "The answer cannot be determined from the filing.",
    "The context does not contain the requested figure.",
    "The context does not mention 2022 CAPEX.",
    "No information about FY2023 revenue is provided.",
    "Unable to find relevant information in the filing.",
])
def test_is_abstain_recognizes_phrase(text):
    assert is_abstain(text), f"expected abstain detection on: {text!r}"


@pytest.mark.parametrize("text", [
    "Apple's FY2022 revenue was $394 billion.",
    "Net income totaled $99.8B according to the income statement.",
    "The CAPEX increased by 12% year over year.",
])
def test_is_abstain_rejects_substantive_answer(text):
    assert not is_abstain(text)


def test_is_abstain_case_insensitive():
    assert is_abstain("INSUFFICIENT EVIDENCE")
    assert is_abstain("I Do Not Know")


def test_is_abstain_handles_none_and_empty():
    assert is_abstain(None) is False
    assert is_abstain("") is False


def test_abstain_phrases_includes_hypo_native_marker():
    # PreHypo's pipeline-specific abstain phrase must remain in the list;
    # removing it would silently mis-classify Hypo's own abstentions as
    # Incorrect Answer with answer_attempted=1.
    assert "insufficient evidence" in ABSTAIN_PHRASES


# ---------------------------------------------------------------------------
# financebench_label
# ---------------------------------------------------------------------------


def test_label_is_correct_when_judge_score_is_high():
    assert financebench_label(1.0, "Revenue was $394B.") == "Correct Answer"
    assert financebench_label(0.5, "Revenue was $394B.") == "Correct Answer"


def test_label_correct_when_score_high_even_if_response_looks_like_abstain():
    # Judge score takes precedence. FinanceBench has questions whose GT is a
    # legitimate non-answer; an honest abstention against such GT is awarded
    # score=1.0 by the judge prompt and must label as "Correct Answer".
    assert financebench_label(1.0, "Insufficient evidence.") == "Correct Answer"


def test_label_is_refusal_when_score_low_and_response_abstains():
    assert financebench_label(0.0, "I do not know.") == "Refusal"
    assert financebench_label(0.4, "The context does not contain this figure.") == "Refusal"


def test_label_is_incorrect_when_score_low_and_response_is_substantive():
    assert financebench_label(0.0, "Revenue was $1 trillion.") == "Incorrect Answer"


def test_label_handles_non_numeric_score():
    # Defensive: garbage input collapses to score=0.0 and must not crash.
    assert financebench_label("not a number", "Insufficient evidence.") == "Refusal"
    assert financebench_label(None, "Revenue was $100B.") == "Incorrect Answer"
