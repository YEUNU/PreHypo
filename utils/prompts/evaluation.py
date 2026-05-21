QUERY_CATEGORIZATION_PROMPT = """
Analyze the provided query and its ground truth evidence to categorize it for an ablation study.
Categories:
1. "Table": The answer is primarily located in a table or requires extracting specific numerical/categorical data from tabular structures.
2. "Global": The query asks about the document as a whole, trends across multiple years, or high-level summaries.
3. "Multi-hop": The query requires connecting 2 or more distinct pieces of information that are not adjacent in the text.
4. "Local": Simple fact extraction from a single, contiguous paragraph.

Query: {query}
Evidence: {evidence}

Respond in the following format:
CATEGORY: [Category Name]
REASON: [Short explanation]
"""

FINANCEBENCH_JUDGE_PROMPT = """
### Task: Score the Model Prediction on (a) correctness vs Ground Truth and
(b) hallucination — with a SINGLE LLM call so the two judgements are
internally consistent.

**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

### Instructions
1. Locate the FINAL answer in the Model Prediction (typically inside
   \\boxed{{...}}, after "Final Answer:", or after the last "@@ANSWER:" marker).
   Judge on that final answer only — ignore intermediate reasoning and
   worked-out arithmetic.
2. Allow equivalent unit scaling (million/billion/M) and formatting
   differences ($, commas, %, rounding) when the underlying value matches.
3. score:
   - 1.0 if the final answer conveys the same factual/financial content
     as the Ground Truth (minor wording differences ok).
   - 0.0 if the final answer is factually wrong, contradicts the Ground
     Truth, or provides an incorrect value.
   - "Insufficient evidence" / abstention is 0.0 when Ground Truth contains
     a substantive answer; if Ground Truth itself is "no value / 0 / not
     disclosed", an abstention is acceptable (1.0).
4. hallucination:
   - 1.0 if the final answer asserts wrong/conflicting factual or numeric
     content.
   - 0.0 if the final answer is factually consistent with the Ground
     Truth, or is an honest abstention ("insufficient evidence" / non-
     answer). An abstention is NOT a hallucination, even when the Ground
     Truth has a substantive answer.
5. Internal consistency: hallucination=1.0 implies score=0.0. score=1.0
   implies hallucination=0.0. score=0.0 with hallucination=0.0 is the
   honest-abstain case.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "hallucination": 1.0 or 0.0, "reason": "brief explanation covering both judgements"}}
"""


MULTIHOPRAG_JUDGE_PROMPT = """
### Task: Score a Model Prediction for a MultiHop-RAG question over a news
corpus on (a) correctness vs Ground Truth and (b) hallucination — in a
SINGLE LLM call so the two judgements stay internally consistent.

MultiHop-RAG answers are short, factual spans (an entity/person/organization
name, a publisher, a date or time period, or a yes/no comparison result),
NOT financial figures. There is no unit scaling or currency formatting to
reconcile here — judge on factual identity, not numeric tolerance.

**Question Type:** {question_type}
**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

### Instructions
1. Locate the FINAL answer in the Model Prediction (typically after
   "Final Answer:", the last "@@ANSWER:" marker, or inside \\boxed{{...}}).
   Judge on that final answer only — ignore intermediate hop-by-hop reasoning.
2. Apply the criterion for the question type:
   - inference_query / comparison_query: the predicted entity / comparison
     outcome must match the Ground Truth entity (surface aliases and
     reorderings are acceptable; e.g. "The Verge" == "the verge").
   - temporal_query: the chronological fact / ordering / date must match.
   - null_query: the Ground Truth indicates the corpus has NO answer
     ("insufficient information"). Here an honest abstention ("the context
     does not contain ...", "insufficient information", "cannot be
     determined") is the CORRECT answer (score 1.0); fabricating a concrete
     answer is wrong (score 0.0).
3. score:
   - 1.0 if the final answer is factually equivalent to the Ground Truth
     (minor wording / alias / casing differences ok).
   - 0.0 if it names the wrong entity/date/outcome, or — for a non-null
     question — abstains when a substantive Ground Truth exists.
4. hallucination:
   - 1.0 if the final answer asserts a concrete but factually wrong entity,
     date, or comparison outcome (including a fabricated answer to a
     null_query).
   - 0.0 if it is factually consistent with the Ground Truth, or is an
     honest abstention.
5. Internal consistency: hallucination=1.0 implies score=0.0; score=1.0
   implies hallucination=0.0; score=0.0 with hallucination=0.0 is the
   honest-abstain case.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "hallucination": 1.0 or 0.0, "reason": "brief explanation covering both judgements"}}
"""
