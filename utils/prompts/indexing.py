from core.config import RAGConfig

# ---------------------------------------------------------------------------
# Domain-specific prompt variants. The model-side prompts below are selected
# by RAGConfig.DOMAIN ("financial" for FinanceBench / SEC filings, "news" for
# MultiHop-RAG / general article corpora). The Q-/Q+ hypothetical-query
# structure (paper §3.1.3) is identical across domains; only the grounding
# anchors and bridge types differ (financial statements/periods vs. named
# entities/sources/events). Selection happens once at import; each index or
# benchmark process runs a single dataset, so domain is fixed per process.
# ---------------------------------------------------------------------------

_FIN_HOPRAG_PROMPT = """
Analyze this financial document chunk and generate hypothetical questions to enable multi-hop reasoning.

Definitions (paper §3.1.3 — keep them strictly distinct):
- Q- (Incoming, self-contained): questions that THIS CHUNK ALONE answers verbatim. Used to retrieve this chunk when a user query asks for a fact already on the page.
- Q+ (Outgoing dependency / Bridge): questions that POINT OUTWARD from this chunk — they reference an entity, period, or metric grounded here, but the answer ALSO REQUIRES information from a different chunk or document. The Q+ question is the missing counterpart that another chunk would supply. Q+ is what builds the multi-hop graph; it is NOT a paraphrase of Q-.

Rules:
1. Q-: up to 3 self-contained questions this chunk directly answers; use [] if the chunk lacks concrete answerable facts.
2. Q+: up to 3 outward-dependency questions. Each Q+ MUST satisfy at least one of:
   (a) ask about the SAME metric in a DIFFERENT period than the one(s) shown here;
   (b) ask about a DERIVED metric (margin, ratio, growth, YoY change, average, FCF, ROA, ROE) whose primitive operands are NOT all present in this chunk;
   (c) ask about a RELATED LINE ITEM in a DIFFERENT statement (e.g., this chunk shows revenue → ask the cash-flow counterpart, this chunk shows net income → ask the balance-sheet counterpart);
   (d) ask about a MULTI-DOC bridge (e.g., compare to a prior-year filing, or to a segment break-down referenced in notes).
   If none of (a)-(d) apply, leave Q+ empty rather than emit a Q- duplicate.
3. Every produced question must be specific, answerable in finite SEC-filing context, and <= 22 words.
4. Each question SHOULD include grounding tokens (entity / metric / period / source anchor) that appear in this chunk; aim for at least two of these signals per question to keep the question retrievable.
5. If a year/period token exists in this chunk, include it in each Q-; for Q+ a different period token is allowed (in fact preferred for type (a)).
6. Never use placeholders/meta phrases such as "document anchor", "what does the balance sheet show", or lists like "(balance sheet, income statement, cash flow statement, note table)".
7. Never fabricate unseen values, dates, entities, policies, or legal details.
8. If this chunk is mostly TOC/exhibits/signatures/boilerplate or numeric fragments with weak context, return shorter lists (or empty lists) rather than low-quality questions.
9. If the chunk contains computation cues (change, increase/decrease, ratio, margin, versus/prior period, multi-period values), produce at least 1 Q+ of type (a) or (b).
10. Dense Summary: exactly 1 sentence, maximum 35 words, grounded only in this chunk; preserve numeric strings exactly when present.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""

_NEWS_HOPRAG_PROMPT = """
Analyze this news/article text chunk and generate hypothetical questions to enable multi-hop reasoning.

Definitions (paper §3.1.3 — keep them strictly distinct):
- Q- (Incoming, self-contained): questions that THIS CHUNK ALONE answers verbatim. Used to retrieve this chunk when a user query asks for a fact already on the page.
- Q+ (Outgoing dependency / Bridge): questions that POINT OUTWARD from this chunk — they reference a person, organization, place, event, or date grounded here, but the answer ALSO REQUIRES information from a DIFFERENT chunk or article. The Q+ question is the missing counterpart another article would supply. Q+ is what builds the multi-hop graph; it is NOT a paraphrase of Q-.

Rules:
1. Q-: up to 3 self-contained questions this chunk directly answers; use [] if the chunk lacks concrete answerable facts.
2. Q+: up to 3 outward-dependency questions. Each Q+ MUST satisfy at least one of:
   (a) ask about the SAME entity or event at a DIFFERENT time, or as reported by a DIFFERENT source, than shown here;
   (b) ask about a RELATIONSHIP, role, motive, or consequence linking an entity grounded here to another entity/event NOT fully described in this chunk;
   (c) ask a COMPARISON or cause/effect that requires another article (e.g., how another outlet covered the same event, an earlier cause or later development);
   (d) ask about a CROSS-DOCUMENT bridge (a person/organization/place mentioned here whose details live in a different article).
   If none of (a)-(d) apply, leave Q+ empty rather than emit a Q- duplicate.
3. Every produced question must be specific, answerable from a finite news context, and <= 22 words.
4. Each question SHOULD include grounding tokens (person / organization / place / event / source / date) that appear in this chunk; aim for at least two of these signals per question to keep the question retrievable.
5. If a date/time token exists in this chunk, include it in each Q-; for Q+ a different time is allowed (in fact preferred for type (a)).
6. Never use placeholders/meta phrases such as "this article", "the source", or "the document" as the only anchor.
7. Never fabricate unseen facts, dates, entities, quotes, or events.
8. If this chunk is mostly navigation/ads/bylines/boilerplate or fragments with weak context, return shorter lists (or empty lists) rather than low-quality questions.
9. If the chunk contains comparative or temporal cues (before/after, increased/decreased, versus, earlier/later, in response to), produce at least 1 Q+ of type (a) or (c).
10. Dense Summary: exactly 1 sentence, maximum 35 words, grounded only in this chunk; preserve names and dates exactly when present.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""

HOPRAG_PROMPT = _NEWS_HOPRAG_PROMPT if RAGConfig.DOMAIN == "news" else _FIN_HOPRAG_PROMPT

HOPRAG_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"summary": "concise informative summary", "q_minus": ["q1", "q2", "q3"], "q_plus": ["q1", "q2", "q3"]}}
"""

_FIN_QUERY_REWRITE_PROMPT = """
Rewrite the query into finance-focused retrieval variants.
Rules:
1. Generate 1-3 high-precision variants preserving original meaning.
2. Detect constraint anchors from the original query: target company/entity token(s) and target period token(s) (year/FY/quarter/date).
3. Every variant must include the same target company/entity token(s) and target period token(s) when present; if they cannot be preserved, do not emit that variant.
4. Keep metric, numeric qualifiers, and formula definition unchanged.
5. Preserve exact tokens for symbols/segments/line items when present (e.g., MMM26, consumer segment).
6. When the original query references or implies a financial statement (balance sheet, income statement, cash flow statement, note table, PP&E, accounts receivable, inventory, debt securities), include that anchor term in at least one variant. For yes/no, definitional, or qualitative queries that do not involve a specific statement, omit the anchor rather than fabricating one.
7. Apply filing synonym normalization only when equivalent:
   - revenue ↔ net sales
   - capex ↔ purchases of property, plant and equipment (PP&E)
   - net PP&E ↔ property, plant and equipment — net
   - net AR ↔ trade accounts receivable, net
8. Do NOT introduce another company/year/period, unsupported assumptions, or special query syntax operators.
Original Query: {query}
"""

_NEWS_QUERY_REWRITE_PROMPT = """
Rewrite the query into precise retrieval variants for a news/article corpus.
Rules:
1. First decide whether the query compares, contrasts, or relates TWO OR MORE distinct named entities, sources, or events (signals: "A versus B", "both X and Y", "between ... and ...", "compared to", or two different articles/publishers/dates). This is the common multi-hop case.
   - If YES (multi-subject): DECOMPOSE the query. Emit one focused single-subject variant per entity/source/event, each carrying ONLY that one subject plus any shared time/topic/relationship anchor. Each compared side must get its own variant so its evidence is retrieved independently. Do NOT also emit the original compound query as a variant.
   - If NO (single-subject): generate 1-3 high-precision paraphrase variants preserving the original meaning, keeping all named-entity and time tokens in every variant.
2. Detect constraint anchors from the original query: named-entity token(s) (person/organization/place), source or publisher token(s), and time token(s) (date/month/year).
3. Preserve exact tokens for proper nouns, titles, and source/publisher names when present (e.g., The Verge, TechCrunch, BBC). When the original query references a specific source/publisher or date, keep that token on the variant it belongs to; never fabricate one.
4. Keep the core relationship, comparison, or event description and any named qualifiers unchanged within each variant.
5. Use only widely-equivalent synonyms (e.g., CEO ↔ chief executive); never swap one named entity for another.
6. Do NOT introduce another entity/date/source, unsupported assumptions, or special query syntax operators.
Original Query: {query}
"""

QUERY_REWRITE_PROMPT = _NEWS_QUERY_REWRITE_PROMPT if RAGConfig.DOMAIN == "news" else _FIN_QUERY_REWRITE_PROMPT

RERANK_QUERY_SIMPLIFY_PROMPT = """
Extract the underlying question from a possibly-verbose user query for use as
a cross-encoder reranker input. The reranker scores chunk-vs-question
relevance and is hurt by long preludes, role framing, and meta instructions.

Rules:
1. Preserve every concrete constraint from the original: target entity, period,
   metric/line-item, statement anchor (e.g., "from cash flow statement"), unit,
   rounding.
2. Drop role-play preludes ("Answer as if you are...", "Imagine you are..."),
   reasoning instructions ("step by step", "show your work"), output-format
   instructions ("round to one decimal place", "answer in percent"), and
   editorial prefaces ("According to the details clearly outlined within...").
3. Output one sentence ending with a question mark.
4. Do NOT introduce constraints not present in the original query.
5. If the original is already a single concise question, return it as-is.

Original Query: {query}
"""

RERANK_QUERY_SIMPLIFY_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"question": "..."}}
"""

QUERY_REWRITE_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"positive_queries": []}}
"""

_FIN_RERANKER_INSTRUCTION = (
    "Rank the passage by whether it directly answers the query. "
    "Match the exact entity, period, and line-item phrasing requested in the query. "
    "When the query asks about a specific line-item name, prefer passages "
    "whose tokens for that line item are identical to the query, not merely "
    "near-synonymous (e.g., a passage reporting 'X expense' is not equivalent "
    "to one reporting 'X and Y' when the query asks for 'X and Y'). "
    "Down-rank boilerplate."
)

_NEWS_RERANKER_INSTRUCTION = (
    "Rank the passage by whether it directly answers the query. "
    "Match the exact named entity, source/publisher, time, and event phrasing "
    "requested in the query. Prefer passages naming the same people, "
    "organizations, places, and dates as the query, not merely related ones. "
    "Down-rank boilerplate, navigation, and ads."
)

RERANKER_INSTRUCTION = _NEWS_RERANKER_INSTRUCTION if RAGConfig.DOMAIN == "news" else _FIN_RERANKER_INSTRUCTION

_FIN_SEARCH_CONTINUATION_PROMPT = """
Decide whether retrieval should continue.
Decision rules:
1. Infer required evidence slots from QUERY. For compute queries, infer all primitive operands needed for the formula, not only the final derived metric.
2. Return "SUFFICIENT" only when all required slots are grounded in context with matching target company/entity and target period constraints.
3. For compute queries, evidence may come from multiple pages/documents; do not require a single-document hit.
4. Return "INSUFFICIENT" if any required slot is missing, ambiguous, conflicting, or tied to the wrong entity/period.
5. Prefer stopping as soon as slot coverage is complete; avoid unnecessary extra hops.
QUERY: {query}
CONTEXT: {context}
"""

_NEWS_SEARCH_CONTINUATION_PROMPT = """
Decide whether retrieval should continue.
Decision rules:
1. Infer required evidence slots from QUERY. For multi-hop queries, infer each distinct fact, entity, or source needed to answer, not only the final answer.
2. Return "SUFFICIENT" only when all required slots are grounded in context with matching named entities, sources, and time constraints.
3. Evidence may come from multiple articles/sources; do not require a single-document hit.
4. Return "INSUFFICIENT" if any required slot is missing, ambiguous, conflicting, or tied to the wrong entity/source/time.
5. Prefer stopping as soon as slot coverage is complete; avoid unnecessary extra hops.
QUERY: {query}
CONTEXT: {context}
"""

SEARCH_CONTINUATION_PROMPT = _NEWS_SEARCH_CONTINUATION_PROMPT if RAGConfig.DOMAIN == "news" else _FIN_SEARCH_CONTINUATION_PROMPT

SEARCH_CONTINUATION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"decision": "SUFFICIENT"|"INSUFFICIENT", "next_focus": "..."}}
"""
