# Phase 4 Walkthrough: LLM-Powered Processing & Extraction

This document explains every file, every class, every method, and every design
decision in Phase 4. It covers how the Gemini client manages retries, caching,
and cost tracking; how confidence scoring works through dual-prompt extraction
and source grounding; how hallucination detection works step by step; why
map-reduce is needed for long papers; how user feedback creates a learning loop;
and how Phase 4 connects Phase 3's ranked chunks to Phase 5's API endpoints.
Read this before a technical interview and you will be able to talk through any
line of code with confidence.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [LLM Processing Pipeline Diagram](#2-llm-processing-pipeline-diagram)
3. [How the Gemini Client Handles Retries, Caching, and Cost Tracking](#3-how-the-gemini-client-handles-retries-caching-and-cost-tracking)
4. [How Confidence Scoring Works for Extractions](#4-how-confidence-scoring-works-for-extractions)
5. [How Hallucination Detection Works Step by Step](#5-how-hallucination-detection-works-step-by-step)
6. [How Map-Reduce Summarization Works](#6-how-map-reduce-summarization-works)
7. [Why the Feedback System Matters for Production](#7-why-the-feedback-system-matters-for-production)
8. [How Phase 4 Connects to Phase 3 and Phase 5](#8-how-phase-4-connects-to-phase-3-and-phase-5)
9. [File-by-File Deep Dive](#9-file-by-file-deep-dive)
   - [src/models/schemas.py (Phase 4 additions)](#91-srcmodelsschemaspyphase-4-additions)
   - [src/models/database.py (ExtractionFeedback)](#92-srcmodelsdatabasepy-extractionfeedback)
   - [src/llm/gemini_client.py](#93-srcllmgemini_clientpy)
   - [src/llm/prompts.py](#94-srcllmpromptspy)
   - [src/llm/extractor.py](#95-srcllmextractorpy)
   - [src/llm/qa_engine.py](#96-srcllmqa_enginepy)
   - [src/llm/summarizer.py](#97-srcllmsummarizerpy)
   - [src/api/routes_extract.py](#98-srcapiroutes_extractpy)
   - [tests/unit/test_gemini_client.py](#99-testsunittest_gemini_clientpy)
   - [tests/unit/test_extractor.py](#910-testsunittest_extractorpy)
   - [tests/unit/test_qa_engine.py](#911-testsunittest_qa_enginepy)
   - [tests/unit/test_summarizer.py](#912-testsunittest_summarizerpy)
10. [Cross-Cutting Design Decisions](#10-cross-cutting-design-decisions)
11. [Interview Questions and Answers](#11-interview-questions-and-answers)

---

## 1. Architecture Overview

Phase 4 is the **intelligence layer**. It takes text, sections, and ranked
chunks from earlier phases and uses Gemini to extract structured knowledge,
answer questions with citations, and produce multi-level summaries. Everything
flows through a single centralized LLM client.

```
Layer              Responsibility
─────────────────  ─────────────────────────────────────────────────────
GeminiClient       Single entry point for ALL LLM calls — caching, retry, cost
PaperExtractor     Dual-prompt extraction → key findings, methodology, results
QAEngine           Citation-tracked QA → retrieve, generate, verify
PaperSummarizer    Three-level summarization → one-line, abstract, map-reduce
FeedbackRouter     Correction endpoint → user fixes feed back into the system
```

Key architectural principles:

- **Centralized LLM access** — Every Gemini call goes through `GeminiClient`.
  In Phase 3 each module had its own `_call_gemini()`. Phase 4 consolidates
  this into one client that adds caching, retry, cost tracking, and metrics.
  This means a bug fix or rate-limit change applies everywhere at once.
- **Confidence at every level** — Extractions carry a numeric confidence score.
  Below 0.7 the system flags `needs_review=True`. This is not a binary
  pass/fail but a continuous signal that tells downstream consumers how much
  to trust the output.
- **Verify before serving** — The QA engine does not just generate an answer.
  It generates, then verifies each claim against its cited source, then returns
  a faithfulness score. This two-pass approach catches hallucinations before
  users see them.
- **Graceful degradation everywhere** — Every method catches `RuntimeError`,
  logs a warning, and returns a safe default (empty list, empty string, 0.0
  score). The system never crashes on an LLM failure.
- **Dependency injection** — `PaperExtractor`, `QAEngine`, and
  `PaperSummarizer` all accept an optional `GeminiClient` in their constructor.
  Tests inject a `MagicMock`. Production code lets them auto-create a real
  client.

---

## 2. LLM Processing Pipeline Diagram

```
                        ┌──────────────────────────┐
                        │      GeminiClient         │
                        │  ┌────────┐ ┌──────────┐  │
                        │  │ Cache  │ │  Retry   │  │
                        │  │(Redis) │ │(backoff) │  │
                        │  └────────┘ └──────────┘  │
                        │  ┌─────────┐ ┌─────────┐  │
                        │  │ Metrics │ │  Cost   │  │
                        │  │(Prom.)  │ │Tracking │  │
                        │  └─────────┘ └─────────┘  │
                        └──────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │  PaperExtractor  │ │     QAEngine     │ │ PaperSummarizer  │
    │                  │ │                  │ │                  │
    │ Dual-prompt      │ │ Retrieve →       │ │ one_line (≤30w)  │
    │ extraction       │ │ Generate →       │ │ abstract (150w)  │
    │ + grounding      │ │ Verify claims    │ │ detailed (MR)    │
    │ + confidence     │ │ + faithfulness   │ │                  │
    └──────────────────┘ └────────┬─────────┘ └──────────────────┘
                                  │
                                  │ uses Phase 3
                                  ▼
                        ┌──────────────────┐
                        │ HybridRetriever  │
                        │ (Phase 3)        │
                        └──────────────────┘

    ┌──────────────────┐
    │ POST /feedback   │  ← user corrections flow back
    │ routes_extract   │
    └──────────────────┘
```

---

## 3. How the Gemini Client Handles Retries, Caching, and Cost Tracking

This section explains the three production concerns that `GeminiClient` handles
so that no other module has to think about them.

### 3.1 Retry with Exponential Backoff

When Gemini returns a transient error (HTTP 429 rate-limit, 500 internal
error, or 503 service unavailable), the client retries automatically.

**Why exponential backoff?** If the server is overloaded and you retry
immediately, you make the problem worse. By waiting longer between each
attempt (1s → 2s → 4s), you give the server breathing room while still
recovering quickly from short hiccups.

```python
# gemini_client.py lines 188-230
def _retry_with_backoff(self, func, max_retries=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()                    # Attempt the call
        except Exception as exc:
            last_exc = exc
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status_code and int(status_code) not in {429, 500, 503}:
                raise                        # Non-transient → fail immediately
            wait = 1.0 * (2 ** attempt)      # 1s, 2s, 4s
            time.sleep(wait)
    raise RuntimeError(f"failed after {max_retries} retries: {last_exc}")
```

**Critical design choices:**

1. **Non-transient errors are NOT retried.** A 400 Bad Request means the
   prompt is malformed. Retrying it will never succeed. The code checks the
   status code and re-raises immediately for anything outside `{429, 500, 503}`.

2. **The delay is `1.0 * 2^attempt`**, not randomized. For a single-instance
   application this is fine. In a distributed system you would add jitter
   (`wait * random(0.5, 1.5)`) to prevent thundering herd.

3. **`RuntimeError` wraps the final exception** with the retry count, so the
   caller gets a clear error message: "failed after 3 retries: rate limit".

### 3.2 Redis-Based Response Caching

Identical prompts produce identical outputs (temperature=0.0). Caching avoids
redundant API calls, which saves both time and money.

```python
# Cache key: SHA256(model_name + ":" + prompt + ":" + system_instruction)
def _compute_cache_key(self, prompt, system_instruction):
    raw = f"{self._model_name}:{prompt}:{system_instruction or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

**Flow:**

1. `generate()` computes the cache key from model + prompt + system instruction.
2. `_check_cache()` calls `redis.get("llm_cache:{hash}")`.
3. **Cache hit** → return `LLMResponse(cached=True, cost_usd=0.0)` immediately.
   No API call, no tokens, no cost.
4. **Cache miss** → call Gemini, then `_store_cache()` stores the response
   with `redis.setex()` and a 1-hour TTL.

**Why SHA256?** Prompts can be 8,000+ characters. Redis keys should be short
and fixed-length. SHA256 gives 64 hex characters and is collision-resistant.

**Why 1-hour TTL?** Research papers do not change, but model behaviour might
after updates. One hour is a good balance — long enough to avoid redundant
calls during a processing session, short enough to pick up model improvements.

**Graceful degradation:** `_get_redis()` wraps the connection in a try/except
and returns `None` if Redis is unavailable. Every `_check_cache` and
`_store_cache` call checks for `None` first. The system works perfectly
without Redis — it just makes more API calls.

```python
def _get_redis(self):
    if self._redis is not None:
        return self._redis          # Already connected
    if self._redis_checked:
        return None                  # Already tried, failed
    self._redis_checked = True       # Only try once
    try:
        import redis as redis_lib
        client = redis_lib.from_url(settings.redis_url, decode_responses=True)
        client.ping()                # Verify connection works
        self._redis = client
        return client
    except Exception:
        return None                  # Caching disabled, system continues
```

### 3.3 Cost Tracking

Every response includes `cost_usd`, computed from Gemini Flash pricing:

```python
# $0.075 per 1M input tokens, $0.30 per 1M output tokens
def _compute_cost(input_tokens, output_tokens):
    return (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
```

**Concrete example:** A dual-prompt extraction sends ~8,000 chars twice. If
each prompt uses ~2,000 input tokens and gets ~500 output tokens:

```
Cost per call = (2000 * 0.075 + 500 * 0.30) / 1,000,000
             = (150 + 150) / 1,000,000
             = $0.0003
Two calls    = $0.0006
```

This is tracked per call in `LLMResponse.cost_usd` and also recorded in
Prometheus counters (`llm_input_tokens_total`, `llm_output_tokens_total`,
`llm_requests_total`) and a histogram (`llm_latency_seconds`). The
Prometheus metrics are initialized lazily with a `try/except ImportError`
guard so the system works without `prometheus_client` installed.

---

## 4. How Confidence Scoring Works for Extractions

Confidence scoring is the key innovation in `PaperExtractor`. Instead of
trusting a single LLM response blindly, the system cross-validates through
three independent signals and combines them into a weighted score.

### 4.1 The Three Signals

**Signal 1: Dual-Prompt Consistency (weight: 0.4)**

Run the same extraction with two differently-worded prompts. If both prompts
produce the same findings, those findings are likely real.

```
Prompt A: "You are an expert research paper analyst. Extract the key findings..."
Prompt B: "You are a systematic reviewer. Identify all significant findings..."
```

The system sends both prompts, parses both responses, then uses
`difflib.SequenceMatcher` to fuzzy-match claims across the two sets:

```python
# extractor.py lines 245-254
for claim_a in claims_a:
    for claim_b in claims_b:
        ratio = SequenceMatcher(None, claim_a, claim_b).ratio()
        if ratio >= 0.6:       # FUZZY_MATCH_THRESHOLD
            overlap += 1
            break

consistency = overlap / max(len(claims_a), len(claims_b))
```

**Why fuzzy matching?** The two prompts will phrase findings differently. Prompt A
might say "The model achieves 94.2% F1" while Prompt B says "94.2% F1-score is
achieved by the proposed model." Exact string matching would miss this. A
SequenceMatcher ratio of 0.6 is lenient enough to catch these rephrasings.

**Signal 2: Source Grounding (weight: 0.4)**

For each finding, the LLM provides a `supporting_quote` — text it claims to
have found in the paper. The system checks whether this quote actually exists
in the source text.

```python
# extractor.py lines 256-282
def _check_source_grounding(quote, source_text):
    q_lower = quote.lower()
    s_lower = source_text.lower()

    if q_lower in s_lower:          # Exact substring → 1.0
        return 1.0

    ratio = SequenceMatcher(None, q_lower, s_lower).ratio()
    return min(1.0, ratio)           # Fuzzy match → 0.0-1.0
```

The fast path checks for an exact substring match first (O(n)). If that fails,
it falls back to SequenceMatcher for fuzzy matching. This catches quotes that
have minor differences like whitespace normalization or punctuation changes.

**Signal 3: Completeness (weight: 0.2)**

A paper with zero findings is suspicious. Completeness rewards finding at least
3 findings: `completeness = min(1.0, len(findings) / 3.0)`.

### 4.2 The Combined Formula

```python
confidence = 0.4 * consistency + 0.4 * avg_grounding + 0.2 * completeness
```

**Numeric Example:**

Suppose Prompt A finds 3 findings and Prompt B finds 4. Two of the three from
A fuzzy-match findings in B:

```
consistency = 2 / max(3, 4) = 0.5

grounding for finding 1: exact match in source   → 1.0
grounding for finding 2: fuzzy match ratio 0.72  → 0.72
grounding for finding 3: no match found           → 0.15
avg_grounding = (1.0 + 0.72 + 0.15) / 3 = 0.623

completeness = min(1.0, 3 / 3.0) = 1.0

confidence = 0.4 * 0.5 + 0.4 * 0.623 + 0.2 * 1.0
           = 0.20 + 0.249 + 0.20
           = 0.649
```

Since 0.649 < 0.7, `needs_review=True` is set. A human reviewer is flagged
to check this extraction.

### 4.3 Why These Weights?

Consistency and grounding each get 40% because they are the strongest signals
of correctness. If two independent prompts agree AND the supporting quotes
exist in the source, the finding is almost certainly real. Completeness gets
20% because it is a weaker signal — a paper might genuinely have only 1 key
finding, and we should not penalize that too heavily.

---

## 5. How Hallucination Detection Works Step by Step

Hallucination detection is spread across two systems: **source grounding** in
the extractor (Section 4) and **claim verification** in the QA engine. This
section walks through a concrete end-to-end example of the QA verification
pipeline.

### 5.1 The Problem

A user asks: "What F1 score does the model achieve?"

The system retrieves context passages and generates an answer. But LLMs can
introduce facts that are not in the passages. We need to catch these before
the user sees them.

### 5.2 Step-by-Step Walkthrough

**Step 1: Retrieve context** (`_retrieve_chunks`)

The QA engine calls `HybridRetriever.retrieve()` which returns the top-k
ranked chunks from Phase 3. These are the ONLY facts the answer should use.

```
Source [1]: "Our model achieves 94.2% F1 on DocBank..."
Source [2]: "The baseline LayoutLM model scores 89.1% F1..."
```

**Step 2: Build context** (`_build_context`)

Each chunk gets a numbered `[N]` marker. A `Citation` object records the
mapping: `[1] → chunk_id "chunk-42", paper "paper-001", section "results"`.

**Step 3: Generate answer** (`_generate`)

The `QA_ANSWER` prompt instructs Gemini to cite every factual claim with `[N]`:

```
"Rules:
 1. Only use information from the provided context
 2. Cite every factual claim with [N] referring to the source passage
 3. If the context does not contain enough information, say so explicitly
 4. Do not make up information not present in the context"
```

Gemini returns: `"The model achieves 94.2% F1 on DocBank [1], a 5.1% improvement over LayoutLM [2]."`

**Step 4: Verify claims** (`_verify`)

The system sends the answer AND the source passages to Gemini with the
`QA_VERIFY_CLAIMS` prompt. Gemini acts as a fact-checker, evaluating each
claim against its cited source:

```json
{
  "verifications": [
    {
      "claim": "The model achieves 94.2% F1 on DocBank",
      "cited_source_index": 1,
      "status": "SUPPORTED",
      "explanation": "Source [1] states '94.2% F1 on DocBank'"
    },
    {
      "claim": "a 5.1% improvement over LayoutLM",
      "cited_source_index": 2,
      "status": "NOT_SUPPORTED",
      "explanation": "Source [2] says LayoutLM scores 89.1%. The improvement is 94.2 - 89.1 = 5.1%, but the source does not state this improvement explicitly."
    }
  ]
}
```

**Step 5: Compute faithfulness score**

```python
supported = 1   # "94.2% F1 on DocBank"
total     = 2   # two claims checked
faithfulness = 1 / 2 = 0.5
```

**Step 6: Flag unsupported claims**

```python
flagged_claims = ["a 5.1% improvement over LayoutLM"]
```

**The output:**

```python
QAResponse(
    query="What F1 score does the model achieve?",
    answer="The model achieves 94.2% F1 on DocBank [1], a 5.1% improvement...",
    citations=[Citation(source_index=1, ...), Citation(source_index=2, ...)],
    claim_verifications=[...],
    faithfulness_score=0.5,
    flagged_claims=["a 5.1% improvement over LayoutLM"],
)
```

The downstream consumer sees `faithfulness_score=0.5` and `flagged_claims` is
non-empty. It can decide to show the answer with a warning, or suppress it
entirely and fall back to quoting the sources directly.

### 5.3 Why Two-Pass Verification?

Why not just ask Gemini to "be careful" in the first pass? Because LLMs are
poor at self-correcting during generation. They commit to a claim mid-sentence
and continue building on it. A separate verification pass treats the answer as
an external document to fact-check, which engages the model's analytical
reasoning rather than its generative mode. Research shows this "generate then
verify" pattern reduces hallucination rates by 30-50%.

### 5.4 The Helper That Was Built But Not Called

`_split_claims()` exists as a utility for programmatically splitting answers
into (sentence, cited_source_index) tuples using regex:

```python
sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
citations = re.findall(r"\[(\d+)\]", sentence)
```

In the current implementation, the verification prompt handles claim splitting
internally (Gemini identifies the claims). But `_split_claims()` is available
for cases where you want to split claims client-side — for example, to add
additional programmatic checks or to feed claims to a different verifier.

---

## 6. How Map-Reduce Summarization Works

### 6.1 The Problem with Long Papers

A typical research paper has 8,000-15,000 words. Gemini Flash has a large
context window, but there are three reasons not to feed the entire paper in
one shot:

1. **Quality degrades with length.** LLMs produce better summaries of shorter
   texts. A 500-word section gets a better summary than a 12,000-word paper.
2. **Prompt truncation.** We truncate to 8,000 characters (`_TEXT_TRUNCATE`)
   to control token costs. A long paper loses its Results and Conclusion.
3. **Section awareness.** The Introduction, Methodology, Results, and
   Conclusion serve different rhetorical functions. Summarizing them together
   flattens these distinctions.

### 6.2 The Three Levels

```
Level       Target         Strategy           Use Case
──────────  ───────────    ─────────────────  ──────────────────────────
one_line    ≤30 words      Single prompt      Paper cards, search results
abstract    100-150 words  Single prompt      Paper listings, quick reads
detailed    300-500 words  Map-reduce         Deep understanding, reports
```

### 6.3 Map-Reduce for Detailed Summaries

**Map Phase:** Summarize each section independently.

```
Input sections:  [Introduction, Methodology, Results, Conclusion]
Skip:            [Title, References, Appendix]   ← no useful summary content

Section 1 (Introduction) → "The paper addresses document understanding..."
Section 2 (Methodology)  → "A multi-modal transformer combines text and visual..."
Section 3 (Results)       → "The model achieves 94.2% F1 on DocBank..."
Section 4 (Conclusion)    → "The work demonstrates state-of-the-art results..."
```

Each section gets its own LLM call with the `SUMMARIZE_SECTION` prompt that
includes the section title and type for context:

```python
# summarizer.py lines 192-208
def _summarize_section(self, section):
    response = self._client.generate(
        SUMMARIZE_SECTION.format(
            section_title=section.title or "",
            section_type=section.section_type.value,
            text=section.text[:8000],          # Truncate per section
        )
    )
    return self._extract_summary(response.content)
```

If any section fails (RuntimeError from Gemini), it is silently skipped. The
remaining sections still produce a useful summary.

**Reduce Phase:** Synthesize section summaries into one narrative.

```python
# summarizer.py lines 210-222
def _synthesize(self, section_summaries):
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(section_summaries, 1))
    response = self._client.generate(
        SUMMARIZE_SYNTHESIZE.format(section_summaries=numbered)
    )
    return self._extract_summary(response.content)
```

The `SUMMARIZE_SYNTHESIZE` prompt tells Gemini to:
1. Open with the paper's main contribution
2. Cover methodology and approach
3. Present key results with specific numbers
4. Discuss significance and limitations

**Fallback:** If the reduce step fails, the method returns a concatenation of
the section summaries. Not as polished, but still useful.

### 6.4 Why Skip References, Title, and Appendix?

```python
skip = {SectionType.REFERENCES, SectionType.TITLE, SectionType.APPENDIX}
```

- **References** are a list of other papers. Summarizing them is meaningless.
- **Title** is already captured in metadata. It is one line, not summarizable.
- **Appendix** contains supplementary tables and proofs that do not belong in
  a summary.

### 6.5 Numeric Example: LLM Calls for a 6-Section Paper

Paper sections: Title, Abstract, Introduction, Methodology, Results, Conclusion, References.

After filtering: Abstract, Introduction, Methodology, Results, Conclusion = **5 sections**.

LLM calls: **5 map calls + 1 reduce call = 6 total calls.**

At $0.0003 per call: **$0.0018 per detailed summary.**

---

## 7. Why the Feedback System Matters for Production

### 7.1 The Feedback Loop

```
User sees extraction  →  "This finding is wrong"
                          │
                          ▼
POST /api/v1/feedback     paper_id, field_name, original_value, corrected_value
                          │
                          ▼
Store correction          _FEEDBACK_STORE (in-memory, MVP)
                          ExtractionFeedback (ORM model, ready for PostgreSQL)
                          │
                          ▼
Threshold check           If corrections for (paper_id, field_name) ≥ 5:
                            → log.warning("Feedback threshold reached")
                            → Flag prompt for review
```

### 7.2 Why This Matters

1. **Prompt engineering is iterative.** No prompt is perfect on the first try.
   When users correct extractions, you discover systematic prompt failures.
   "The methodology extraction always misses the dataset name" tells you to
   add "Pay special attention to dataset names" to the prompt.

2. **Confidence calibration.** If users frequently correct findings that had
   confidence 0.75, you know your threshold of 0.7 is too low. Feedback data
   lets you tune the threshold empirically.

3. **Model drift detection.** If feedback spikes after a Gemini model update,
   you know the new model behaves differently and your prompts need adjustment.

4. **Training data for fine-tuning.** Every (original_value, corrected_value)
   pair is a labelled example. Collect enough of these and you can fine-tune
   a smaller model to match or exceed the base model on your specific domain.

### 7.3 The Threshold Mechanism

```python
# routes_extract.py lines 49-62
count = sum(
    1 for f in _FEEDBACK_STORE
    if f["paper_id"] == request.paper_id
    and f["field_name"] == request.field_name
)
if count >= _FEEDBACK_THRESHOLD:  # 5
    logger.warning("Feedback threshold reached for paper_id=%s field=%s (%d corrections)")
```

When the same field on the same paper gets corrected 5 or more times, the
system logs a warning. In production, this would trigger an alert in your
monitoring system (Grafana, PagerDuty). The threshold is deliberately low
because if 5 different users all say the same field is wrong, you should
investigate immediately.

### 7.4 MVP vs. Production

The MVP uses an in-memory list (`_FEEDBACK_STORE: list[dict]`). This is
intentional — it avoids async database sessions and keeps the endpoint simple.
The `ExtractionFeedback` ORM model is already defined in `database.py` with
a composite index on `(paper_id, field_name)`, ready for a one-line migration
to PostgreSQL:

```python
# database.py lines 154-169
class ExtractionFeedback(Base):
    __tablename__ = "extraction_feedback"
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    paper_id = Column(String(256), nullable=False)
    field_name = Column(String(128), nullable=False)
    original_value = Column(Text, nullable=False)
    corrected_value = Column(Text, nullable=False)
    user_comment = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_feedback_paper_field", "paper_id", "field_name"),
    )
```

---

## 8. How Phase 4 Connects to Phase 3 and Phase 5

### 8.1 Phase 3 → Phase 4: Retrieval Feeds QA

Phase 3 produces `RankedResult` objects — chunks scored and ranked by
BM25 + vector + RRF + optional reranking. Phase 4's `QAEngine` consumes
these directly:

```python
# qa_engine.py line 22
from src.retrieval.hybrid_retriever import HybridRetriever, RankedResult
```

The `QAEngine.__init__` takes a `retriever: HybridRetriever` and the
`_retrieve_chunks` method calls `retriever.retrieve(query, top_k=top_k)`.
The `RankedResult` fields used by Phase 4:

```
RankedResult.text            → becomes the context passage
RankedResult.chunk_id        → becomes Citation.chunk_id
RankedResult.paper_id        → becomes Citation.paper_id
RankedResult.section_type    → becomes Citation.section_type
RankedResult.page_numbers    → becomes Citation.page_numbers
RankedResult.final_score     → used for sorting when merging multi-paper results
```

### 8.2 Phase 1 → Phase 4: Sections Feed Extraction and Summarization

Phase 1's `LayoutAnalyzer` produces `DetectedSection` objects. Phase 4 uses
them in two places:

- `PaperSummarizer.summarize(sections=sections, level="detailed")` — feeds
  sections into the map-reduce pipeline.
- `PaperExtractor` receives the full paper text (concatenated from sections).

### 8.3 Phase 4 → Phase 5: Structured Output for API Endpoints

Phase 4 produces structured Pydantic models that Phase 5 will serve:

```
PaperExtraction   → GET /api/v1/papers/{id}/extraction
QAResponse        → POST /api/v1/qa
SummaryResult     → GET /api/v1/papers/{id}/summary?level=detailed
FeedbackResponse  → POST /api/v1/feedback  (already implemented)
```

Every output is a Pydantic model with `Field(ge=0.0, le=1.0)` constraints on
scores. This means Phase 5 can serialize them directly to JSON with automatic
validation — no additional transformation needed.

### 8.4 The Complete Data Flow

```
PDF Upload (Phase 1)
  → Pages, Sections, Tables, Metadata
    → Chunks (Phase 2: SectionAwareChunker)
      → Embeddings + BM25 Index (Phase 2-3)
        → Ranked Chunks (Phase 3: HybridRetriever)
          → QA Answers with Citations (Phase 4: QAEngine)
          → Structured Extractions (Phase 4: PaperExtractor)
          → Multi-level Summaries (Phase 4: PaperSummarizer)
            → REST API Endpoints (Phase 5)
              → User Feedback (Phase 4: FeedbackRouter)
                → Prompt Improvement (Production Loop)
```

---

## 9. File-by-File Deep Dive

### 9.1 `src/models/schemas.py` (Phase 4 additions)

Phase 4 adds 12 new Pydantic models after the existing Phase 1-3 schemas.

**`LLMResponse`** (line 207-216) — The return type of every `GeminiClient.generate()` call.

```python
class LLMResponse(BaseModel):
    content: str                     # Raw text from Gemini
    input_tokens: int = Field(ge=0)  # Prompt tokens consumed
    output_tokens: int = Field(ge=0) # Response tokens generated
    latency_ms: float = Field(ge=0)  # Wall-clock time in milliseconds
    model: str                       # Model identifier used
    cached: bool = False             # True if served from Redis
    cost_usd: float = Field(ge=0)    # Estimated cost in USD
```

**Why track all this?** Because in production you need to answer: "How much
did we spend on LLM calls last month?", "What is our p99 latency?", "What
percentage of calls are cache hits?". Every field exists to power a dashboard.

**`Finding`** (line 223-228) — A single key finding. Has `claim`,
`supporting_quote` (for source grounding verification), and `confidence`
constrained to `[0.0, 1.0]`.

**`PaperExtraction`** (line 250-258) — Aggregates findings, methodology,
results, an overall confidence, and `needs_review: bool`. This is the output
of the full extraction pipeline.

**`Citation`** (line 265-273) — Links an `[N]` marker in an answer to a
specific chunk, paper, section, and page numbers. This is what makes the QA
system auditable — every claim can be traced back to its source.

**`ClaimVerification`** (line 276-282) — The result of verifying one claim:
`status` is one of `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `NOT_SUPPORTED`.

**`QAResponse`** (line 285-293) — The full QA output: answer text, list of
citations, list of claim verifications, overall faithfulness score, and a
list of flagged (unsupported) claims.

**`SummaryLevel`** (line 300-303) — Enum with `ONE_LINE`, `ABSTRACT`,
`DETAILED`. Using a string enum means the API accepts `?level=detailed` and
Pydantic validates it automatically.

**`FeedbackRequest` / `FeedbackResponse`** (lines 320-339) — Request body and
response for the feedback endpoint. `FeedbackResponse` has
`ConfigDict(from_attributes=True)` so it can be created from an ORM row when
the system migrates from in-memory storage to PostgreSQL.

### 9.2 `src/models/database.py` (ExtractionFeedback)

```python
class ExtractionFeedback(Base):
    __tablename__ = "extraction_feedback"
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    paper_id = Column(String(256), nullable=False)
    field_name = Column(String(128), nullable=False)
    ...
    __table_args__ = (
        Index("ix_feedback_paper_field", "paper_id", "field_name"),
    )
```

**Why a composite index?** The threshold check queries by `(paper_id, field_name)`.
Without an index, every feedback submission would scan the entire table. The
composite index makes this O(log n).

### 9.3 `src/llm/gemini_client.py`

**324 lines.** Single-responsibility: wraps ALL Gemini API access.

**Module-level constants (lines 21-26):**
```python
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0           # 1s, 2s, 4s
_RETRYABLE_STATUS_CODES = {429, 500, 503}
_CACHE_TTL_SECONDS = 3600     # 1 hour
_FLASH_INPUT_COST_PER_M = 0.075
_FLASH_OUTPUT_COST_PER_M = 0.30
```

**`_init_metrics()` (lines 36-66):**
Lazily creates Prometheus `Counter` and `Histogram` objects. Uses a module-level
`_METRICS_INITIALISED` flag to ensure this runs only once. The `try/except
ImportError` guard means the system works without `prometheus_client`.

**`GeminiClient.__init__` (lines 79-89):**
Stores model name, retry count, cache TTL. Redis connection is lazy (`None`
until first cache check). `_redis_checked` prevents repeated connection
attempts after the first failure.

**`generate()` (lines 95-182):**
The main public method. Flow:
1. `_ensure_api_key()` — fail fast if no key
2. `_compute_cache_key()` + `_check_cache()` — return early on hit
3. `import google.generativeai as genai` — lazy import (saves startup time)
4. `genai.configure()` — set API key
5. Create model (with or without system instruction)
6. Set `response_mime_type="application/json"` and `temperature=0.0`
7. Wrap API call in `_retry_with_backoff()`
8. Extract `usage_metadata.prompt_token_count` and `candidates_token_count`
9. Compute cost, store in cache, track metrics
10. Return `LLMResponse`

**Why `response_mime_type="application/json"`?** This tells Gemini to produce
valid JSON. Combined with our prompt templates that specify exact JSON formats,
this dramatically reduces parsing failures.

**Why `temperature=0.0`?** We want deterministic, reproducible outputs. Higher
temperature adds randomness, which is undesirable for extraction tasks.

**Why lazy import?** `google.generativeai` takes ~200ms to import and
initializes gRPC channels. Deferring this to the first API call avoids
slowing down application startup.

### 9.4 `src/llm/prompts.py`

**207 lines.** All prompt templates as module-level string constants.

**Design decision: constants, not a class.** Prompts are static text with
`{placeholders}`. Making them constants means:
- They are importable: `from src.llm.prompts import QA_ANSWER`
- They are testable: `assert "{query}" in QA_ANSWER`
- They are versionable: `git diff` shows exactly what changed
- A/B testing is trivial: import `QA_ANSWER_V2` instead

**Prompt categories:**

| Prompt | Used by | Placeholders | Output format |
|--------|---------|-------------|---------------|
| `EXTRACT_KEY_FINDINGS` | Extractor (primary) | `{text}` | `{"findings": [...]}` |
| `EXTRACT_KEY_FINDINGS_ALT` | Extractor (dual) | `{text}` | `{"findings": [...]}` |
| `EXTRACT_METHODOLOGY` | Extractor | `{text}` | `{"approach": ...}` |
| `EXTRACT_RESULTS` | Extractor | `{text}` | `{"results": [...]}` |
| `EXTRACT_METADATA` | (available for use) | `{text}` | Title, authors, DOI |
| `QA_ANSWER` | QAEngine | `{query}`, `{context}` | Free text with [N] |
| `QA_VERIFY_CLAIMS` | QAEngine | `{answer}`, `{sources}` | `{"verifications": [...]}` |
| `SUMMARIZE_ONE_LINE` | Summarizer | `{text}` | `{"summary": "..."}` |
| `SUMMARIZE_ABSTRACT` | Summarizer | `{text}` | `{"summary": "..."}` |
| `SUMMARIZE_SECTION` | Summarizer | `{section_title}`, `{section_type}`, `{text}` | `{"summary": "..."}` |
| `SUMMARIZE_SYNTHESIZE` | Summarizer | `{section_summaries}` | `{"summary": "..."}` |
| `HALLUCINATION_CHECK` | (available for use) | `{claim}`, `{source_text}` | `{"supported": bool}` |

**Why doubled braces `{{`?** Python `str.format()` uses `{name}` for
substitution. To include literal braces in the output (for JSON examples), you
escape them by doubling: `{{` produces `{` in the final string.

### 9.5 `src/llm/extractor.py`

**354 lines.** The structured extraction pipeline with confidence scoring.

**`PaperExtractor.__init__` (line 50):**
Takes an optional `GeminiClient`. Creates one if not provided. This is the
dependency injection pattern used across all Phase 4 classes.

**`extract()` (lines 57-96):**
The full pipeline orchestrator:
1. Determine `paper_id` from metadata (or "unknown")
2. Call `extract_key_findings()` — dual-prompt, grounded, scored
3. Call `extract_methodology()` — single prompt
4. Call `extract_results()` — single prompt
5. Compute overall confidence = average of component confidences
6. Set `needs_review = overall < 0.7`

**Why does methodology get confidence 0.8 by default?** Methodology does not
have a dual-prompt or source-grounding pipeline (it is extracted once). Rather
than giving it an arbitrary 1.0, the system uses 0.8 as a "reasonable but not
perfect" default. This prevents methodology-only papers from getting
inflated confidence.

**`extract_key_findings()` (lines 98-146):**
The most complex extraction:
1. Guard clause: empty text → empty list, no LLM call
2. Call `_dual_prompt_extract()` with both prompts
3. For each finding, compute source grounding
4. Combine into weighted confidence
5. Clamp to [0.0, 1.0] with `min(1.0, max(0.0, confidence))`

**`_dual_prompt_extract()` (lines 211-254):**
Runs both prompts, parses both responses, computes consistency:
1. `resp_a = client.generate(prompt_a.format(text=text[:8000]))`
2. `resp_b = client.generate(prompt_b.format(text=text[:8000]))`
3. Parse both into lists of dicts
4. Fuzzy-match claims between the two sets
5. Return `(items_a, consistency_score)`

**Why return `items_a`?** The primary prompt is used as the canonical result.
Prompt B is only used to compute consistency. This means the system always
returns findings in the same format regardless of which prompt produces
better results. Consistency with Prompt A is the simplest design.

**Parsing helpers (lines 306-353):**
Every parser follows the same pattern:
1. Strip markdown fences (`_strip_markdown_fences`)
2. `json.loads()`
3. Check for expected top-level key (`"findings"`, `"approach"`, `"results"`)
4. Fall back to bare list format
5. Return empty list/None on any error

**Why strip markdown fences?** Even with `response_mime_type="application/json"`,
Gemini occasionally wraps JSON in ` ```json ... ``` ` markdown fences. The
regex `r"```(?:json)?\s*(.*?)\s*```"` handles this gracefully.

### 9.6 `src/llm/qa_engine.py`

**280 lines.** Citation-tracked QA with verification.

**`QAEngine.__init__` (lines 39-45):**
Takes a `GeminiClient` and a `HybridRetriever`. The retriever is optional —
without it, `answer()` returns "No relevant context found."

**`answer()` (lines 51-91):**
The five-step pipeline:
1. Guard: empty query → empty answer
2. `_retrieve_chunks()` — get top-k passages
3. `_build_context()` — format with [N] markers + create Citation objects
4. `_generate()` — Gemini call with QA_ANSWER prompt
5. `_verify()` — second Gemini call to fact-check

**`_retrieve_chunks()` (lines 97-129):**
Handles multi-paper retrieval:
- No paper IDs → retrieve globally
- With paper IDs → retrieve per paper, merge, deduplicate by `chunk_id`,
  sort by `final_score`, take top-k

The deduplication is important because the same chunk could appear in results
for multiple paper IDs. Without dedup, you would waste context window space
on repeated passages.

**`_build_context()` (lines 131-154):**
Formats chunks for the prompt:
```
Source [1]:
The model achieves 94.2% F1...

Source [2]:
The baseline LayoutLM scores 89.1%...
```
Also creates `Citation` objects that map `[N]` back to chunk metadata.
`text_snippet` is truncated to 200 characters for the verification step
(saves tokens).

**`_verify()` (lines 174-213):**
The hallucination detection step:
1. Build sources text from citation snippets
2. Call Gemini with QA_VERIFY_CLAIMS prompt
3. Parse verifications
4. Count SUPPORTED claims → faithfulness = supported / total
5. Collect flagged claims (NOT_SUPPORTED)

**`_split_claims()` (lines 215-240):**
Regex-based claim splitter:
- Split on sentence boundaries: `(?<=[.!?])\s+`
- Extract `[N]` markers: `\[(\d+)\]`
- Return `(sentence, source_index)` tuples

**`_parse_verifications()` (lines 246-270):**
Handles both `{"verifications": [...]}` and bare list formats. Defaults
status to `NOT_SUPPORTED` if missing — pessimistic by design. An unverifiable
claim should not count as supported.

### 9.7 `src/llm/summarizer.py`

**253 lines.** Three-level summarization with map-reduce.

**`PaperSummarizer.__init__` (line 49):**
Takes an optional `GeminiClient`.

**`summarize()` (lines 56-123):**
The routing method:
1. Validate level against `SummaryLevel` enum
2. Route to `_summarize_one_line()`, `_summarize_abstract()`, or
   `_summarize_detailed()`
3. If detailed but no sections provided → fallback to abstract
4. Always return `SummaryResult` with word count

**`_summarize_one_line()` (lines 129-143):**
Calls Gemini with `SUMMARIZE_ONE_LINE`, then enforces the 30-word limit:
```python
words = summary.split()
if len(words) > 30:
    summary = " ".join(words[:30])
```

**Why enforce client-side?** The prompt says "maximum 30 words" but LLMs
sometimes exceed soft limits. The hard truncation guarantees compliance.

**`_summarize_detailed()` (lines 156-190):**
Map-reduce implementation:
1. Filter: skip References, Title, Appendix
2. Map: `_summarize_section()` per section → list of `"label: summary"` strings
3. Reduce: `_synthesize()` combines into narrative

**`_extract_summary()` (lines 228-238):**
Parses `{"summary": "..."}` from Gemini. Falls back to raw text if JSON
parsing fails. This resilience is important because occasionally the model
returns plain text instead of JSON.

**`_count_words()` (lines 241-243):**
`len(text.split()) if text else 0`. Simple but correct for our use case.
A more sophisticated word counter (handling hyphens, contractions) is
unnecessary because we only need approximate counts.

### 9.8 `src/api/routes_extract.py`

**70 lines.** The feedback endpoint.

```python
router = APIRouter(prefix="/api/v1", tags=["feedback"])
_FEEDBACK_STORE: list[dict] = []
_FEEDBACK_THRESHOLD = 5
```

**`submit_feedback()` (lines 28-69):**
1. Generate UUID and UTC timestamp
2. Append to in-memory store
3. Count existing feedback for this (paper_id, field_name)
4. If count ≥ 5, log warning
5. Return `FeedbackResponse`

**Why async?** FastAPI endpoints are `async` by convention. Even though this
particular endpoint does no I/O (in-memory store), keeping it async allows
future migration to async database writes without changing the function
signature.

**Registration in `main.py`:**
```python
from src.api.routes_extract import router as feedback_router
app.include_router(feedback_router)
```

This mounts the router at `/api/v1/feedback` (the prefix comes from the
router, not `include_router`).

### 9.9 `tests/unit/test_gemini_client.py`

**248 lines, 16 tests.** Organized into four groups.

**Mocking strategy:** Tests mock `_retry_with_backoff` to return a mock
response object. This avoids mocking the `google.generativeai` namespace
package import, which is fragile due to Python's namespace package resolution.

**The `_mock_response()` helper** creates a MagicMock with `.text` and
`.usage_metadata.prompt_token_count` / `.candidates_token_count`, mimicking the
real Gemini response structure.

**Key test groups:**

| Group | Tests | What they verify |
|-------|-------|-----------------|
| Basic generation | 4 | LLMResponse shape, missing API key, system instruction, latency |
| Retry logic | 4 | Transient recovery, max attempts, non-transient passthrough, backoff delays |
| Caching | 5 | Cache hit, cache miss, deterministic keys, model-specific keys, Redis degradation |
| Cost computation | 3 | Zero tokens, known values ($0.375 for 1M+1M), small tokens |

**Notable test — exponential backoff delays:**
```python
def test_retry_exponential_backoff_delays(self):
    delays = []
    def always_fail():
        exc = Exception("error")
        exc.status_code = 503
        raise exc
    with patch("...time.sleep", side_effect=lambda d: delays.append(d)):
        with pytest.raises(RuntimeError):
            client._retry_with_backoff(always_fail, max_retries=3)
    assert delays == [1.0, 2.0, 4.0]
```

This captures the actual sleep durations and verifies the 2^n pattern.

### 9.10 `tests/unit/test_extractor.py`

**338 lines, 22 tests.** Six test classes covering every extraction path.

**Mocking strategy:** `GeminiClient` is a MagicMock whose `.generate()` returns
an `LLMResponse` with JSON content. The `_llm_response()` helper creates these.

**Key tests:**

- `test_dual_prompt_computes_consistency` — both prompts get the same mock
  response → consistency = 1.0. Verifies the fuzzy matching logic.
- `test_extract_low_confidence_flags_review` — empty findings → confidence
  0.0 → `needs_review=True`.
- `test_source_grounding_exact_match` — quote is a substring of source → 1.0.
- `test_compute_confidence_partial_scores` — verifies the weighted formula
  numerically: `0.4*0.5 + 0.4*0.7 + 0.2*0.33 = 0.546`.

### 9.11 `tests/unit/test_qa_engine.py`

**334 lines, 19 tests.** Seven test classes.

**Notable pattern:** The `_ranked()` helper creates `RankedResult` objects that
bridge Phase 3 to Phase 4 tests. This shows the integration point.

**Key tests:**

- `test_answer_with_retriever_returns_response` — end-to-end flow with mock
  retriever and mock client. First `generate()` call returns the answer,
  second returns verifications.
- `test_verify_mixed_support` — 1 SUPPORTED + 1 NOT_SUPPORTED = faithfulness 0.5.
- `test_retrieve_chunks_deduplicates` — same chunk from two papers → 1 result.

### 9.12 `tests/unit/test_summarizer.py`

**277 lines, 20 tests.** Five test classes.

**Key tests:**

- `test_one_line_truncates_to_30_words` — mock returns 50 words → truncated
  to 30 by client-side enforcement.
- `test_detailed_uses_sections` — 2 sections → 2 map calls + 1 reduce call → 3 total
  `generate()` calls, verified by `side_effect` list length.
- `test_detailed_skips_references_and_title` — 4 sections (Title, Intro,
  References, Appendix) → only Introduction is summarized.
- `test_detailed_section_failure_skips_section` — first section fails with
  RuntimeError → only second section appears in `sections_used`.

---

## 10. Cross-Cutting Design Decisions

### 10.1 Why Centralize LLM Calls?

Phase 3 had `_call_gemini()` methods scattered across `GeminiReranker`,
`QueryProcessor`, and `MetadataExtractor`. Each had its own import, configure,
and error handling. Phase 4 consolidates everything into `GeminiClient` because:

1. **One retry policy.** All calls get the same backoff behaviour.
2. **One cache.** Two modules requesting the same prompt share the cache hit.
3. **One cost tracker.** `llm_requests_total` captures ALL LLM calls, not just
   the ones from one module.
4. **One place to add features.** Rate limiting, request queuing, or model
   fallback can be added to `GeminiClient` without touching any caller.

Phase 3 code was NOT refactored to use `GeminiClient`. This is deliberate.
Backward compatibility means Phase 3 continues to work independently.

### 10.2 Why `difflib.SequenceMatcher` Instead of Embeddings?

Source grounding uses `SequenceMatcher` (stdlib) instead of embedding
similarity because:

1. **No dependency.** SequenceMatcher is in the standard library.
2. **Exact match fast path.** `if q_lower in s_lower: return 1.0` handles the
   common case in O(n) time.
3. **Character-level matching.** When the LLM slightly paraphrases a quote,
   character-level similarity catches it. Embedding similarity might give a
   high score to a semantically similar but factually different sentence.
4. **Predictable.** The ratio is always in [0.0, 1.0] and is purely
   deterministic. No model loading, no API calls, no latency.

### 10.3 Why In-Memory Feedback Store?

The feedback endpoint uses `_FEEDBACK_STORE: list[dict]` instead of the
database because:

1. **No async session complexity.** FastAPI async endpoints with SQLAlchemy
   require `async_session` which adds significant boilerplate.
2. **MVP speed.** The feature ships in 70 lines instead of 200.
3. **The ORM model is ready.** `ExtractionFeedback` in `database.py` has the
   exact same schema. Migration is a 10-line change.

### 10.4 Why Prompt Templates as Constants?

Alternatives considered:
- **Jinja2 templates** — overkill for simple substitution, adds a dependency
- **Class with methods** — `PromptBuilder().key_findings(text)` adds
  indirection without benefit
- **YAML/JSON config files** — harder to test, requires file I/O at import time

Module-level string constants with `str.format()` are the simplest thing that
works. Every prompt is importable, testable, and diff-visible.

---

## 11. Interview Questions and Answers

### Q1: "How do you detect hallucinations in your RAG system?"

**Answer:**

We detect hallucinations at two levels.

**At extraction time**, we use dual-prompt confidence scoring. Each finding is
extracted twice with differently-worded prompts. We fuzzy-match the results
using `difflib.SequenceMatcher` (threshold 0.6). Findings that only appear in
one extraction are penalized. Additionally, each finding includes a
`supporting_quote` that we verify exists in the source text — exact substring
match gives 1.0, fuzzy match gives a proportional score. The final confidence
is `0.4 * consistency + 0.4 * grounding + 0.2 * completeness`. Anything below
0.7 is flagged for human review.

**At QA time**, we use a two-pass generate-then-verify approach. The first
pass generates an answer with `[N]` citation markers. The second pass sends
the answer AND the cited source passages to Gemini as a fact-checking task.
Each claim gets a status: SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED.
The faithfulness score is `supported_claims / total_claims`. Unsupported claims
are surfaced in `flagged_claims` so the UI can warn users.

The key insight is that LLMs are better at verifying than at generating without
hallucination. By separating generation and verification, we get the
expressiveness of free-form generation with the accuracy of analytical
fact-checking.

---

### Q2: "How do you handle Gemini rate limits in production?"

**Answer:**

We handle it at three layers.

**Layer 1: Exponential backoff retry.** The `GeminiClient._retry_with_backoff()`
method catches HTTP 429, 500, and 503 errors and retries with delays of 1s,
2s, 4s (up to 3 attempts). Non-transient errors like 400 Bad Request are NOT
retried — they propagate immediately because retrying them will never succeed.

**Layer 2: Response caching.** Every call is cached in Redis with a 1-hour TTL.
The cache key is `SHA256(model_name + prompt + system_instruction)`. Identical
prompts — which are common in extraction (same paper processed multiple times)
— hit the cache instead of the API. Cache hits return immediately with zero
cost and zero latency. If Redis is unavailable, the system degrades gracefully
and just makes more API calls.

**Layer 3: Cost tracking.** Every call records `input_tokens`, `output_tokens`,
and `cost_usd` in the `LLMResponse`. Prometheus counters
(`llm_requests_total`, `llm_input_tokens_total`) feed into Grafana dashboards
for real-time monitoring. If you see costs spiking, you can add a rate limiter
to `GeminiClient.generate()` as a single-point change.

In a production deployment, I would add two more layers: a client-side rate
limiter (e.g., token bucket at 50 RPM) in `GeminiClient`, and an async task
queue (Celery) that processes papers sequentially rather than concurrently, so
you never burst above the API quota.

---

### Q3: "What is your confidence scoring approach?"

**Answer:**

We compute confidence as a weighted combination of three independent signals:

1. **Dual-prompt consistency (40% weight):** We run the same extraction with
   two differently-framed prompts. If both produce the same findings, the
   findings are likely real. We fuzzy-match using `SequenceMatcher` with a
   0.6 threshold. Consistency = `matched_findings / max(count_A, count_B)`.

2. **Source grounding (40% weight):** Each finding includes a supporting quote.
   We verify that this quote exists in the source paper text. Exact substring
   match gives 1.0. Fuzzy match uses SequenceMatcher's ratio. We average the
   grounding scores across all findings.

3. **Completeness (20% weight):** `min(1.0, num_findings / 3)`. This rewards
   papers where the model found at least 3 findings. The weight is low because
   some papers genuinely have fewer findings.

The final formula: `confidence = 0.4 * consistency + 0.4 * avg_grounding + 0.2 * completeness`.

Below 0.7, the extraction is flagged `needs_review = True`. We chose 0.7
because it roughly corresponds to the threshold where at least one signal is
strong (0.4 * 1.0 = 0.4) and the other is moderate (0.4 * 0.5 = 0.2) with
decent completeness (0.2 * 0.5 = 0.1), summing to 0.7.

The approach is designed to be calibrated with user feedback. If users
frequently correct extractions at confidence 0.75, we know our threshold is
too low and can adjust it upward.

---

### Q4: "How does user feedback improve the system?"

**Answer:**

User feedback creates a closed loop between the extraction system and human
domain experts.

**The mechanism:** When a user sees an extraction they disagree with, they
submit a correction via `POST /api/v1/feedback` with the paper ID, field name,
original extracted value, and the corrected value. The system stores this and
counts corrections per `(paper_id, field_name)` pair.

**Threshold alerting:** When corrections for a specific field exceed 5 (the
threshold), the system logs a warning. In production, this would trigger a
PagerDuty alert or Slack notification. This catches systematic prompt failures
early — for example, if the methodology prompt consistently misses dataset
names.

**Four ways feedback improves the system:**

1. **Prompt tuning.** Patterns in corrections reveal prompt weaknesses. If
   users keep correcting the same type of extraction, the prompt needs to be
   reworded. The corrections are essentially free annotations showing exactly
   what the prompt gets wrong.

2. **Confidence calibration.** Feedback data lets you compute the actual error
   rate at each confidence level. If extractions at confidence 0.75 have a 30%
   correction rate, you know the threshold should be higher.

3. **Model drift detection.** A spike in corrections after a Gemini model update
   indicates the new model behaves differently. The feedback rate becomes an
   early warning system.

4. **Fine-tuning data.** Each (original, corrected) pair is a labelled training
   example. Accumulate enough pairs and you can fine-tune a smaller, cheaper
   model that matches the base model on your specific paper types.

The system is designed for progressive enhancement: MVP uses an in-memory store,
the PostgreSQL ORM model is already defined with indexes, and the Pydantic
schemas support both. The migration path is deliberately smooth.
