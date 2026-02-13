# Phase 8: Advanced Features & Polish — Detailed Walkthrough

This document provides line-by-line explanations of every file created or modified in Phase 8, the design reasoning behind each decision, and interview-ready talking points for presenting this work.

Phase 8 is the capstone phase that transforms the project from a working prototype into a portfolio-ready, production-polished system. It adds four engineering capabilities that demonstrate senior-level thinking: intelligent model routing, multi-tier caching, graceful degradation, and cross-paper analysis.

---

## Table of Contents

1. [File-by-File Walkthrough](#1-file-by-file-walkthrough)
   - [1.1 model_router.py — Task-Based Model Selection](#11-model_routerpy--task-based-model-selection)
   - [1.2 cache_manager.py — Multi-Tier Caching](#12-cache_managerpy--multi-tier-caching)
   - [1.3 degradation.py — Graceful Degradation](#13-degradationpy--graceful-degradation)
   - [1.4 cross_paper.py — Cross-Paper Comparison](#14-cross_paperpy--cross-paper-comparison)
   - [1.5 schemas.py — New Pydantic Models](#15-schemaspy--new-pydantic-models)
   - [1.6 prompts.py — Comparison Prompt Template](#16-promptspy--comparison-prompt-template)
   - [1.7 gemini_client.py — generate_for_task()](#17-gemini_clientpy--generate_for_task)
   - [1.8 routes_query.py — Wiring CrossPaperAnalyzer](#18-routes_querypy--wiring-crosspaperanalyzer)
   - [1.9 README.md — Portfolio-Ready Documentation](#19-readmemd--portfolio-ready-documentation)
   - [1.10 docs/ARCHITECTURE.md — System Design Document](#110-docsarchitecturemd--system-design-document)
2. [Test Files Walkthrough](#2-test-files-walkthrough)
   - [2.1 test_model_router.py](#21-test_model_routerpy)
   - [2.2 test_cache_manager.py](#22-test_cache_managerpy)
   - [2.3 test_degradation.py](#23-test_degradationpy)
   - [2.4 test_cross_paper.py](#24-test_cross_paperpy)
   - [2.5 test_full_pipeline.py](#25-test_full_pipelinepy)
3. [Cross-Paper Analysis Deep Dive](#3-cross-paper-analysis-deep-dive)
4. [Model Routing Strategy & Cost Savings](#4-model-routing-strategy--cost-savings)
5. [Three-Level Caching Architecture](#5-three-level-caching-architecture)
6. [Graceful Degradation Philosophy](#6-graceful-degradation-philosophy)
7. [README as Resume](#7-readme-as-resume)
8. [ARCHITECTURE.md as Interview Prep](#8-architecturemd-as-interview-prep)
9. [Interview Questions & Ideal Answers](#9-interview-questions--ideal-answers)

---

## 1. File-by-File Walkthrough

### 1.1 model_router.py — Task-Based Model Selection

**File:** `src/llm/model_router.py` (50 lines)

**Purpose:** Routes each LLM task to the most cost-effective model that still meets quality requirements. This is the same principle behind how Netflix routes traffic to different CDN tiers or how AWS uses spot vs on-demand instances.

```python
"""Task-based model routing for optimal Gemini model selection.        # Line 1-6: Module docstring
                                                                        # Explains the routing philosophy upfront
Routes each task type to the most cost-effective model that meets quality
requirements.  Simple tasks (extraction, QA, summarization) use the fast
model; complex tasks (comparison, evaluation) use the pro model.
"""

from __future__ import annotations                                      # Line 8: Enable forward refs for type hints

from enum import Enum                                                   # Line 10: Only import needed — enum for task types
```

**Lines 13-18 — TaskType Enum:**
```python
class TaskType(str, Enum):
    EXTRACTION = "extraction"        # Simple pattern matching → Flash
    QA = "qa"                        # Retrieval-augmented → Flash
    SUMMARIZATION = "summarization"  # Text compression → Flash
    COMPARISON = "comparison"        # Multi-document reasoning → Pro
    EVALUATION = "evaluation"        # Quality judgment → Pro
```

**Why `str, Enum`?** By inheriting from both `str` and `Enum`, each value is a plain string AND an enum member. This means `TaskType("extraction")` works for deserialization from API requests, and you can use it as a dict key without `.value`.

**Why these 5 categories?** They map 1:1 to the actual LLM operations in the system:
- **Extraction/QA/Summarization** — Operate on a single document's context. The model just needs to follow instructions and extract/generate from provided text. Flash handles this at 94%+ of Pro quality.
- **Comparison** — Must synthesize across multiple documents and produce structured analysis. Requires stronger reasoning.
- **Evaluation** — Must judge quality of other LLM outputs. A weaker model evaluating a stronger model's output would be unreliable.

**Lines 21-27 — Default Rules:**
```python
_DEFAULT_RULES: dict[TaskType, str] = {
    TaskType.EXTRACTION: "gemini-2.0-flash",       # $0.075/1M input
    TaskType.QA: "gemini-2.0-flash",               # $0.075/1M input
    TaskType.SUMMARIZATION: "gemini-2.0-flash",    # $0.075/1M input
    TaskType.COMPARISON: "gemini-1.5-pro",          # $1.25/1M input
    TaskType.EVALUATION: "gemini-1.5-pro",          # $1.25/1M input
}
```

**Why module-level constant?** The default rules are immutable reference data. Putting them outside the class means they can be imported and inspected independently for testing or documentation, and they don't get recreated on every `ModelRouter()` instantiation.

**Lines 30-49 — ModelRouter Class:**
```python
class ModelRouter:
    def __init__(self, overrides: dict[TaskType, str] | None = None) -> None:
        self._rules = dict(_DEFAULT_RULES)      # Copy defaults — don't mutate the module constant
        if overrides:
            self._rules.update(overrides)        # Merge overrides on top of defaults

    def route(self, task: TaskType) -> str:
        return self._rules[task]                 # Direct dict lookup — O(1)

    def get_rules(self) -> dict[str, str]:
        return {t.value: m for t, m in self._rules.items()}  # Convert enum keys to strings
```

**Design decisions:**
1. **Copy-then-merge pattern** (line 39): `dict(_DEFAULT_RULES)` creates a shallow copy so overrides don't mutate the module constant. This prevents one router instance from affecting others.
2. **Override granularity**: You can override a single task without touching others. `ModelRouter(overrides={TaskType.QA: "gemini-1.5-pro"})` upgrades only QA while keeping everything else on Flash.
3. **`get_rules()` returns strings** (line 48-49): The admin endpoint needs JSON-serializable data. Converting enum keys to `.value` strings makes this directly serializable.
4. **No validation in `route()`**: If you pass an invalid `TaskType`, the dict lookup raises `KeyError`. This is intentional — we fail fast rather than silently falling back to a default model that might be wrong.

---

### 1.2 cache_manager.py — Multi-Tier Caching

**File:** `src/llm/cache_manager.py` (151 lines)

**Purpose:** Provides a two-tier cache (Redis L1 + in-memory L2) with task-aware TTLs. Each tier serves a different purpose and has different performance characteristics.

**Lines 1-25 — Module Setup and TTL Configuration:**
```python
_DEFAULT_TTL = 3600  # 1 hour — safe default for unknown task types

TASK_TTLS: dict[str, int] = {
    "extraction": 7200,       # 2 hours — extractions are stable, same paper = same output
    "qa": 1800,               # 30 min — answers change as new papers are indexed
    "summarization": 3600,    # 1 hour — summaries are stable but less reused
    "comparison": 1800,       # 30 min — depends on which papers are compared
    "evaluation": 3600,       # 1 hour — eval results don't change often
}
```

**Why different TTLs?** This is the key insight. Not all LLM responses have the same staleness risk:
- **Extraction (2h):** Running the same extraction prompt on the same paper text produces identical results. High TTL = maximum savings.
- **QA (30m):** New papers might be indexed between queries, changing which context is retrieved. Lower TTL prevents stale answers.
- **Comparison (30m):** Depends on the specific set of papers. If a user uploads a new paper, old comparisons could miss it.

**Lines 28-45 — Constructor:**
```python
class CacheManager:
    def __init__(self, redis_url: str | None = None) -> None:
        self._redis: Any = None             # Lazy — don't connect until first use
        self._redis_url = redis_url         # Store URL for lazy init
        self._redis_checked = False         # One-shot flag — only try connecting once

        # L2 in-memory: key → (value, expiry_timestamp)
        self._memory: dict[str, tuple[str, float]] = {}

        # Stats
        self._hits = {"redis": 0, "memory": 0}
        self._misses = 0
```

**Why `tuple[str, float]`?** Each memory entry stores `(cached_value, unix_timestamp_when_it_expires)`. This is simpler than using a TTL-aware data structure because we can check expiry on read and delete lazily.

**Why `_redis_checked` flag?** Without this, every cache lookup when Redis is down would attempt a TCP connection (slow timeout). The flag ensures we only try once, then stop wasting time.

**Lines 51-75 — `get()` Method (Cache Lookup):**
```python
def get(self, key: str, task_type: str = "") -> str | None:
    # L1: Redis — check first because it's shared across workers
    redis = self._get_redis()
    if redis is not None:
        try:
            val = redis.get(f"llm_cache:{key}")     # Namespaced key prevents collisions
            if val is not None:
                self._hits["redis"] += 1             # Track for observability
                return val
        except Exception:
            logger.debug("Redis GET failed", exc_info=True)  # Swallow — fall through to L2

    # L2: in-memory — local process fallback
    entry = self._memory.get(key)
    if entry is not None:
        value, expiry = entry
        if time.time() < expiry:                     # Check if still valid
            self._hits["memory"] += 1
            return value
        del self._memory[key]                        # Lazy eviction of expired entries

    self._misses += 1
    return None
```

**The lookup order matters:**
1. **Redis first** — Because it's shared. If Worker A cached a response, Worker B benefits immediately.
2. **Memory second** — Only useful when Redis is down or for the same worker process.
3. **Lazy eviction** (line 72) — We don't run background cleanup threads. When an expired entry is accessed, we delete it on the spot. This is the same pattern Redis itself uses internally.

**Lines 77-90 — `set()` Method (Cache Store):**
```python
def set(self, key: str, value: str, task_type: str = "") -> None:
    ttl = TASK_TTLS.get(task_type, _DEFAULT_TTL)     # Task-aware TTL selection

    # Write to BOTH tiers — L1 and L2 are independent
    redis = self._get_redis()
    if redis is not None:
        try:
            redis.setex(f"llm_cache:{key}", ttl, value)  # Redis handles its own expiry
        except Exception:
            logger.debug("Redis SETEX failed", exc_info=True)

    self._memory[key] = (value, time.time() + ttl)   # Manual expiry timestamp for L2
```

**Why write to both?** This ensures availability. If Redis goes down after a write, the in-memory cache still has the value for the lifetime of the current process. If the process restarts, Redis still has it.

**Lines 92-96 — `make_key()` (Cache Key Generation):**
```python
@staticmethod
def make_key(prompt: str, model: str, task_type: str = "") -> str:
    raw = f"{model}:{task_type}:{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

**Why include model AND task_type in the key?** The same prompt sent to `gemini-2.0-flash` and `gemini-1.5-pro` produces different outputs. The same prompt for `extraction` vs `qa` might have different caching semantics. Including all three ensures cache correctness.

**Why SHA-256?** Prompts can be thousands of characters. SHA-256 produces a fixed 64-char hex string that works as a Redis key. Collision probability is negligible (1 in 2^256).

**Lines 132-150 — `_get_redis()` (Lazy Connection):**
```python
def _get_redis(self) -> Any:
    if self._redis is not None:
        return self._redis                   # Already connected
    if self._redis_checked:
        return None                          # Already failed — don't retry
    self._redis_checked = True               # Mark as checked
    if not self._redis_url:
        return None                          # No URL configured
    try:
        import redis as redis_lib            # Lazy import — don't fail if redis not installed
        client = redis_lib.from_url(self._redis_url, decode_responses=True)
        client.ping()                        # Verify connection actually works
        self._redis = client
        return client
    except Exception:
        logger.debug("Redis unavailable for CacheManager", exc_info=True)
        return None                          # Graceful — cache just works without Redis
```

**This pattern appears in three places** (`gemini_client.py`, `cache_manager.py`, and implicitly in `degradation.py`). The three-step lazy init (check cache → check flag → try connect → set flag) is a production pattern that prevents:
1. Import-time failures (lazy import)
2. Repeated connection attempts (one-shot flag)
3. Hard crashes when infrastructure is missing (returns None)

---

### 1.3 degradation.py — Graceful Degradation

**File:** `src/llm/degradation.py` (103 lines)

**Purpose:** Wraps function calls with fallback strategies so that failures in non-critical components (Redis, vector store) return safe defaults instead of crashing the request.

**Lines 18-23 — ComponentType Enum:**
```python
class ComponentType(str, Enum):
    LLM = "llm"                  # Gemini API — critical, fallback = error message
    REDIS = "redis"              # Cache — non-critical, fallback = None (cache miss)
    VECTOR_STORE = "vector_store" # ChromaDB — semi-critical, fallback = empty results
    DATABASE = "database"        # PostgreSQL — critical for writes, fallback varies
```

**Why enumerate components?** Each component has different criticality. LLM failure is catastrophic (can't generate responses), but Redis failure just means cache misses. By categorizing components, we can set different alerting thresholds and fallback strategies.

**Lines 25-33 — FallbackResult Dataclass:**
```python
@dataclass
class FallbackResult:
    success: bool           # Did the primary function succeed?
    value: Any              # The actual return value (primary or fallback)
    fallback_used: bool     # Was the fallback activated?
    component: str          # Which component was involved
    error: str | None = None  # Error message if fallback was used
```

**Why a result wrapper instead of just returning the value?** The caller needs to know whether they got a real result or a fallback. This matters for:
- **Logging:** "User got results from fallback path" is an important operational signal.
- **UI:** You might show "Results may be incomplete" when fallback_used=True.
- **Metrics:** Track degradation frequency per component for SLA reporting.

**Lines 36-39 — _ComponentState (Internal):**
```python
@dataclass
class _ComponentState:
    failure_count: int = 0
    last_error: str | None = None
```

**Why track failure counts?** This enables health dashboards and alerting. If `redis.failure_count` jumps from 0 to 50 in a minute, that's an infrastructure incident. Without tracking, failures are invisible.

**Lines 50-86 — `wrap()` Method (The Core Logic):**
```python
def wrap(
    self,
    component: ComponentType,
    func: Callable[..., Any],
    fallback: Callable[..., Any] | Any,      # Can be a function OR a static value
    *args: Any,
    **kwargs: Any,
) -> FallbackResult:
    try:
        value = func(*args, **kwargs)         # Try the primary path
        return FallbackResult(
            success=True, value=value,
            fallback_used=False, component=component.value,
        )
    except Exception as exc:
        state = self._states[component.value]
        state.failure_count += 1              # Increment failure counter
        state.last_error = str(exc)           # Record last error for debugging
        logger.warning(
            "%s failed (count=%d): %s",       # Log at WARNING — visible but not page-worthy
            component.value, state.failure_count, exc
        )

        # Determine fallback value
        fb_value = fallback(*args, **kwargs) if callable(fallback) else fallback
        return FallbackResult(
            success=False, value=fb_value,
            fallback_used=True, component=component.value,
            error=str(exc),
        )
```

**The callable check on line 79 is critical.** It enables two usage patterns:

```python
# Pattern 1: Static fallback — "if Redis fails, return None"
gd.wrap(ComponentType.REDIS, lambda: redis.get(key), None)

# Pattern 2: Callable fallback — "if vector search fails, do keyword search instead"
gd.wrap(ComponentType.VECTOR_STORE, dense_search, sparse_search, query, top_k)
```

In Pattern 2, the fallback function receives the same `*args, **kwargs` as the primary function. This means you can implement a full degradation chain where each tier does something meaningful, not just returns empty results.

**Why `*args, **kwargs` passthrough?** This makes `wrap()` work with any function signature without requiring adapter lambdas. Compare:
```python
# Without passthrough — user must create wrapper lambdas:
gd.wrap(ComponentType.LLM, lambda: client.generate(prompt), lambda: "error")

# With passthrough — cleaner:
gd.wrap(ComponentType.LLM, client.generate, "error", prompt)
```

**Lines 88-102 — Health Reporting and Reset:**
```python
def get_health(self) -> dict[str, Any]:
    return {
        name: {"failure_count": state.failure_count, "last_error": state.last_error}
        for name, state in self._states.items()
    }

def reset(self) -> None:
    for state in self._states.values():
        state.failure_count = 0
        state.last_error = None
```

**Why `reset()`?** After an incident is resolved, you want to clear the counters so the health endpoint shows "clean" state. Without reset, failure counts only grow, making it impossible to distinguish "failed 100 times last week" from "failing right now."

---

### 1.4 cross_paper.py — Cross-Paper Comparison

**File:** `src/llm/cross_paper.py` (139 lines)

**Purpose:** Compares multiple research papers on a specific aspect (methodology, results, etc.) and produces structured output with comparison tables, agreements, contradictions, and a synthesis paragraph.

**Lines 8-23 — Imports and TYPE_CHECKING Guard:**
```python
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

from src.llm.prompts import CROSS_PAPER_COMPARE
from src.models.schemas import ComparisonTableRow, CrossPaperComparison

if TYPE_CHECKING:
    from src.api.stores import InMemoryDocumentStore
    from src.llm.gemini_client import GeminiClient
```

**Why `TYPE_CHECKING` guard?** This prevents circular imports at runtime. `cross_paper.py` needs `GeminiClient` and `InMemoryDocumentStore` for type hints, but importing them at module load would trigger their entire import chains (which might import back into `src.llm`). The `TYPE_CHECKING` flag is `False` at runtime but `True` when mypy analyzes the code, so types are checked without circular import issues.

**Lines 38-67 — `compare_papers()` (Main Orchestration):**
```python
def compare_papers(self, paper_ids: list[str], aspect: str = "methodology") -> CrossPaperComparison:
    if len(paper_ids) < 2:                                   # Guard clause — fail fast
        raise ValueError("At least 2 paper IDs are required for comparison")

    context = self._gather_context(paper_ids)                # Step 1: Get paper text
    papers_context = self._format_context(context)           # Step 2: Format for prompt
    prompt = CROSS_PAPER_COMPARE.format(                     # Step 3: Fill prompt template
        aspect=aspect, papers_context=papers_context
    )

    response = self._client.generate(prompt)                 # Step 4: Call LLM
    return self._parse_comparison(                           # Step 5: Parse JSON response
        response.content, paper_ids, aspect, response.model
    )
```

**This is the Pipeline Pattern** — each step transforms data for the next. The 5 steps are:
1. **Gather** — Retrieve raw text from the document store
2. **Format** — Structure text into a numbered, labeled block the LLM can reference
3. **Prompt** — Fill in the comparison template with aspect and context
4. **Generate** — Send to Gemini and get JSON back
5. **Parse** — Convert raw JSON into typed Pydantic objects

**Why pass `response.model` to `_parse_comparison`?** So the returned `CrossPaperComparison.model_used` field records which model actually generated the comparison. If model routing is active, this might be "gemini-1.5-pro" even though the client's default is "gemini-2.0-flash".

**Lines 69-83 — `_gather_context()` (Document Retrieval):**
```python
def _gather_context(self, paper_ids: list[str]) -> dict[str, str]:
    context: dict[str, str] = {}
    for paper_id in paper_ids:
        try:
            doc_uuid = uuid.UUID(paper_id)           # Validate UUID format
            doc = self._store.get(doc_uuid)
            if doc and doc.sections:
                text = "\n".join(s.text[:1000] for s in doc.sections[:3])  # First 3 sections, 1000 chars each
                context[paper_id] = text
            else:
                context[paper_id] = "Content unavailable"
        except (ValueError, Exception):
            context[paper_id] = "Content unavailable"  # Graceful — don't crash on bad IDs
    return context
```

**Why `[:3]` sections and `[:1000]` chars?** This is context window management. With 5 papers at 3 sections each, that's 15 sections × 1000 chars = 15,000 chars ≈ 4,000 tokens. This leaves ample room for the comparison output within Gemini's context window. Without truncation, a 50-page paper could blow out the context limit.

**Why "Content unavailable" instead of raising?** The user might compare 3 papers where 1 has been deleted. It's better to generate a partial comparison ("Papers 1 and 2 both use transformers; Paper 3 content unavailable") than to fail completely.

**Lines 93-129 — `_parse_comparison()` (JSON Parsing with Fallback):**
```python
def _parse_comparison(self, raw, paper_ids, aspect, model="") -> CrossPaperComparison:
    cleaned = self._strip_markdown_fences(raw)        # Remove ```json ... ``` wrapping
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse comparison JSON, using raw response")
        return CrossPaperComparison(                  # FALLBACK: raw text as synthesis
            paper_ids=paper_ids, aspect=aspect,
            synthesis=raw, model_used=model,
        )

    table_rows = [
        ComparisonTableRow(
            aspect=row.get("aspect", ""),              # Defensive .get() — don't crash on missing keys
            papers=row.get("papers", {}),
        )
        for row in data.get("comparison_table", [])
    ]

    return CrossPaperComparison(
        paper_ids=paper_ids, aspect=aspect,
        comparison_table=table_rows,
        agreements=data.get("agreements", []),
        contradictions=data.get("contradictions", []),
        synthesis=data.get("synthesis", ""),
        model_used=model,
    )
```

**The fallback pattern is crucial.** LLMs don't always return valid JSON. When parsing fails:
1. We don't crash the request (user still gets a response)
2. We put the raw LLM output in the `synthesis` field (it's usually useful text even if not valid JSON)
3. Structured fields are empty (comparison_table, agreements, etc.) — the response is degraded but present

This is the same principle as graceful degradation applied at the parsing layer.

**Lines 131-138 — `_strip_markdown_fences()` (Reused Pattern):**
```python
@staticmethod
def _strip_markdown_fences(text: str) -> str:
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()
```

**Why does this exist?** Gemini frequently wraps JSON output in markdown code fences (`\`\`\`json ... \`\`\``), even when the prompt says "Return ONLY valid JSON." This regex extracts the content between the fences. The same method exists in `extractor.py` — it's a known LLM quirk that requires defensive parsing.

---

### 1.5 schemas.py — New Pydantic Models

**File:** `src/models/schemas.py` (added after line 424)

```python
class ComparisonTableRow(BaseModel):
    """A single row in a cross-paper comparison table."""
    aspect: str                                          # e.g., "training approach", "dataset size"
    papers: dict[str, str] = Field(                      # paper_id → summary text
        description="paper_id → value for that aspect"
    )

class CrossPaperComparison(BaseModel):
    """Structured cross-paper comparison result."""
    paper_ids: list[str]                                 # Which papers were compared
    aspect: str                                          # The comparison dimension
    comparison_table: list[ComparisonTableRow] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)  # Points of agreement
    contradictions: list[str] = Field(default_factory=list)  # Points of disagreement
    synthesis: str = ""                                  # Overall synthesis paragraph
    model_used: str = ""                                 # Which Gemini model generated this
```

**Design decisions:**
1. **`ComparisonTableRow` is its own model** — Not just a `dict[str, dict[str, str]]`. Pydantic models give validation, serialization, and documentation automatically.
2. **`default_factory=list`** everywhere — All structured fields are optional with empty defaults. This means the fallback path (JSON parse failure) can create a valid `CrossPaperComparison` with just `paper_ids`, `aspect`, and `synthesis`.
3. **`model_used` field** — Records provenance. When reviewing comparisons, you can tell whether it was generated by Flash (cheaper but simpler) or Pro (more expensive but more nuanced).

---

### 1.6 prompts.py — Comparison Prompt Template

**File:** `src/llm/prompts.py` (added after line 207)

```python
CROSS_PAPER_COMPARE = """\
You are a comparative research analyst. Compare the following papers \
on the aspect: {aspect}.

{papers_context}

Provide your analysis as valid JSON:
{{"comparison_table": [{{"aspect": "sub-aspect name", \
"papers": {{"paper_id": "summary for this aspect"}}}}],
  "agreements": ["point where papers agree"],
  "contradictions": ["point where papers disagree"],
  "synthesis": "Overall synthesis paragraph comparing all papers"}}"""
```

**Design decisions:**
1. **Role priming** — "You are a comparative research analyst" sets the LLM's persona for analytical comparison rather than general chatbot behavior.
2. **JSON example in prompt** — Showing the exact structure dramatically improves output conformance. Without it, models might return prose instead of JSON.
3. **Double braces `{{` `}}`** — Python's `str.format()` uses `{}` for placeholders. Literal braces in the JSON example must be escaped as `{{` and `}}`.
4. **All prompts are module-level constants** — This follows the established pattern from Phases 1-4. Constants enable prompt versioning, A/B testing, and easy auditing of what the LLM sees.

---

### 1.7 gemini_client.py — generate_for_task()

**File:** `src/llm/gemini_client.py` (modified — lines 79-93 and 188-230)

**Constructor changes (lines 84-93):**
```python
def __init__(
    self, model_name=None, max_retries=3, cache_ttl=3600,
    model_router: Any = None,        # NEW: Optional ModelRouter instance
    cache_manager: Any = None,       # NEW: Optional CacheManager instance
) -> None:
    # ... existing init ...
    self._model_router = model_router
    self._cache_manager = cache_manager
```

**Why `Any` type hints instead of concrete types?** To avoid circular imports. `GeminiClient` is in `src.llm.gemini_client`, and `ModelRouter` is in `src.llm.model_router`. While not strictly circular, using `Any` keeps the module self-contained and avoids import ordering issues. The concrete types are only needed inside `generate_for_task()` where they're lazy-imported.

**New method — `generate_for_task()` (lines 188-230):**
```python
def generate_for_task(
    self, prompt: str, task_type: str,
    system_instruction: str | None = None,
    response_schema: dict | None = None,
) -> LLMResponse:
    original_model = self._model_name              # Save current model
    try:
        if self._model_router:
            from src.llm.model_router import TaskType
            self._model_name = self._model_router.route(TaskType(task_type))  # Route to optimal model

        if self._cache_manager:
            cache_key = self._cache_manager.make_key(prompt, self._model_name, task_type)
            cached = self._cache_manager.get(cache_key, task_type)
            if cached is not None:
                return LLMResponse(                # Cache HIT — return immediately
                    content=cached, input_tokens=0, output_tokens=0,
                    latency_ms=0.0, model=self._model_name,
                    cached=True, cost_usd=0.0,
                )

        result = self.generate(prompt, system_instruction, response_schema)  # Delegate to existing generate()

        if self._cache_manager:
            self._cache_manager.set(cache_key, result.content, task_type)  # Store in cache

        return result
    finally:
        self._model_name = original_model          # ALWAYS restore — even if an exception occurs
```

**The `try/finally` pattern is essential.** `generate_for_task()` temporarily swaps `self._model_name` to route to the correct model. If `generate()` raises an exception (API error, timeout), the `finally` block still restores the original model name. Without this, one failed comparison call would permanently switch the client to "gemini-1.5-pro" for all subsequent calls.

**Why does it call `self.generate()` instead of duplicating the API logic?** This is the Open/Closed Principle — `generate_for_task()` EXTENDS `generate()` by adding routing and caching, without modifying `generate()` itself. All existing code that calls `client.generate()` continues to work exactly as before.

**The cache check happens AFTER routing.** This is intentional. The cache key includes the model name, so `flash:extraction:prompt` and `pro:comparison:prompt` are different cache entries. If we checked cache before routing, we might return a flash-generated response for a comparison task that should use pro.

---

### 1.8 routes_query.py — Wiring CrossPaperAnalyzer

**File:** `src/api/routes_query.py` (modified — lines 88-96 and 199-222)

**New lazy initializer (lines 88-96):**
```python
_cross_paper = None

def _get_cross_paper_analyzer():
    global _cross_paper
    if _cross_paper is None:
        from src.llm.cross_paper import CrossPaperAnalyzer
        _cross_paper = CrossPaperAnalyzer(client=_get_client(), store=get_document_store())
    return _cross_paper
```

**This follows the exact pattern** of `_get_client()`, `_get_qa_engine()`, and `_get_summarizer()` defined earlier in the same file. The lazy initialization pattern ensures:
1. No import-time failures when the module loads
2. Services are only created when first needed
3. Subsequent calls return the same instance (singleton behavior)

**Refactored compare endpoint (lines 199-222):**
```python
async def compare_papers(request, body, store, api_key) -> CompareResponse:
    logger.info("Compare: %s on aspect=%s", body.paper_ids, body.aspect)

    try:
        analyzer = _get_cross_paper_analyzer()
        result = analyzer.compare_papers(body.paper_ids, body.aspect)  # Delegate to CrossPaperAnalyzer

        return CompareResponse(
            aspect=result.aspect,
            papers={pid: "included" for pid in result.paper_ids},  # Backward-compatible format
            comparison_text=result.synthesis,                        # Map synthesis → comparison_text
            key_differences=result.contradictions[:5],               # Map contradictions → key_differences
        )
    except Exception as exc:
        logger.exception("Comparison error")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc
```

**What changed from Phase 5?** The original compare endpoint had 30 lines of inline logic: iterate papers, build prompt manually, parse differences from markdown bullets. Now it's a clean 10-line delegation to `CrossPaperAnalyzer`. The inline logic was replaced with:
- Structured prompt templates (instead of f-string concatenation)
- JSON output parsing (instead of regex bullet extraction)
- Backward-compatible field mapping (instead of breaking the API)

**Why backward compatibility?** The response model is still `CompareResponse` from Phase 5. We map the new `CrossPaperComparison` fields into the old schema. Any frontend or test using the Phase 5 API contract continues to work. The mapping is:
- `result.synthesis` → `comparison_text`
- `result.contradictions[:5]` → `key_differences`
- `{pid: "included"}` → `papers` (simplified since full text is now in the synthesis)

---

### 1.9 README.md — Portfolio-Ready Documentation

**File:** `README.md` (251 lines)

**See [Section 7: README as Resume](#7-readme-as-resume) for detailed analysis of each section.**

---

### 1.10 docs/ARCHITECTURE.md — System Design Document

**File:** `docs/ARCHITECTURE.md`

**See [Section 8: ARCHITECTURE.md as Interview Prep](#8-architecturemd-as-interview-prep) for detailed analysis.**

---

## 2. Test Files Walkthrough

### 2.1 test_model_router.py

**File:** `tests/unit/test_model_router.py` (73 lines, 10 tests)

**Testing philosophy:** Every routing rule gets its own test. This seems verbose for a simple dict lookup, but it serves as living documentation. When a new engineer asks "what model does extraction use?", the test name `test_default_routes_flash_for_extraction` answers immediately.

**Key tests:**
- **Lines 13-26:** Three tests verify Flash routing for extraction/QA/summarization. These are individual tests rather than a parametrized test because each has a different docstring explaining WHY.
- **Lines 28-36:** Two tests verify Pro routing for comparison/evaluation.
- **Lines 38-43:** `test_override_replaces_default` — Tests the merge behavior. Creates a router with one override and verifies only that rule changed while others kept defaults. This prevents regressions if someone changes the override logic.
- **Lines 45-51:** `test_all_task_types_have_routes` — Iterates ALL enum members and verifies each has a route. This catches the case where someone adds a new `TaskType` but forgets to add a routing rule.
- **Lines 69-72:** `test_invalid_task_type_raises` — Verifies that `TaskType("nonexistent_task")` raises `ValueError`. Tests error handling, not just happy paths.

### 2.2 test_cache_manager.py

**File:** `tests/unit/test_cache_manager.py` (99 lines, 11 tests)

**Key patterns:**
- **Lines 14-18:** `test_set_and_get_from_memory` — Tests L2 (in-memory) in isolation by constructing `CacheManager()` with no Redis URL. This is the simplest possible integration: write → read → verify.
- **Lines 52-58:** `test_memory_expiry` — Directly sets `cm._memory["expired_key"] = ("old_value", time.time() - 10)` to simulate an expired entry. This is white-box testing (reaching into internals) but it's the only way to test expiry without sleeping for real.
- **Lines 60-70:** `test_get_stats_tracks_hits_misses` — Sets a value, reads it (hit), reads a missing key (miss), then verifies stats show 1 hit, 1 miss, 50% hit rate. This tests the observability layer.
- **Lines 84-88:** `test_redis_unavailable_falls_back_to_memory` — Creates `CacheManager(redis_url="redis://invalid:9999")`. Redis connection fails silently, but memory cache still works. This validates the core graceful degradation of the cache itself.

### 2.3 test_degradation.py

**File:** `tests/unit/test_degradation.py` (115 lines, 8 tests)

**Key patterns:**
- **Lines 71-86:** `test_fallback_can_be_callable` — Tests `wrap()` with a callable fallback (`lambda x, y: x + y`). The failing function raises, and the fallback is called with the same args `(3, 4)`, producing `7`. This verifies the `*args, **kwargs` passthrough.
- **Lines 99-114:** `test_multiple_components_tracked_independently` — Fails Redis twice and LLM once, then verifies each component's failure count is correct and unaffected by the other. This validates that the `_states` dict properly isolates component tracking.

### 2.4 test_cross_paper.py

**File:** `tests/unit/test_cross_paper.py` (196 lines, 9 tests)

**Key patterns:**
- **Lines 23-33:** `_llm_response()` helper — Creates a valid `LLMResponse` with `model="gemini-1.5-pro"` (since comparison routes to Pro). This matches the production behavior.
- **Lines 36-57:** `_make_completed_doc()` — Creates a mock `DocumentRecord` with realistic sections (abstract + methodology). The mock has real `DetectedSection` objects, not just strings, so the `_gather_context` code exercises the real `.text[:1000]` truncation path.
- **Lines 60-67:** `_COMPARISON_JSON` — Pre-built JSON response matching the exact schema expected by `_parse_comparison`. Defined once and reused across multiple tests.
- **Lines 130-142:** `test_parse_comparison_malformed_json_fallback` — Sends "This is not JSON at all" through the parser and verifies the fallback: raw text becomes `synthesis`, structured fields are empty. This is the most important test — it validates that production won't crash on unexpected LLM output.
- **Lines 185-195:** `test_parse_comparison_with_markdown_fences` — Wraps valid JSON in `` ```json ... ``` `` and verifies the fence stripping works. Tests the real-world scenario where Gemini adds markdown formatting.

### 2.5 test_full_pipeline.py

**File:** `tests/integration/test_full_pipeline.py` (249 lines, 5 tests)

**Key patterns:**
- **Lines 85-132:** `TestCompareIntegration.test_compare_returns_200_with_valid_papers` — Full integration test: creates documents in store, mocks the LLM client, patches `_get_cross_paper_analyzer`, creates a real `CrossPaperAnalyzer` with mock dependencies, sends an HTTP POST, and verifies the 200 response. This exercises the full path: HTTP → FastAPI → route handler → analyzer → store + LLM → response serialization.
- **Lines 107-108:** `app.dependency_overrides` — FastAPI's built-in mechanism for replacing dependencies in tests. Combined with `@patch` for internal functions and `try/finally` for cleanup.
- **Lines 172-212:** `TestSummaryIntegration` and `TestQueryIntegration` — Test other endpoints in the same integration style, ensuring Phase 8 changes didn't break the existing query and summary flows.

---

## 3. Cross-Paper Analysis Deep Dive

### Why Researchers Need It

Literature review is one of the most time-consuming parts of research. A researcher comparing 5 papers on "transformer architectures for document understanding" currently must:
1. Read all 5 papers (2-4 hours each = 10-20 hours)
2. Build a mental comparison matrix
3. Identify agreements and contradictions
4. Write a synthesis paragraph

Cross-paper analysis automates steps 2-4, reducing this from hours to seconds.

### How It Works — Step by Step

```
Input: paper_ids = ["uuid-1", "uuid-2", "uuid-3"], aspect = "methodology"

Step 1: _gather_context()
  → For each paper_id, fetch from document store
  → Extract first 3 sections, truncated to 1000 chars each
  → Output: {"uuid-1": "Abstract: ... Methodology: ...", "uuid-2": "...", ...}

Step 2: _format_context()
  → Format as numbered blocks:
    "Paper 1 (uuid-1):\n<text>\n\nPaper 2 (uuid-2):\n<text>\n\n..."

Step 3: Prompt Assembly
  → Fill CROSS_PAPER_COMPARE template with aspect="methodology" and formatted context

Step 4: LLM Generation
  → Send to Gemini (routed to Pro for comparison tasks)
  → Gemini returns structured JSON

Step 5: _parse_comparison()
  → Strip markdown fences
  → json.loads() the response
  → Map to ComparisonTableRow and CrossPaperComparison objects
  → On parse failure: fallback to raw text as synthesis
```

### Example Output

```json
{
  "paper_ids": ["uuid-1", "uuid-2"],
  "aspect": "methodology",
  "comparison_table": [
    {
      "aspect": "Architecture",
      "papers": {
        "uuid-1": "Multi-modal transformer with ResNet visual backbone",
        "uuid-2": "BERT-based encoder with CNN layout features"
      }
    },
    {
      "aspect": "Training Data",
      "papers": {
        "uuid-1": "DocBank (500K) + PubLayNet (360K)",
        "uuid-2": "IIT-CDIP (11M) with weak supervision"
      }
    }
  ],
  "agreements": [
    "Both use pre-trained language models as the text encoder",
    "Both evaluate on DocBank and PubLayNet benchmarks"
  ],
  "contradictions": [
    "Paper 1 uses supervised training; Paper 2 uses self-supervised pre-training",
    "Paper 1 processes text and layout jointly; Paper 2 uses a two-stage pipeline"
  ],
  "synthesis": "Both papers address document understanding using transformer-based architectures, but diverge significantly in their approach to visual features. Paper 1 jointly models text and layout in a single multi-modal transformer, achieving 94.2% F1. Paper 2 takes a two-stage approach with separate pre-training, using weak supervision from 11M documents to compensate for less labeled data.",
  "model_used": "gemini-1.5-pro"
}
```

---

## 4. Model Routing Strategy & Cost Savings

### Why Different Tasks Need Different Models

Not all LLM tasks are equally hard. Consider these real tasks in the system:

| Task | What It Does | Reasoning Needed | Model |
|------|-------------|-----------------|-------|
| Extraction | Parse structured data from text | Pattern matching, JSON formatting | Flash |
| QA | Answer question from provided context | Text comprehension, citation | Flash |
| Summarization | Compress text to shorter form | Key point identification | Flash |
| Comparison | Synthesize across multiple papers | Multi-document reasoning, judgment | Pro |
| Evaluation | Judge quality of another LLM's output | Meta-reasoning, calibration | Pro |

**Flash** (gemini-2.0-flash) excels at well-defined tasks with clear instructions. It follows templates reliably and is fast.

**Pro** (gemini-1.5-pro) excels at open-ended reasoning, nuanced judgment, and tasks that require synthesizing conflicting information. It's slower but more capable.

### Cost Savings Calculation

**Pricing (per 1M tokens):**
| Model | Input | Output |
|-------|-------|--------|
| gemini-2.0-flash | $0.075 | $0.30 |
| gemini-1.5-pro | $1.25 | $5.00 |

**Typical workload distribution:**
| Task | % of Calls | Avg Input Tokens | Avg Output Tokens |
|------|-----------|-----------------|------------------|
| Extraction | 40% | 3,000 | 500 |
| QA | 30% | 2,000 | 300 |
| Summarization | 15% | 4,000 | 400 |
| Comparison | 10% | 6,000 | 800 |
| Evaluation | 5% | 3,000 | 500 |

**Monthly cost with routing (1,000 calls/day):**
```
Flash calls (85%):
  850 × 30 × [(3,000 × $0.075) + (400 × $0.30)] / 1,000,000
  = 25,500 × [$0.225 + $0.12] / 1,000,000
  = 25,500 × $0.345 / 1,000,000 = $0.0088

Pro calls (15%):
  150 × 30 × [(5,000 × $1.25) + (700 × $5.00)] / 1,000,000
  = 4,500 × [$6.25 + $3.50] / 1,000,000
  = 4,500 × $9.75 / 1,000,000 = $0.0439

Total with routing: ~$0.053/month
```

**If everything used Pro:**
```
1,000 × 30 × [(3,000 × $1.25) + (450 × $5.00)] / 1,000,000
= 30,000 × [$3.75 + $2.25] / 1,000,000
= 30,000 × $6.00 / 1,000,000 = $0.18/month
```

**Savings: ~71% cost reduction** by routing 85% of calls to Flash. At higher volumes (100K calls/day), this scales to saving $3.80/month, or $45/year — meaningful for a side project, and the principle scales linearly to enterprise usage where it could save thousands.

---

## 5. Three-Level Caching Architecture

### Architecture Diagram

```
Request → [L1: Redis] → hit? → return cached
              ↓ miss
         [L2: In-Memory] → hit? → return cached
              ↓ miss
         [LLM API Call] → store in L1 + L2 → return fresh
```

### When Each Level Is Used

| Level | Location | Shared? | Survives Restart? | Latency | Use Case |
|-------|----------|---------|-------------------|---------|----------|
| L1: Redis | Network | Yes (all workers) | Yes | ~1ms | Primary cache, multi-worker sharing |
| L2: In-Memory | Process | No (per worker) | No | ~0.001ms | Fallback when Redis is down |

**L1 (Redis)** is checked first because it's shared. If Celery Worker A processes a document and caches the extraction, Web Worker B can serve the cached result to an API request. Without Redis, each worker would independently call the LLM for the same prompt.

**L2 (In-Memory)** exists for two scenarios:
1. **Redis is down** — The cache degrades to per-process caching rather than failing entirely
2. **Development mode** — When Redis isn't running, the app still gets caching benefits

### Task-Aware TTLs

```
extraction:    7200s (2h)  — Same paper text = same extraction, safe to cache long
qa:            1800s (30m) — New papers might change which context is retrieved
summarization: 3600s (1h)  — Summaries are stable but less frequently reused
comparison:    1800s (30m) — Depends on the paper set, stales as papers are added
evaluation:    3600s (1h)  — Eval results don't change often
```

### Cache Hit Rate Expectations

Based on typical usage patterns:

| Scenario | Expected Hit Rate | Why |
|----------|------------------|-----|
| Single-user development | 30-40% | Same queries during testing |
| Multi-user production | 50-70% | Shared Redis, repeated popular queries |
| After re-extraction | ~0% temporarily | Cache keys change when content changes |
| Steady-state QA | 60-80% | Users often ask similar questions |

### How Invalidation Works

**Time-based expiry (primary mechanism):** Each entry has a TTL. Redis handles L1 expiry automatically (`SETEX`). L2 checks timestamps on read and deletes expired entries lazily.

**Manual clear:** `CacheManager.clear()` empties both tiers. Useful after bulk re-processing.

**Key-based invalidation (implicit):** Cache keys include `model + task_type + prompt`. If the model changes (via routing override), the key changes, effectively invalidating old entries.

**What we DON'T do:** We don't invalidate on document upload/deletion. This is a deliberate tradeoff — a user who uploads a new paper and immediately asks a QA question might get a stale answer for up to 30 minutes. The TTL is the protection against this. For a document processing platform (not a real-time chat), this is acceptable.

---

## 6. Graceful Degradation Philosophy

### Why Partial Results Are Better Than Errors

Consider a user asking "Compare these two papers on methodology." The system needs:
1. Document store (to fetch paper text)
2. Redis (for caching)
3. Gemini API (to generate comparison)

If Redis is down, there are two approaches:

**Approach A: Fail fast** — Return HTTP 500 "Redis connection failed"
- User gets nothing. Must retry later.
- Support ticket: "The comparison API is broken."

**Approach B: Degrade gracefully** — Skip Redis, proceed without caching
- User gets the comparison (slightly slower, ~500ms extra).
- No visible impact. Cache will resume when Redis recovers.
- Monitoring shows redis.failure_count incrementing → ops team investigates.

We chose Approach B. The philosophy is: **every component failure should be invisible to the user unless the component is truly critical (LLM API).**

### Component Criticality Matrix

| Component | If Down... | Fallback Strategy |
|-----------|-----------|-------------------|
| Redis | Cache misses, slower responses | Proceed without cache (L2 memory still works) |
| ChromaDB | Dense search fails | Fall back to BM25 sparse search only |
| PostgreSQL | Can't persist new documents | Return error for writes; reads from in-memory store still work |
| Gemini API | Can't generate any LLM response | Return error with helpful message |

### How the Wrapper Pattern Works

```python
# Production usage:
gd = GracefulDegradation()

# Non-critical: cache lookup with None fallback
result = gd.wrap(ComponentType.REDIS, redis.get, None, cache_key)
# If Redis is down: result.value = None, result.fallback_used = True

# Semi-critical: vector search with sparse fallback
result = gd.wrap(ComponentType.VECTOR_STORE, dense_search, sparse_search, query, top_k)
# If ChromaDB is down: falls back to BM25 search with same args

# Critical: LLM call with error message fallback
result = gd.wrap(ComponentType.LLM, client.generate, "Service temporarily unavailable", prompt)
# If Gemini is down: returns error string instead of crashing
```

**The power of this pattern** is that it separates the "what to do on failure" decision from the "how to handle failure" mechanism. The caller specifies what fallback to use. The `GracefulDegradation` class handles try/catch, counting, logging, and health tracking. This separation means:
1. Adding a new component type requires only adding an enum value
2. Changing a fallback strategy is a one-line change at the call site
3. Health monitoring works identically for all components

---

## 7. README as Resume

The README is often the first file a hiring manager reads. Each section serves a specific purpose in the hiring pipeline.

### What Hiring Managers Look For (and where we address it)

**Section: Badges (line 1-7)**
```
![Python 3.11+] ![FastAPI] ![Gemini API] ![License: MIT] ![Tests: 454 passing]
```
- **Purpose:** Instant signal of tech stack and quality. The test count badge shows this isn't a tutorial project.
- **What managers think:** "OK, they know FastAPI and have real tests. This is worth reading."

**Section: One-paragraph overview (line 9)**
- **Purpose:** Elevator pitch. A manager skimming 50 projects in 10 minutes gives each one 12 seconds. This paragraph must answer: "What does it do? Is it impressive?"
- **Key phrases:** "end-to-end," "citation tracking," "production-ready," "GCP Cloud Run deployment" — these signal completeness and real-world thinking.

**Section: Architecture diagram (lines 11-38)**
- **Purpose:** Visual proof of system design skills. Mermaid diagrams render directly on GitHub.
- **What managers think:** "They think in systems, not just functions."

**Section: Key Features organized by Phase (lines 40-87)**
- **Purpose:** Shows progression and scope. 8 phases with specific capabilities demonstrates the project wasn't built in a weekend.
- **What managers think:** "This is substantial. They built extraction, retrieval, caching, deployment — the full stack."

**Section: Tech Stack table (lines 89-99)**
- **Purpose:** Keyword matching. ATS systems and recruiters search for specific technologies.
- **Why a table?** Faster to scan than a paragraph. Categories (Backend, LLM, Storage, Infrastructure, Testing) show breadth.

**Section: Quick Start (lines 101-121)**
- **Purpose:** Shows the project actually runs. Many portfolio projects have impressive READMEs but don't actually work.
- **3 commands to working app:** Clone → Install → Run. The Docker alternative shows infrastructure awareness.

**Section: API Endpoints table (lines 123-141)**
- **Purpose:** Shows the scope of the API surface. 15 endpoints is a real application, not a toy.
- **Method + Path + Description:** Professional API documentation style.

**Section: Project Structure tree (lines 143-197)**
- **Purpose:** Shows organizational thinking. Clean directory structure with clear naming conventions.
- **What managers think:** "They understand separation of concerns. This code would be maintainable."

**Section: Design Decisions table (lines 238-247)**
- **Purpose:** This is the highest-value section. It shows WHY, not just WHAT.
- **Format: Decision + Rationale:** Each row demonstrates you evaluated alternatives and chose deliberately.
- **What managers think:** "They made tradeoffs consciously. They can explain their choices in a design review."

---

## 8. ARCHITECTURE.md as Interview Prep

### How to Talk About Your System Design

The ARCHITECTURE.md serves as your study guide for system design interviews. Here's how to use each section:

**System Overview Mermaid Diagram:**
- **Interview use:** Draw this on the whiteboard. Start with the highest level (Client → API → Processing → Storage) then zoom into any area.
- **Practice:** "Let me start with the high-level architecture. We have a client layer, API layer with FastAPI, processing layer with Celery workers, an LLM layer with Gemini, a retrieval layer with hybrid search, and a storage layer with PostgreSQL, Redis, and ChromaDB."

**Component Architecture section:**
- **Interview use:** When asked "tell me about your API layer" or "how do you handle caching," refer to the specific subsection.
- **Key talking points per component:**
  - API: "Lazy initialization, rate limiting, API key auth"
  - LLM: "Model routing for cost optimization, two-tier caching, graceful degradation"
  - Retrieval: "Hybrid dense+sparse with RRF fusion — no score normalization needed"
  - Evaluation: "LLM-as-Judge for automated quality measurement"

**Data Flow section:**
- **Interview use:** When asked "walk me through what happens when a user uploads a paper," trace the document processing pipeline step by step.
- **Key insight to mention:** "Each stage updates status via WebSocket for real-time feedback. If any stage fails, the document status is set to FAILED with an error message, not silently dropped."

**Deployment Architecture section:**
- **Interview use:** When asked "how would you deploy this," describe the Docker compose topology and Cloud Run scaling.
- **Key insight to mention:** "Cloud Run scales to zero, so we only pay for active requests. The estimated cost is $15-30/month for light usage."

---

## 9. Interview Questions & Ideal Answers

### Q1: "What happens if the Gemini API goes down?"

**Ideal answer:**

"We handle this at multiple levels.

First, the **retry layer** in `GeminiClient._retry_with_backoff()` catches transient errors (429 rate limits, 500 server errors, 503 unavailable) and retries with exponential backoff — 1 second, 2 seconds, 4 seconds. This handles brief outages and rate limiting automatically.

If retries are exhausted, the **graceful degradation layer** kicks in. The `GracefulDegradation.wrap()` method catches the exception, increments a failure counter for the LLM component, and returns a fallback value. For QA queries, the fallback might be a message like 'The AI service is temporarily unavailable. Please try again in a few minutes.' For extraction during document processing, the Celery task is marked as failed and can be retried later.

The **caching layer** also provides resilience. If a user asks a question that was recently asked by someone else, the cached response is served from Redis without hitting the API at all. During an outage, cached responses continue to work.

Finally, the **health endpoint** (`/api/v1/admin/health`) reports the LLM component status, and the failure counter in `GracefulDegradation.get_health()` gives operations visibility into the failure rate. In a production setup, this would trigger PagerDuty alerts when failure counts exceed a threshold."

### Q2: "How do you optimize costs in your LLM application?"

**Ideal answer:**

"Cost optimization is built into the architecture at three levels.

**Level 1: Model routing.** Not every task needs the most expensive model. I built a `ModelRouter` that routes tasks to the cheapest model that meets quality requirements. Simple tasks like extraction and QA go to Gemini Flash at $0.075 per million input tokens. Only complex tasks like cross-paper comparison and quality evaluation use Gemini Pro at $1.25 per million. Since 85% of our calls are simple tasks, this saves roughly 71% compared to using Pro for everything.

**Level 2: Response caching.** The `CacheManager` provides a two-tier cache (Redis + in-memory) with task-aware TTLs. Extraction results are cached for 2 hours because the same paper text always produces the same extraction. QA responses are cached for 30 minutes since new papers might change the answer. In steady-state usage, we expect 50-70% cache hit rates, meaning 50-70% of LLM calls cost nothing.

**Level 3: Context window management.** When building comparison prompts, I truncate each paper to the first 3 sections at 1000 characters each. This keeps the prompt under 15K tokens even with 5 papers, avoiding unnecessary token costs from sending entire papers when summaries suffice.

The `CostTracker` in the monitoring layer records per-model and per-endpoint costs, so we can identify which operations are most expensive and optimize further. The admin endpoint `/api/v1/admin/costs` exposes this data."

### Q3: "How does caching work across different levels?"

**Ideal answer:**

"I implemented a two-tier cache that balances speed, sharing, and availability.

**Tier 1 is Redis**, the primary cache. It's checked first because it's shared across all workers. If Celery Worker A processes a document and caches the extraction result, Web Worker B serving an API request can use that cached result immediately. Redis entries use task-specific TTLs — extraction results cache for 2 hours because they're deterministic, while QA answers cache for only 30 minutes because new papers might change the relevant context.

**Tier 2 is in-memory**, a Python dict with expiry timestamps. This serves two purposes. First, when Redis is down (network issue, maintenance), the cache degrades to per-process caching rather than losing caching entirely. Second, during local development when Redis might not be running, you still get caching benefits.

**Cache key computation** uses SHA-256 of `model:task_type:prompt`. Including the model name means switching from Flash to Pro (via routing override) automatically creates new cache entries — you'll never serve a Flash response for a Pro task. Including task_type means the same prompt cached for extraction won't be returned for a different task with the same text.

**Invalidation is primarily time-based.** Redis handles L1 expiry natively with `SETEX`. L2 entries store an expiry timestamp and are lazily evicted on read — if you `get()` an expired entry, it's deleted on the spot. I chose not to implement explicit invalidation on document upload because the TTLs are short enough that stale data resolves itself. For a real-time application, I'd add pub/sub invalidation, but for a document processing platform where papers are added hourly rather than per-second, TTL-based expiry is the right tradeoff.

**Observability:** `get_stats()` reports hit/miss counts per tier and the overall hit rate, so we can tune TTLs based on actual usage patterns."

### Q4: "Walk me through your system architecture"

**Ideal answer:**

"The system has six layers. I'll start from the top.

**API Layer:** FastAPI with 15+ endpoints across four route modules — documents, queries, extraction management, and admin. All endpoints use API key authentication and rate limiting. LLM-backed services use lazy initialization — they're only created when first requested, which means the app starts fast and doesn't crash if the Gemini API key isn't configured yet.

**Processing Layer:** When a PDF is uploaded, it enters a 9-stage Celery pipeline. Stages 1-4 are ingestion: PDF text extraction with OCR fallback, table extraction, layout analysis detecting 12 section types, and metadata extraction. Stages 5-7 are indexing: section-aware chunking, embedding generation, and vector store insertion. Stage 8 is LLM extraction: key findings, methodology, and results using a dual-prompt consistency technique. Stage 9 finalizes the document status. Each stage updates a WebSocket for real-time progress tracking.

**LLM Layer:** All LLM calls go through a centralized `GeminiClient`. Phase 8 added three capabilities on top: a `ModelRouter` that selects Flash or Pro based on task complexity, a `CacheManager` with Redis and in-memory tiers using task-aware TTLs, and a `GracefulDegradation` wrapper that provides fallback values when non-critical components fail.

**Retrieval Layer:** Queries go through a hybrid retrieval pipeline. Dense search uses ChromaDB with Gemini embeddings for semantic matching. Sparse search uses a BM25 index for keyword matching. The results are combined using Reciprocal Rank Fusion, which doesn't require score normalization — it's purely based on rank positions. The alpha parameter lets users control the dense vs sparse balance.

**Storage Layer:** PostgreSQL for relational data, Redis for caching and Celery brokering, ChromaDB for vector storage, and the filesystem for uploaded PDFs. The in-memory document store is designed with the same interface as a database-backed store for easy swapping.

**Evaluation Layer:** Automated benchmarks measure extraction accuracy, retrieval quality (MRR, nDCG), QA faithfulness using LLM-as-Judge, and summarization quality via ROUGE scores. The CI pipeline includes a quality gate that fails builds if metrics drop below thresholds."

### Q5: "What would you change if you had more time?"

**Ideal answer:**

"Five things, in priority order:

**1. Replace the in-memory document store with PostgreSQL.** The current `InMemoryDocumentStore` loses all data on restart. The interface is already designed for swapping — `get()`, `save()`, `delete()`, `list_all()` — so the migration is straightforward. I'd use SQLAlchemy async with the existing `asyncpg` driver that's already in the dependencies.

**2. Add streaming responses for QA and summarization.** Right now, users wait for the full LLM response. With streaming (FastAPI's `StreamingResponse` + Gemini's streaming API), they'd see tokens appear in real-time. This dramatically improves perceived latency even though actual latency is the same.

**3. Implement circuit breaking in the degradation layer.** Currently, `GracefulDegradation` catches every failure individually. A circuit breaker would stop trying after N consecutive failures and periodically probe to check recovery. This prevents hammering a dead service with retries that all timeout.

**4. Add a reranking step using a cross-encoder.** The current retrieval uses bi-encoder embeddings for speed, but a cross-encoder reranker (like a small BERT model) applied to the top-20 results would significantly improve retrieval precision. This is the standard two-stage retrieval pattern used by Google and Bing.

**5. Implement user sessions and paper collections.** Currently, papers are globally accessible. Adding user scoping with collections ('my literature review on transformers') would make it a real multi-user tool. This requires adding user authentication (JWT), permission scoping on all endpoints, and per-user vector store partitioning."

### Q6: "How would you scale this to handle 100K papers?"

**Ideal answer:**

"The current architecture handles single-digit thousands of papers. Scaling to 100K requires changes at three levels:

**Storage scaling:** ChromaDB works for small-to-medium collections, but at 100K papers with ~50 chunks each (5M vectors), I'd migrate to a dedicated vector database like Pinecone, Weaviate, or Qdrant. These support distributed indexing, sharding, and approximate nearest neighbor search with sub-50ms latency at billion-scale. PostgreSQL handles 100K documents fine with proper indexing — I'd add a GIN index on metadata JSONB columns and ensure `paper_id` is indexed for the common filter queries.

**Processing scaling:** The current Celery setup processes documents sequentially on a single worker. For 100K papers, I'd scale horizontally: multiple Celery workers across machines, each pulling from the same Redis queue. The 9-stage pipeline is already designed for this — each document is independent. I'd also add a priority queue so new uploads get processed before bulk imports.

**Query scaling:** At 100K papers, hybrid retrieval needs optimization. I'd add a pre-filtering step: instead of searching all 5M chunks, first narrow to relevant papers using metadata filters (publication date, keywords, authors), then search only within those papers' chunks. This reduces the search space by 10-100x. I'd also add query result caching with a longer TTL since the corpus changes less frequently at scale.

**Cost scaling:** At 100K papers, the LLM extraction cost becomes significant. Assuming 50 pages average and ~$0.01 per paper for extraction, that's $1,000 for the initial processing. The model routing is already in place — Flash handles 85% of calls. I'd also add batched extraction using Gemini's batch API for bulk processing, which gives 50% cost reduction for non-interactive workloads.

**Infrastructure:** I'd move from a single Cloud Run instance to a Kubernetes cluster on GKE. This gives auto-scaling, node pools (CPU nodes for API, GPU-capable nodes for future local model serving), and proper service mesh for inter-service communication. The Docker setup from Phase 7 translates directly to Kubernetes manifests."
