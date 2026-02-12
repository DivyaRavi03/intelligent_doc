# Phase 3 Walkthrough: Hybrid Retrieval System

This document explains every file, every class, every method, and every design
decision in Phase 3. It covers how hybrid retrieval works, why it beats pure
vector search, and how Reciprocal Rank Fusion and two-stage reranking produce
production-quality results. Read this before a technical interview and you will
be able to talk through any line of code with confidence.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Retrieval Pipeline Diagram](#2-retrieval-pipeline-diagram)
3. [Why Hybrid Retrieval Beats Pure Vector Search](#3-why-hybrid-retrieval-beats-pure-vector-search)
4. [How Reciprocal Rank Fusion Works](#4-how-reciprocal-rank-fusion-works)
5. [Why Two-Stage Retrieval Is the Production Standard](#5-why-two-stage-retrieval-is-the-production-standard)
6. [Why Query Expansion Helps](#6-why-query-expansion-helps)
7. [File-by-File Deep Dive](#7-file-by-file-deep-dive)
   - [src/retrieval/bm25_index.py](#71-srcretrievalbm25_indexpy)
   - [src/retrieval/hybrid_retriever.py](#72-srcretrievalhybrid_retrieverpy)
   - [src/retrieval/reranker.py](#73-srcretrievalrerankerpy)
   - [src/retrieval/query_processor.py](#74-srcretrievalquery_processorpy)
   - [tests/unit/test_bm25_index.py](#75-testsunittest_bm25_indexpy)
   - [tests/unit/test_hybrid_retriever.py](#76-testsunittest_hybrid_retrieverpy)
   - [tests/unit/test_reranker.py](#77-testsunittest_rerankerpy)
   - [tests/unit/test_query_processor.py](#78-testsunittest_query_processorpy)
   - [tests/conftest.py (Phase 3 additions)](#79-testsconftestpy-phase-3-additions)
8. [Phase 2 to Phase 3 to Phase 4 Connections](#8-phase-2-to-phase-3-to-phase-4-connections)
9. [Performance Characteristics](#9-performance-characteristics)
10. [Cross-Cutting Design Decisions](#10-cross-cutting-design-decisions)
11. [Interview Questions and Answers](#11-interview-questions-and-answers)

---

## 1. Architecture Overview

Phase 3 is the **retrieval layer**. It takes a user query and the indexed
chunks from Phase 2, and produces a ranked, reranked list of the most relevant
passages. This is the critical bridge between stored knowledge and the
question-answering system in Phase 4.

```
Layer             Responsibility
────────────────  ──────────────────────────────────────────────────
Query Processor   Classify query type, expand query, orchestrate pipeline
Hybrid Retriever  Fuse dense (vector) and sparse (BM25) results via RRF
BM25 Index        Sparse keyword matching for exact terms
Reranker          Gemini LLM scoring for semantic precision
```

Key architectural principles:

- **Hybrid search** — Neither dense nor sparse retrieval alone is sufficient.
  Dense excels at semantic understanding, sparse excels at exact keyword
  matching. Combining them covers both failure modes.
- **Two-stage retrieval** — Stage 1 (BM25 + vector) is fast and high-recall.
  Stage 2 (Gemini reranking) is slow but high-precision. This matches how
  Google, Bing, and every production search engine works.
- **Dependency injection** — Every component accepts its dependencies through
  the constructor. `HybridRetriever` takes a `VectorStore`, `EmbeddingService`,
  and `BM25Index`. This makes unit testing trivial (inject mocks) and allows
  swapping implementations without changing business logic.
- **Graceful degradation** — Every single component has fallback behaviour.
  If dense retrieval fails, sparse results are returned. If the reranker fails,
  RRF ordering is preserved. If query classification fails, `FACTUAL` is the
  default. The system never crashes, it just gets slightly less accurate.

---

## 2. Retrieval Pipeline Diagram

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                    QueryProcessor                         │
│                                                          │
│  1. classify_query() ──── Gemini ──── QueryType          │
│     │                                                    │
│     ▼                                                    │
│  2. Look up params  (alpha, top_k per QueryType)         │
│     │                                                    │
│     ├── METADATA? ────► short-circuit, return metadata    │
│     │                                                    │
│     ▼                                                    │
│  3. expand_query() ──── Gemini ──── [original, alt1, …]  │
│     │                                                    │
│     ▼                                                    │
│  4. _retrieve_and_merge()                                │
│     │                                                    │
│     │  For each expanded query:                          │
│     │  ┌──────────────────────────────────────────┐      │
│     │  │          HybridRetriever.retrieve()       │      │
│     │  │                                          │      │
│     │  │   ┌──────────┐      ┌──────────┐         │      │
│     │  │   │  Dense    │      │  Sparse  │         │      │
│     │  │   │  Vector   │      │  BM25    │         │      │
│     │  │   │  Search   │      │  Search  │         │      │
│     │  │   └─────┬─────┘      └─────┬────┘         │      │
│     │  │         │                  │               │      │
│     │  │         └──────┬───────────┘               │      │
│     │  │                ▼                           │      │
│     │  │    Reciprocal Rank Fusion (RRF)            │      │
│     │  │         alpha weighting                    │      │
│     │  └──────────────────────────────────────────┘      │
│     │                                                    │
│     │  Cross-query RRF merge (deduplicate + boost)       │
│     │                                                    │
│     ▼                                                    │
│  5. GeminiReranker.rerank()                              │
│     │  Single Gemini call → 1-10 scores                  │
│     │  final = 0.7 * rerank + 0.3 * rrf                  │
│     │                                                    │
│     ▼                                                    │
│  QueryResult (ranked passages ready for QA)              │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Why Hybrid Retrieval Beats Pure Vector Search

### The problem with dense-only retrieval

Embedding models encode **semantic meaning** into high-dimensional vectors.
Two sentences with similar meaning will have nearby vectors even if they share
no words. This is powerful, but it has blind spots:

**Blind spot 1: Exact technical terms.** Search for "BM25Okapi" — the embedding
model has likely never seen this specific term in training and will map it to
something generic. BM25 matches it character-for-character.

**Blind spot 2: Author names and identifiers.** Search for "Smith et al. 2023"
— dense embeddings treat this as generic academic language. BM25 matches the
literal string "Smith" in the references section.

**Blind spot 3: Acronyms and abbreviations.** Search for "mAP on PubLayNet" —
the embedding may not distinguish mAP from MAP from map. BM25 finds the exact
match.

### The problem with sparse-only retrieval

BM25 counts term frequencies and inverse document frequencies. It has no
concept of meaning:

**Blind spot 1: Paraphrasing.** Search for "what is the main contribution" — the
paper may never use the word "contribution" but says "we propose" or "our key
innovation". Dense retrieval understands these are semantically equivalent.

**Blind spot 2: Conceptual queries.** Search for "how does the model handle
long documents?" — BM25 can only match the individual words. Dense retrieval
understands this is asking about a processing strategy.

**Blind spot 3: Vocabulary mismatch.** The user says "accuracy", the paper says
"F1 score". Dense embeddings know these are related; BM25 sees zero overlap.

### The hybrid advantage

| Query | Dense | Sparse | Hybrid |
|-------|-------|--------|--------|
| "transformer attention" | Good — understands concept | Excellent — exact match | Excellent |
| "what is the main contribution" | Excellent — semantic match | Poor — no keyword overlap | Excellent |
| "Smith et al. 2023" | Poor — generic embedding | Excellent — exact match | Excellent |
| "how does dropout regularize" | Excellent — conceptual | Moderate — partial match | Excellent |
| "94.2% F1 on DocBank" | Poor — numbers collapse | Excellent — exact match | Excellent |

Hybrid retrieval covers both failure modes. The `alpha` parameter controls the
blend: `alpha=0.7` (default) gives 70% weight to dense and 30% to sparse.
For factual queries with specific terms, `alpha=0.5` gives equal weight to both.

---

## 4. How Reciprocal Rank Fusion Works

### The core idea

You have two ranked lists from two different retrieval systems. You want to
combine them into a single list. Simply averaging scores does not work because
the score scales are completely different (cosine similarity 0-1 vs BM25 scores
that can be any positive number).

RRF solves this by converting scores to ranks, then combining ranks. It was
introduced by Cormack et al. (2009) and has become the standard fusion method.

### The formula

```
RRF_score(doc) = alpha * 1/(k + rank_dense) + (1-alpha) * 1/(k + rank_sparse)
```

Where:
- `k = 60` is a smoothing constant (standard value from the original paper)
- `rank_dense` = the document's position in the dense results (1-based)
- `rank_sparse` = the document's position in the sparse results (1-based)
- If a document is missing from one list, it gets `rank = 1000` (penalty)

### Step-by-step numeric example

Suppose we search for "transformer attention mechanism" and get these results:

**Dense (vector) results:**
| Rank | Chunk ID | Text (abbreviated) |
|------|----------|-------------------|
| 1 | c3 | "multi-head attention in transformer architectures" |
| 2 | c1 | "transformer model with self-attention" |
| 3 | c5 | "attention mechanisms for sequence modelling" |
| 4 | c2 | "deep learning for NLP tasks" |

**Sparse (BM25) results:**
| Rank | Chunk ID | Text (abbreviated) |
|------|----------|-------------------|
| 1 | c1 | "transformer model with self-attention" |
| 2 | c3 | "multi-head attention in transformer architectures" |
| 3 | c4 | "transformer-based encoder architecture" |
| 4 | c5 | "attention mechanisms for sequence modelling" |

Now compute RRF scores with `alpha = 0.7`, `k = 60`:

**Chunk c3** (rank 1 in dense, rank 2 in sparse):
```
RRF = 0.7 * 1/(60+1) + 0.3 * 1/(60+2)
    = 0.7 * 0.01639  + 0.3 * 0.01613
    = 0.01148 + 0.00484
    = 0.01632
```

**Chunk c1** (rank 2 in dense, rank 1 in sparse):
```
RRF = 0.7 * 1/(60+2) + 0.3 * 1/(60+1)
    = 0.7 * 0.01613  + 0.3 * 0.01639
    = 0.01129 + 0.00492
    = 0.01621
```

**Chunk c5** (rank 3 in dense, rank 4 in sparse):
```
RRF = 0.7 * 1/(60+3) + 0.3 * 1/(60+4)
    = 0.7 * 0.01587  + 0.3 * 0.01563
    = 0.01111 + 0.00469
    = 0.01580
```

**Chunk c4** (rank 1000 in dense (missing!), rank 3 in sparse):
```
RRF = 0.7 * 1/(60+1000) + 0.3 * 1/(60+3)
    = 0.7 * 0.00094     + 0.3 * 0.01587
    = 0.00066 + 0.00476
    = 0.00542
```

**Chunk c2** (rank 4 in dense, rank 1000 in sparse (missing!)):
```
RRF = 0.7 * 1/(60+4)    + 0.3 * 1/(60+1000)
    = 0.7 * 0.01563     + 0.3 * 0.00094
    = 0.01094 + 0.00028
    = 0.01122
```

**Final fused ranking:**
| Rank | Chunk ID | RRF Score | Why |
|------|----------|-----------|-----|
| 1 | c3 | 0.01632 | Top-2 in both lists |
| 2 | c1 | 0.01621 | Top-2 in both lists |
| 3 | c5 | 0.01580 | Present in both, but lower |
| 4 | c2 | 0.01122 | Dense-only, sparse penalty |
| 5 | c4 | 0.00542 | Sparse-only, dense penalty |

### Why k = 60?

The `k` constant controls how much the top ranks dominate. With `k = 60`:
- Rank 1 contributes `1/61 = 0.01639`
- Rank 2 contributes `1/62 = 0.01613`
- The difference between rank 1 and rank 2 is only `0.00026`

This means being ranked #1 vs #2 makes very little difference, which is
desirable because both retrieval systems have noise. With a small `k` (like
`k = 1`), rank 1 gets `1/2 = 0.5` and rank 2 gets `1/3 = 0.33` — a 34%
penalty for being just one position lower. That is too aggressive.

The value `k = 60` is from the original Cormack et al. paper and is now the de
facto standard. It has been validated empirically across hundreds of IR
benchmarks.

---

## 5. Why Two-Stage Retrieval Is the Production Standard

### The speed vs. accuracy trade-off

| Stage | What | Speed | Accuracy | Candidates |
|-------|------|-------|----------|------------|
| Stage 1: Retrieval | BM25 + Vector + RRF | ~25ms | Good recall | 20 candidates |
| Stage 2: Reranking | Gemini LLM scoring | ~500ms | Excellent precision | 5 final results |

**Stage 1 is fast but imprecise.** BM25 is a bag-of-words model. Vector search
uses approximate nearest neighbours (HNSW). Both can retrieve 20 candidates in
under 25ms combined, but some of those candidates will be irrelevant.

**Stage 2 is slow but precise.** An LLM reads every candidate passage alongside
the query and judges semantic relevance on a 1-10 scale. This is 20-50x slower
than Stage 1, but it can distinguish between a passage that mentions
"transformer" (the electrical component) and one about "transformer" (the neural
architecture).

### Why not just use the LLM for everything?

Cost and latency. If you have 10,000 chunks in your index, asking Gemini to
score all 10,000 would cost ~$2 per query and take 30+ seconds. Instead:

1. Stage 1 narrows 10,000 chunks down to 20 candidates (~25ms, $0)
2. Stage 2 scores 20 candidates with Gemini (~500ms, ~$0.001)

This 500x reduction in LLM calls is why every production search system uses
two-stage retrieval: Google (fast index → neural reranker), Bing (BM25 → BERT
reranker), Elasticsearch (BM25 → learned sparse → cross-encoder reranker).

### Why not just increase top_k in Stage 1?

Diminishing returns. Going from top-5 to top-20 in Stage 1 improves recall by
~15%. Going from top-20 to top-100 improves recall by only ~3% while making
Stage 2 much more expensive. The sweet spot is 15-25 candidates per source.

---

## 6. Why Query Expansion Helps

### The vocabulary mismatch problem

The user searches for "how does the model handle long documents". The paper
says "our approach uses a sliding window over extended input sequences". There
is zero keyword overlap and only moderate semantic similarity.

Query expansion generates alternative phrasings using Gemini:
- Original: "how does the model handle long documents"
- Expansion 1: "sliding window approach for extended input sequences"
- Expansion 2: "processing long text documents with transformers"

Now the system retrieves for all three queries. The original catches semantic
matches. Expansion 1 catches the exact terminology used in the paper. Expansion
2 bridges the gap with different vocabulary.

### How cross-query merge works

When multiple expanded queries return results, duplicates need to be merged.
A chunk that appears in results for 2 out of 3 queries is more likely relevant
than one appearing in only 1. The system uses another round of RRF:

```
cross_rrf(chunk) = (1/N) * SUM over queries of 1/(k + rank_in_query_i)
```

Where `N` is the number of queries. Chunks missing from a query's results get
`rank = 1000`. This naturally boosts chunks that multiple query formulations
agree on.

---

## 7. File-by-File Deep Dive

### 7.1 `src/retrieval/bm25_index.py`

This file provides sparse keyword retrieval using the BM25Okapi algorithm.

#### Lines 1-8: Module docstring

```python
"""BM25 sparse keyword index for document chunks."""
```

The docstring explains the class purpose and its role in the hybrid retrieval
system. BM25 complements dense search for exact keyword matching.

#### Lines 10-18: Imports and logger

```python
from src.models.schemas import EnrichedChunk, SectionType
from src.retrieval.vector_store import SearchResult
```

`EnrichedChunk` is the input (from Phase 2 chunking). `SearchResult` is the
output format — same dataclass used by `VectorStore.search()`. This is a key
design decision: by returning `SearchResult`, the BM25 index has the same
interface as the vector store, making the hybrid retriever's job much simpler.

#### Lines 21-34: `_STOPWORDS` frozenset

```python
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", ...
})
```

**Why a built-in frozenset instead of NLTK?** NLTK requires downloading a 30MB
data package (`nltk.download('stopwords')`). That adds a runtime dependency, a
Docker build step, and a potential point of failure. Our ~80-word frozenset
covers the most common English stopwords and is instantiated at import time with
zero I/O.

**Why `frozenset`?** Membership testing (`t not in _STOPWORDS`) is O(1) for
frozensets vs O(n) for lists. With potentially millions of tokens being checked,
this matters. `frozenset` also signals immutability — these words never change.

#### Lines 36-37: `_PUNCT_TABLE`

```python
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
```

Pre-builds a translation table that maps every punctuation character to `None`.
`str.translate()` is implemented in C and is 5-10x faster than regex for
character-level removal. This table is built once at module import and reused
for every tokenization call.

#### Lines 40-51: `BM25Index.__init__`

```python
class BM25Index:
    def __init__(self, *, remove_stopwords: bool = True) -> None:
        self._remove_stopwords = remove_stopwords
        self._chunks: list[EnrichedChunk] = []
        self._bm25 = None  # rank_bm25.BM25Okapi (lazy)
```

The `*` forces `remove_stopwords` to be keyword-only, preventing accidental
positional use. The BM25Okapi instance is `None` until `build_index()` is
called — this is the lazy initialisation pattern used throughout the codebase.

#### Lines 57-74: `build_index`

```python
def build_index(self, chunks: list[EnrichedChunk]) -> None:
    if not chunks:
        raise ValueError("Cannot build BM25 index from empty chunk list")

    from rank_bm25 import BM25Okapi

    self._chunks = list(chunks)
    tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
    self._bm25 = BM25Okapi(tokenized_corpus)
```

**Lazy import of `rank_bm25`** — The import happens inside the method, not at
module level. This means the `rank_bm25` package is only loaded when someone
actually builds an index. If the BM25 feature is never used, the import never
happens. This pattern is consistent with how `google.generativeai` is imported
in the embedding service and reranker.

**`list(chunks)` defensive copy** — Prevents the caller from modifying the
original list after passing it. The BM25 index's internal state would become
inconsistent if the chunk list changed externally.

**BM25Okapi construction** — Takes a list of tokenized documents (each document
is a `list[str]`). Internally it computes term frequencies (TF), inverse
document frequencies (IDF), and document length normalization factors. The IDF
formula is: `log((N - n + 0.5) / (n + 0.5))` where `N` is total documents and
`n` is the number of documents containing the term.

#### Lines 76-154: `search`

The search method is the core of the module:

```python
raw_scores = self._bm25.get_scores(tokens)
```

`get_scores()` returns a numpy array with one BM25 score per document in the
corpus. This is O(V * L) where V is query vocabulary size and L is the average
document length.

```python
scored: list[tuple[int, float]] = [
    (idx, float(score))
    for idx, score in enumerate(raw_scores)
    if score > 0.0
]
```

**Filtering zero scores** — Documents with no matching terms get score 0.0.
Keeping them would waste time on sorting and metadata filtering. The
`float(score)` converts numpy float64 to Python float for consistent behavior.

```python
scored = self._filter_by_metadata(scored, paper_id, section_type)
```

**Post-score filtering** — Metadata filters are applied after scoring, not
before. This is deliberate: BM25Okapi computes scores for the entire corpus in
a single vectorized operation. Filtering the corpus first would require
rebuilding the index (changing IDF values). Post-filtering preserves correct IDF
computation.

```python
max_score = scored[0][1]
# ...
score=round(score / max_score, 4),
```

**Score normalization** — Raw BM25 scores are unbounded positive numbers that
depend on document length, query length, and corpus statistics. Dividing by the
maximum score normalizes to [0, 1]. The top result always gets score 1.0, which
makes scores comparable across different queries and combinable with dense
scores in the hybrid retriever.

#### Lines 168-187: `_tokenize`

```python
def _tokenize(self, text: str) -> list[str]:
    lowered = text.lower()
    cleaned = lowered.translate(_PUNCT_TABLE)
    tokens = cleaned.split()

    if self._remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    return tokens
```

Four steps: lowercase → strip punctuation → split on whitespace → remove
stopwords. This is deliberately simple. Research-grade BM25 systems might add
stemming (Porter stemmer) or lemmatization, but those add complexity and
external dependencies for marginal gains in a system that also has dense
retrieval as backup.

**Why `split()` instead of regex `\w+`?** `str.split()` with no arguments
splits on any whitespace and handles multiple spaces, tabs, and newlines
correctly. It is also faster than regex. Since we have already removed
punctuation, the result is the same.

#### Lines 189-208: `_filter_by_metadata`

```python
def _filter_by_metadata(self, scored, paper_id, section_type):
    if not paper_id and not section_type:
        return scored  # Fast path: no filtering needed
```

**Early return optimization** — When no filters are requested, skip the loop
entirely. This is the common case (most searches do not filter by metadata).

The filter loops through scored results and checks each chunk's `paper_id` and
`section_type` against the filter values. Both filters are AND-combined: a chunk
must match both to be included.

---

### 7.2 `src/retrieval/hybrid_retriever.py`

This file fuses dense and sparse retrieval using Reciprocal Rank Fusion.

#### Lines 1-23: Module-level constants

```python
_DEFAULT_ALPHA = 0.7   # Weight for dense retrieval (1.0 = all dense)
_DEFAULT_CANDIDATES = 20  # Candidates to retrieve from each source
_RRF_K = 60           # Smoothing constant for RRF
```

**Why `alpha = 0.7` default?** Research literature consistently shows that
semantic (dense) retrieval contributes more to overall quality than keyword
(sparse) retrieval for most query types. A 70/30 split favours dense while
still giving sparse enough weight to catch exact matches.

**Why 20 candidates per source?** This is 4x the typical final result count
(top_k=5). Retrieving too few misses relevant documents. Retrieving too many
wastes reranking budget. 20 is the empirical sweet spot validated in BEIR
benchmarks.

**Why `_RRF_K = 60`?** See Section 4 for the mathematical justification.

#### Lines 26-42: `RankedResult` dataclass

```python
@dataclass
class RankedResult:
    chunk_id: str
    text: str
    paper_id: str
    section_type: str
    # ... metadata fields ...
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
```

**Why a new dataclass instead of extending SearchResult?** `SearchResult` is
defined in `vector_store.py` (Phase 2) and has a single `score` field. The
hybrid retrieval pipeline needs five different scores tracking the result through
each stage. Modifying `SearchResult` would break Phase 2 code and tests.
Creating `RankedResult` keeps Phase 2 unchanged and provides a clear data model
for the multi-stage pipeline.

**Why is it defined here instead of in `schemas.py`?** Because `RankedResult` is
a retrieval-pipeline concept. It is first produced by `HybridRetriever` and
consumed by `GeminiReranker` and `QueryProcessor`. Putting it in `schemas.py`
would create a dependency from schemas to retrieval concepts, inverting the
dependency direction.

#### Lines 45-70: `HybridRetriever.__init__`

```python
class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        bm25_index: BM25Index,
        default_alpha: float = _DEFAULT_ALPHA,
        candidates_per_source: int = _DEFAULT_CANDIDATES,
    ) -> None:
```

**Dependency injection** — All three retrieval components are passed in as
constructor arguments. This is the single most important design decision in
Phase 3. Benefits:

1. **Testability** — Unit tests pass `MagicMock(spec=VectorStore)` instead of
   a real ChromaDB instance. Tests run in <0.1s with no database or API calls.
2. **Flexibility** — You can swap ChromaDB for Pinecone, or BM25Okapi for
   Elasticsearch, without changing any code in this file.
3. **Configuration** — `default_alpha` and `candidates_per_source` can be tuned
   per deployment without subclassing.

#### Lines 76-110: `retrieve`

```python
def retrieve(self, query, top_k=10, alpha=None, ...) -> list[RankedResult]:
    effective_alpha = alpha if alpha is not None else self._default_alpha
    n = self._candidates_per_source

    dense_results = self._get_dense_results(query, n, paper_id, section_type)
    sparse_results = self._get_sparse_results(query, n, paper_id, section_type)

    if not dense_results and not sparse_results:
        return []

    fused = self._reciprocal_rank_fusion(dense_results, sparse_results, effective_alpha)
    return fused[:top_k]
```

**`alpha is not None`** — Using `is not None` instead of truthiness because
`alpha=0.0` is a valid value (all-sparse mode). `if alpha:` would incorrectly
treat `0.0` as falsy and use the default.

**Both sources called independently** — Dense and sparse retrieval are
independent operations. If one fails, the other still returns results. The
early return on both-empty is the only case where we return nothing.

#### Lines 116-150: `_get_dense_results` and `_get_sparse_results`

```python
def _get_dense_results(self, query, n, paper_id, section_type):
    try:
        embedding = self._embedding_service.embed_query(query)
        return self._vector_store.search(embedding, n_results=n, ...)
    except Exception:
        logger.warning("Dense retrieval failed, falling back to sparse-only", exc_info=True)
        return []
```

**Catch-all `except Exception`** — This is unusual in production code but
correct here. Dense retrieval can fail for many reasons: API rate limit (Google
embedding API), network timeout, ChromaDB corruption, out-of-memory. All of
these should be logged and fallen back from, never propagated to the user.

**`_get_sparse_results` checks `is_built()`** — BM25 requires explicit
`build_index()` before searching. Unlike dense search (which uses a persistent
ChromaDB collection), BM25 is in-memory and must be rebuilt when the application
restarts. Checking `is_built()` avoids a cryptic error.

#### Lines 156-229: `_reciprocal_rank_fusion`

This is the core algorithm. Line by line:

```python
dense_rank: dict[str, int] = {
    r.chunk_id: rank + 1 for rank, r in enumerate(dense_results)
}
sparse_rank: dict[str, int] = {
    r.chunk_id: rank + 1 for rank, r in enumerate(sparse_results)
}
```

**1-based ranking** — `enumerate` gives 0-based indices. Adding 1 makes ranks
1-based. This matters because `1/(k+1)` and `1/(k+0)` are very different.
The RRF formula assumes 1-based ranks.

```python
all_chunks: dict[str, SearchResult] = {}
for r in dense_results:
    if r.chunk_id not in all_chunks:
        all_chunks[r.chunk_id] = r
for r in sparse_results:
    if r.chunk_id not in all_chunks:
        all_chunks[r.chunk_id] = r
```

**First-occurrence-wins deduplication** — A chunk that appears in both dense and
sparse results is stored once. Dense results are iterated first, so the dense
`SearchResult` object is preferred (it has the cosine similarity score as the
`score` field). This is a minor detail but ensures `dense_score` in the output
is populated from the actual dense result.

```python
for chunk_id, result in all_chunks.items():
    rd = dense_rank.get(chunk_id, missing_rank)
    rs = sparse_rank.get(chunk_id, missing_rank)
    rrf = alpha * (1.0 / (k + rd)) + (1.0 - alpha) * (1.0 / (k + rs))
```

The core formula. `.get(chunk_id, missing_rank)` returns `1000` for chunks
missing from that source, giving them a heavy penalty. The penalty rank of 1000
means `1/(60+1000) = 0.00094`, compared to rank 1's `1/(60+1) = 0.01639`. A
missing-from-one-source chunk gets ~17x less contribution from that source.

```python
rrf_score=round(rrf, 6),
```

**Rounding to 6 decimal places** — Avoids floating-point noise in comparisons
and test assertions. Six digits is more than enough precision for ranking.

---

### 7.3 `src/retrieval/reranker.py`

This file implements Gemini-based passage reranking.

#### Lines 1-27: Module-level constants

```python
_RERANK_WEIGHT = 0.7   # Weight for reranker score in final combination
_RRF_WEIGHT = 0.3      # Weight for original RRF score in final combination
_MAX_PASSAGE_CHARS = 500  # Truncate passages to stay within context limits
```

**Why 70/30 rerank/RRF weight?** The reranker (Gemini LLM) is more accurate
than RRF for determining relevance, so it gets the majority weight. But RRF
scores encode information the reranker does not see — specifically, the
agreement between dense and sparse retrieval. A chunk that both BM25 and vector
search rank highly is probably relevant even if the reranker gives it a moderate
score. The 30% RRF weight preserves this signal.

**Why truncate at 500 characters?** Gemini models have context limits. With 20
candidates at 500 chars each, the total prompt is ~10K characters. Sending full
chunk text (potentially 2K+ chars each) would use 40K+ characters, which risks
hitting context limits and increases latency/cost.

#### Lines 29-49: `GeminiReranker.__init__`

```python
class GeminiReranker:
    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = _MAX_RETRIES,
        rerank_weight: float = _RERANK_WEIGHT,
        rrf_weight: float = _RRF_WEIGHT,
    ) -> None:
        self._model_name = model_name or settings.gemini_model
```

**Constructor with configurable weights** — Allows tuning the rerank/RRF balance
without subclassing. In production you might A/B test different weight ratios.

#### Lines 55-120: `rerank`

```python
def rerank(self, query, candidates, top_k=5) -> list[RankedResult]:
    if not candidates or not query:
        return candidates[:top_k]

    if not settings.gemini_api_key:
        logger.warning("No API key configured, skipping reranking")
        return candidates[:top_k]
```

**Three levels of graceful degradation** at the top of the method:
1. Empty input → empty output
2. No API key → return candidates in original RRF order
3. (Later) API failure or parse failure → return candidates in original order

The system never crashes or raises an exception from reranking. It just returns
the best results it can.

```python
prompt = self._build_prompt(query, candidates)
try:
    response_text = self._call_gemini(prompt)
except RuntimeError:
    return candidates[:top_k]  # Graceful degradation
```

If Gemini is down after all retries, return results without reranking.

```python
for i, candidate in enumerate(candidates):
    normalised = self._normalize_score(scores[i])
    final = self._rerank_weight * normalised + self._rrf_weight * candidate.rrf_score
```

**Score combination** — The final score blends two signals:
- `normalised`: Gemini's 1-10 rating, mapped to 0.0-1.0
- `candidate.rrf_score`: The original RRF score from hybrid retrieval

Example: A candidate with Gemini score 8/10 and RRF score 0.015:
```
normalised = (8 - 1) / 9 = 0.778
final = 0.7 * 0.778 + 0.3 * 0.015 = 0.545 + 0.005 = 0.549
```

#### Lines 126-147: `_build_prompt`

```python
def _build_prompt(self, query, candidates):
    passages: list[str] = []
    for i, c in enumerate(candidates, 1):
        truncated = c.text[:_MAX_PASSAGE_CHARS]
        passages.append(f"Passage {i}: {truncated}")
```

**Single-call design** — All candidates are in one prompt. This is 10-20x
cheaper than one API call per candidate. The prompt asks for JSON output:
`{"scores": [7, 3, 9, ...]}`. Using `response_mime_type="application/json"`
in the Gemini config encourages the model to return valid JSON.

#### Lines 153-197: `_call_gemini`

```python
def _call_gemini(self, prompt: str) -> str:
    self._ensure_api_key()
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(self._model_name)
```

**Lazy import** — `google.generativeai` is imported inside the method. This
avoids loading the heavy protobuf-based library when the reranker is not used.

**Exponential backoff retry** — Waits 1s, 2s, 4s between retries. This handles
transient 429 (rate limit) and 503 (service unavailable) errors from Google's
API. The pattern is identical to `EmbeddingService._embed_with_retry()` in
Phase 2.

#### Lines 203-249: `_parse_scores`

```python
def _parse_scores(self, response_text, num_candidates):
    cleaned = self._strip_markdown_fences(response_text)
    data = json.loads(cleaned)

    if isinstance(data, dict) and "scores" in data:
        scores = data["scores"]
    elif isinstance(data, list):
        scores = data
```

**Three JSON formats handled:**
1. `{"scores": [7, 3, 9]}` — The expected format
2. `[7, 3, 9]` — Sometimes LLMs omit the wrapper object
3. `` ```json\n{"scores": [7, 3, 9]}\n``` `` — Markdown fences

```python
if len(scores) != num_candidates:
    return []
```

**Length validation** — If Gemini returns 3 scores for 5 candidates, the mapping
is ambiguous. Return empty and fall back to RRF ordering.

```python
val = max(1.0, min(10.0, val))
```

**Score clamping** — If Gemini returns 0 or 15, clamp to the valid 1-10 range.
This prevents normalization from producing negative values or values > 1.0.

#### Lines 260-263: `_normalize_score`

```python
@staticmethod
def _normalize_score(score: float) -> float:
    return max(0.0, min(1.0, (score - 1.0) / 9.0))
```

Maps 1-10 to 0.0-1.0: `score 1 → 0.0`, `score 5.5 → 0.5`, `score 10 → 1.0`.
The formula `(x - min) / (max - min)` is min-max normalization. The outer
`max(0.0, min(1.0, ...))` is a safety clamp.

---

### 7.4 `src/retrieval/query_processor.py`

This file orchestrates the entire retrieval pipeline.

#### Lines 32-38: `QueryType` enum

```python
class QueryType(str, Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"
    METADATA = "metadata"
```

**Inherits from `str`** — This makes `QueryType.FACTUAL == "factual"` true and
allows JSON serialization without custom encoders. It also means Gemini's
response (`{"type": "factual"}`) can be directly passed to `QueryType(value)`.

**Four query types** based on information retrieval taxonomy:
- **FACTUAL** — "What is the F1 score?" — Needs precise keyword + semantic match
- **CONCEPTUAL** — "How does attention work?" — Needs broad semantic coverage
- **COMPARISON** — "Compare BERT and GPT" — Needs passages about both entities
- **METADATA** — "Who are the authors?" — Does not need retrieval at all

#### Lines 42-47: `_QUERY_TYPE_PARAMS`

```python
_QUERY_TYPE_PARAMS: dict[QueryType, dict] = {
    QueryType.FACTUAL:     {"alpha": 0.5, "top_k": 5},
    QueryType.CONCEPTUAL:  {"alpha": 0.8, "top_k": 10},
    QueryType.COMPARISON:  {"alpha": 0.7, "top_k": 15},
    QueryType.METADATA:    {"alpha": 0.5, "top_k": 5},
}
```

**Why different alpha per query type?**

- **Factual (alpha=0.5)**: "What is the F1 score?" — The answer contains a
  specific number. BM25 excels at matching "F1" and "score" exactly. Equal
  weight to both systems.
- **Conceptual (alpha=0.8)**: "How does attention work?" — Requires
  understanding meaning, not just matching words. Dense retrieval dominates.
- **Comparison (alpha=0.7)**: "Compare BERT and GPT" — Needs both named entity
  matching (BM25 for "BERT", "GPT") and conceptual understanding. Slight dense
  preference.

**Why different top_k?**
- Factual queries need 1-2 precise answers → `top_k=5` is sufficient
- Conceptual queries need broader context → `top_k=10`
- Comparison queries need passages about multiple entities → `top_k=15`

#### Lines 50-58: `QueryResult` dataclass

```python
@dataclass
class QueryResult:
    query: str
    query_type: QueryType
    results: list[RankedResult] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    metadata_answer: str | None = None
```

This is the final output of the entire retrieval pipeline. It bundles together
the ranked results, the classified query type, and any expanded queries. The
`metadata_answer` field is only populated for METADATA queries that short-circuit
the retrieval pipeline.

#### Lines 87-134: `process` — the main pipeline

```python
def process(self, query, paper_id=None, section_type=None) -> QueryResult:
    # 1. Classify
    query_type = self.classify_query(query)
    params = _QUERY_TYPE_PARAMS[query_type]

    # 2. Short-circuit metadata queries
    if query_type == QueryType.METADATA:
        return self._handle_metadata_query(query, paper_id)

    # 3. Expand
    expanded = self.expand_query(query)

    # 4. Retrieve and merge
    candidates = self._retrieve_and_merge(
        queries=expanded, top_k=params["top_k"], alpha=params["alpha"], ...
    )

    # 5. Rerank
    reranked = self._reranker.rerank(query, candidates, top_k=params["top_k"])

    return QueryResult(query=query, query_type=query_type, results=reranked, ...)
```

Five clear steps. Each step can fail independently, and the pipeline continues
with degraded quality. If classification fails → FACTUAL. If expansion fails →
just the original query. If retrieval fails → empty results. If reranking fails
→ RRF-ordered results.

**Metadata short-circuit** — "Who are the authors?" does not need to search
through chunk text. The answer is in the document metadata extracted in Phase 1.
Skipping retrieval saves ~525ms of latency.

#### Lines 140-173: `classify_query`

```python
def classify_query(self, query: str) -> QueryType:
    if not settings.gemini_api_key:
        return QueryType.FACTUAL

    prompt = (
        "Classify the following search query into exactly one category.\n\n"
        "Categories:\n"
        "- factual: asking for a specific fact, number, definition...\n"
        "- conceptual: asking to understand a concept, method, theory...\n"
        "- comparison: comparing two or more methods, models...\n"
        "- metadata: asking about authors, publication date, title...\n\n"
        f"Query: {query}\n\n"
        'Return ONLY valid JSON: {{"type": "category_name"}}'
    )

    try:
        response_text = self._call_gemini(prompt)
        cleaned = self._strip_markdown_fences(response_text)
        data = json.loads(cleaned)
        type_str = data.get("type", "factual").lower().strip()
        return QueryType(type_str)
    except (json.JSONDecodeError, ValueError, RuntimeError):
        return QueryType.FACTUAL
```

The prompt gives clear category definitions with examples. The response is
expected to be simple JSON: `{"type": "factual"}`. If parsing fails for any
reason, FACTUAL is the safe default — it uses balanced alpha and moderate top_k,
which works reasonably well for any query type.

#### Lines 179-218: `expand_query`

```python
def expand_query(self, query: str) -> list[str]:
    # ... Gemini call ...
    expansions = [str(q) for q in queries if q][:_MAX_EXPANSIONS]
    return [query] + expansions
```

**Original query always first** — The original query is never removed. Expansions
are added after it. If Gemini generates nonsense, the original query still
retrieves reasonable results.

**`[:_MAX_EXPANSIONS]` cap** — Limits to 3 expansions. More queries means more
Gemini reranking time and more retrieval calls. Three is the sweet spot: enough
to cover vocabulary gaps, not so many that latency suffers.

#### Lines 224-326: `_retrieve_and_merge` and `_cross_query_rrf`

```python
def _retrieve_and_merge(self, queries, top_k, alpha, paper_id, section_type):
    if len(queries) == 1:
        return self._retriever.retrieve(queries[0], ...)  # No merge needed

    per_query_results: list[list[RankedResult]] = []
    for q in queries:
        results = self._retriever.retrieve(q, ...)
        per_query_results.append(results)

    return self._cross_query_rrf(per_query_results)
```

**Single-query optimization** — When there is only one query (expansion failed
or no API key), skip the entire merge step. This saves the overhead of building
rank maps and computing cross-query RRF.

```python
def _cross_query_rrf(self, per_query_results):
    for chunk_id, result in all_chunks.items():
        rrf_sum = 0.0
        for rank_map in rank_maps:
            rank = rank_map.get(chunk_id, missing_rank)
            rrf_sum += 1.0 / (k + rank)
        rrf_score = rrf_sum / num_queries
```

**Division by `num_queries`** — This normalizes the score so it does not scale
with the number of expanded queries. Without this, adding more expansions would
inflate all scores equally.

#### Lines 332-355: `_handle_metadata_query`

```python
def _handle_metadata_query(self, query, paper_id):
    if paper_id:
        answer = f"This is a metadata query about paper '{paper_id}'..."
    else:
        answer = "This is a metadata query. Please specify a paper ID..."

    return QueryResult(
        query=query, query_type=QueryType.METADATA,
        results=[], metadata_answer=answer,
    )
```

Returns an empty result list with a `metadata_answer` string. Phase 4's QA
system will check for `metadata_answer` and respond directly from the document
metadata store instead of searching through chunks.

#### Lines 361-403: `_call_gemini`

Identical pattern to `GeminiReranker._call_gemini()`: lazy import, configure
API key, exponential backoff. The duplication is deliberate — each module is
self-contained and independently testable. This matches the project convention
established by `EmbeddingService` and `MetadataExtractor` in earlier phases.

---

### 7.5 `tests/unit/test_bm25_index.py`

21 tests organized into four groups:

#### Tokenization tests (6 tests)

```python
def test_tokenize_lowercases(self):
    idx = BM25Index(remove_stopwords=False)
    tokens = idx._tokenize("Hello World DEEP")
    assert tokens == ["hello", "world", "deep"]
```

Tests that the tokenizer lowercases, removes punctuation, handles stopword
removal vs retention, and handles edge cases (empty string, all-stopwords).

**Why `remove_stopwords=False` in some tests?** To isolate what is being tested.
When testing lowercase behavior, we do not want stopword removal to interfere.

#### Index building tests (4 tests)

Tests that `build_index()` updates `chunk_count()`, sets `is_built()`, raises
on empty input, and replaces the previous index on rebuild.

#### Search tests (8 tests)

```python
def test_search_returns_search_results(self):
    idx = BM25Index()
    idx.build_index([
        _make_chunk("c1", "deep learning for NLP"),
        _make_chunk("c2", "quantum computing physics"),
        _make_chunk("c3", "biology and chemistry research"),
    ])
    results = idx.search("deep learning")
```

**Why three chunks when testing for one match?** BM25Okapi's IDF formula is
`log((N - n + 0.5) / (n + 0.5))`. If a term appears in all documents (n=N),
IDF = `log(0.5 / (N+0.5))` which is negative, and rank-bm25 sets it to 0.
Adding unrelated "distractor" chunks ensures query terms appear in fewer than
half the documents, producing positive IDF values. This was a bug fix during
development — an important lesson about BM25 behavior.

#### Metadata filter tests (3 tests)

Tests that `paper_id`, `section_type`, and combined filters correctly restrict
results.

---

### 7.6 `tests/unit/test_hybrid_retriever.py`

18 tests organized into five groups:

#### RankedResult tests (2 tests)

Verify default score values and field storage.

#### Fusion tests (7 tests)

```python
def test_rrf_formula_correct(self):
    self.mock_vs.search.return_value = [_sr("c1")]  # rank 1
    self.mock_bm25.search.return_value = [_sr("c1")]  # rank 1

    results = self.retriever.retrieve("test", alpha=0.7)

    k = _RRF_K
    expected = 0.7 * (1 / (k + 1)) + 0.3 * (1 / (k + 1))
    assert abs(results[0].rrf_score - round(expected, 6)) < 1e-5
```

**Manual formula verification** — The test computes the expected RRF score by
hand and compares it to the actual output. This is the gold standard for testing
mathematical algorithms — if the formula changes accidentally, this test fails.

Similar tests verify the penalty rank (1000) for chunks missing from one source,
deduplication, sorting, and `top_k` limiting.

#### Alpha override tests (3 tests)

Tests `alpha=1.0` (pure dense), `alpha=0.0` (pure sparse), and `alpha=None`
(uses default).

#### Graceful degradation tests (3 tests)

Tests that embedding failure falls back to sparse, BM25-not-built falls back to
dense, and both-empty returns empty.

#### Filter passthrough tests (2 tests)

Verifies that `paper_id` and `section_type` are forwarded to both dense and
sparse search calls.

---

### 7.7 `tests/unit/test_reranker.py`

19 tests organized into five groups:

#### Score parsing tests (6 tests)

```python
def test_parse_scores_valid_json(self):
    reranker = GeminiReranker()
    scores = reranker._parse_scores('{"scores": [7, 3, 9]}', 3)
    assert scores == [7.0, 3.0, 9.0]
```

Tests all three JSON formats (dict, bare list, markdown-fenced), invalid JSON,
wrong score count, and out-of-range clamping.

#### Score normalization tests (3 tests)

```python
def test_normalize_score_5_point_5(self):
    assert abs(GeminiReranker._normalize_score(5.5) - 0.5) < 1e-6
```

Verifies the boundary cases: 1→0.0, 10→1.0, and midpoint 5.5→0.5.

#### Reranking tests (4 tests)

```python
def test_rerank_combines_with_rrf_score(self, mock_settings):
    reranker = GeminiReranker(rerank_weight=0.7, rrf_weight=0.3)
    candidates = [_ranked("c1", rrf_score=0.5)]

    with patch.object(reranker, "_call_gemini", return_value='{"scores": [10]}'):
        results = reranker.rerank("test", candidates, top_k=5)

    # rerank_score for 10 → normalised to 1.0
    # final = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
    assert abs(results[0].final_score - 0.85) < 0.01
```

**Manual score combination verification** — Computes the expected final score
by hand. This ensures the weight combination formula is correct.

#### Graceful degradation tests (4 tests)

Tests: no API key → original order, API failure → original order, parse
failure → original order, empty candidates → empty result.

#### Markdown fence tests (2 tests)

Tests that `` ```json\n...\n``` `` is stripped correctly and plain text passes
through unchanged.

---

### 7.8 `tests/unit/test_query_processor.py`

23 tests organized into six groups:

#### QueryType tests (2 tests)

Verify enum values and construction from strings.

#### Classification tests (6 tests)

Each test mocks `_call_gemini` to return a specific type and verifies
classification. Failure and no-API-key cases default to FACTUAL.

```python
@patch("src.retrieval.query_processor.settings")
def test_classify_factual(self, mock_settings):
    mock_settings.gemini_api_key = "fake-key"
    mock_settings.gemini_model = "gemini-2.0-flash"

    with patch.object(self.processor, "_call_gemini", return_value='{"type": "factual"}'):
        result = self.processor.classify_query("What is the F1 score?")
    assert result == QueryType.FACTUAL
```

**Why `@patch("src.retrieval.query_processor.settings")`?** The `settings`
singleton reads from environment variables. In tests, we mock it to provide
fake API keys and model names. This avoids requiring real API credentials in CI.

#### Expansion tests (4 tests)

Tests that expansion includes the original query, adds reformulations, and
falls back to `[original]` on failure.

#### Pipeline tests (7 tests)

Tests the full `process()` flow: correct return type, per-type parameter
selection, metadata short-circuit, reranker invocation, and filter passthrough.

```python
def test_process_factual_sets_correct_params(self, mock_settings):
    with patch.object(self.processor, "classify_query", return_value=QueryType.FACTUAL):
        with patch.object(self.processor, "expand_query", return_value=["test"]):
            self.processor.process("test")

    call_kwargs = self.mock_retriever.retrieve.call_args
    assert call_kwargs.kwargs.get("alpha") == 0.5
    assert call_kwargs.kwargs.get("top_k") == 5
```

**Verifying parameter passthrough** — Does not test retrieval results (those
are mocked). Tests that the correct alpha and top_k values reach the retriever
based on the classified query type.

#### Cross-query merge tests (3 tests)

Tests deduplication (same chunk from multiple queries appears once),
score boosting (chunks in multiple queries rank higher), and single-query
bypass (no merge needed).

---

### 7.9 `tests/conftest.py` (Phase 3 additions)

Two new fixtures added:

```python
@pytest.fixture()
def sample_search_results():
    return [
        SearchResult(chunk_id=f"chunk-{i}", text=f"Sample text...", score=1.0 - (i * 0.1), ...)
        for i in range(5)
    ]

@pytest.fixture()
def sample_ranked_results():
    return [
        RankedResult(chunk_id=f"chunk-{i}", ..., rrf_score=1.0 / (60 + i + 1))
        for i in range(5)
    ]
```

These provide pre-built test data for integration tests that span multiple
Phase 3 components. The `sample_search_results` fixture uses realistic
decreasing scores. The `sample_ranked_results` fixture uses actual RRF scores
calculated with k=60.

---

## 8. Phase 2 to Phase 3 to Phase 4 Connections

### Phase 2 → Phase 3 (Input)

Phase 2 produces two key artifacts that Phase 3 consumes:

1. **`EnrichedChunk` objects** — The `SectionAwareChunker` (Phase 2) produces
   chunks with `text`, `chunk_id`, `paper_id`, `section_type`, and metadata.
   Phase 3's `BM25Index.build_index()` takes these same chunks and builds the
   sparse keyword index.

2. **`VectorStore` with embedded chunks** — Phase 2's `EmbeddingService`
   generates Gemini embeddings and stores them in ChromaDB via `VectorStore`.
   Phase 3's `HybridRetriever` calls `VectorStore.search()` for dense retrieval.

3. **`EmbeddingService`** — Phase 3 reuses the same embedding service to embed
   queries (with `task_type="retrieval_query"`) before searching the vector
   store. The asymmetric task types (document vs query) are a Gemini embedding
   feature that improves retrieval quality.

```
Phase 2 output:                    Phase 3 input:
EnrichedChunk[] ──────────────────► BM25Index.build_index()
VectorStore (ChromaDB) ───────────► HybridRetriever._get_dense_results()
EmbeddingService ─────────────────► HybridRetriever._get_dense_results()
```

### Phase 3 → Phase 4 (Output)

Phase 3 produces `QueryResult` objects containing:

1. **`results: list[RankedResult]`** — The ranked passages most relevant to the
   query, ordered by `final_score`. Phase 4's QA system will use these passages
   as context for generating answers.

2. **`query_type: QueryType`** — Tells Phase 4 how to format the answer.
   Factual queries get concise answers. Conceptual queries get detailed
   explanations. Comparison queries get structured comparisons.

3. **`metadata_answer: str`** — For metadata queries, Phase 4 can skip RAG
   generation entirely and return the metadata directly.

```
Phase 3 output:                    Phase 4 input:
QueryResult.results ──────────────► QA context passages
QueryResult.query_type ───────────► Answer formatting strategy
QueryResult.metadata_answer ──────► Direct metadata response
```

---

## 9. Performance Characteristics

### Latency Breakdown

```
Component                     Latency      Notes
────────────────────────────  ──────────   ──────────────────────────────
Query classification          ~300ms       Single Gemini API call
Query expansion               ~300ms       Single Gemini API call
BM25 search                   ~2-5ms       In-memory, numpy vectorized
Query embedding               ~100ms       Gemini embedding API call
Vector search (ChromaDB)      ~10-20ms     HNSW approximate NN search
RRF computation               ~0.1ms       Pure Python dict operations
Gemini reranking              ~500-800ms   Single Gemini API call (20 passages)
────────────────────────────  ──────────
Total pipeline                ~1.2-1.5s    Dominated by Gemini API calls
```

### Latency optimization opportunities

1. **Parallel dense + sparse** — BM25 search and vector search are independent.
   Running them concurrently with `asyncio.gather()` would save ~5-20ms.

2. **Classification + expansion in one call** — Instead of two Gemini calls,
   send a single prompt that returns both classification and expansions. This
   saves ~300ms.

3. **Reranker batching** — When processing multiple queries, batch all reranking
   requests into a single Gemini call. Saves one API round-trip per additional
   query.

4. **Caching** — Cache embeddings, BM25 scores, and reranker scores for repeated
   queries. The BM25 index is already in-memory (zero cache miss).

5. **Streaming** — Return Stage 1 results immediately while Stage 2 (reranking)
   runs in the background. The UI shows initial results that improve when
   reranking completes.

### Memory characteristics

```
Component           Memory Usage    Notes
──────────────────  ─────────────   ──────────────────────────
BM25 index          ~2-5 MB         Term frequencies + IDF for 1000 chunks
ChromaDB (HNSW)     ~50-100 MB      768-dim vectors for 1000 chunks
Stopwords set       ~4 KB           80 strings in a frozenset
Punct table         ~1 KB           256-entry translation table
```

---

## 10. Cross-Cutting Design Decisions

### Dependency injection over global state

Every Phase 3 class accepts its dependencies through the constructor:
- `HybridRetriever(vector_store, embedding_service, bm25_index)`
- `QueryProcessor(hybrid_retriever, reranker)`
- `GeminiReranker(model_name, rerank_weight, rrf_weight)`

This makes unit testing trivial (inject mocks) and production wiring explicit.
The alternative — global singletons or module-level state — would make tests
interdependent and non-deterministic.

### SearchResult as the common interface

Both `VectorStore.search()` and `BM25Index.search()` return `list[SearchResult]`.
This means `HybridRetriever` does not need to know which system produced the
results. It just works with a uniform interface. If you add a third retrieval
source (e.g., ColBERT), it just needs to return `SearchResult` objects.

### Graceful degradation everywhere

Every single external dependency has a fallback:

| Component | Failure | Fallback |
|-----------|---------|----------|
| Dense retrieval | Embedding API down | Sparse-only results |
| Sparse retrieval | BM25 not built | Dense-only results |
| Both retrievers | Both fail | Empty results |
| Query classification | Gemini failure | Default to FACTUAL |
| Query expansion | Gemini failure | Original query only |
| Reranking | Gemini failure | Return RRF-ordered results |
| Score parsing | Invalid JSON | Return RRF-ordered results |

This means the system degrades from "excellent" to "good" to "acceptable" but
never crashes. Users get results (possibly less accurate) even when subsystems
fail.

### Duplicate `_call_gemini` methods

Both `GeminiReranker` and `QueryProcessor` have their own `_call_gemini` method.
This looks like a DRY violation, but is deliberate. Each module is
self-contained and independently deployable. If the reranker needs a different
retry strategy or model configuration, it can diverge without affecting query
processing. This follows the codebase convention established by Phase 1's
`MetadataExtractor` and Phase 2's `EmbeddingService`.

---

## 11. Interview Questions and Answers

### Q: "Why not just use vector search?"

**Answer:** Vector search excels at semantic understanding but has three critical
blind spots. First, exact technical terms: search for "BM25Okapi" and the
embedding model has likely never seen this term in training. It maps to a generic
point in embedding space. BM25 matches it character-for-character. Second, author
names and identifiers: "Smith et al. 2023" looks like generic academic text to an
embedding model, but BM25 finds the exact string in the references section.
Third, numbers and metrics: "94.2% F1 on DocBank" — the embedding collapses the
specific number. BM25 matches it exactly.

In our system, we use hybrid retrieval with Reciprocal Rank Fusion. The alpha
parameter controls the blend — factual queries get `alpha=0.5` (equal weight)
while conceptual queries get `alpha=0.8` (dense-heavy). This query-type-aware
alpha tuning is one of our key design decisions: we classify the query first,
then set retrieval parameters accordingly.

The empirical evidence is strong. The BEIR benchmark suite shows that hybrid
retrieval outperforms dense-only retrieval on 11 of 15 datasets, with average
nDCG@10 improvements of 3-8%.

### Q: "Explain RRF and why k=60"

**Answer:** Reciprocal Rank Fusion is a method for combining two ranked lists
from different retrieval systems. The core insight is that you cannot average
scores directly because the score scales are incomparable — cosine similarity is
0 to 1, BM25 scores are unbounded positive numbers. RRF sidesteps this by
converting scores to ranks and combining ranks.

The formula is: `RRF(d) = alpha * 1/(k + rank_dense(d)) + (1-alpha) * 1/(k + rank_sparse(d))`.

The `k` parameter is a smoothing constant that controls how much the top ranks
dominate. With `k=60`, the contribution of rank 1 is `1/61 = 0.0164` and rank 2
is `1/62 = 0.0161`. The difference is tiny — only 2%. This is desirable because
individual rankers have noise; being rank 1 vs rank 2 in BM25 often depends on
document length normalization quirks, not true relevance.

If `k` were small, say `k=1`, rank 1 gets `1/2 = 0.5` and rank 2 gets
`1/3 = 0.33` — a 34% penalty for one position difference. That magnifies noise.

The value `k=60` comes from the original Cormack, Clarke, and Butt paper (2009)
where it was found empirically optimal across TREC benchmark collections. It has
since been validated across hundreds of IR benchmarks and is the de facto
standard in the field.

For documents missing from one ranker's list, we assign `rank=1000`, which gives
a contribution of `1/1060 ≈ 0.0009` — essentially a 17x penalty compared to
rank 1. This means a document must be very highly ranked in the other list to
overcome the missing-rank penalty.

### Q: "How do you handle queries that need exact keyword matching?"

**Answer:** Our system handles this at two levels. First, structurally: every
query goes through both BM25 (exact keyword matching) and vector search
(semantic matching). The BM25 component uses the BM25Okapi algorithm from
rank-bm25, which computes term frequency, inverse document frequency, and
document length normalization. A query like "BERT-base model" will get exact
matches from BM25 for the tokens "bert-base" and "model".

Second, dynamically: the query classifier adjusts the alpha parameter based on
query type. Factual queries like "What is the F1 score?" get `alpha=0.5`, giving
BM25 equal weight with vector search. This is because factual queries often
contain specific terms (numbers, model names, metric names) that BM25 matches
better than semantic search.

The BM25 tokenizer is deliberately simple: lowercase, remove punctuation, split
on whitespace, remove stopwords. We do not use stemming because stemming can
hurt exact matching — "BERT" might be stemmed to something unexpected. The
stopword list is a built-in frozenset of ~80 common English words, avoiding any
dependency on NLTK or spaCy.

Metadata filters are applied post-scoring, not pre-scoring. This preserves
correct IDF computation — if you filtered the corpus before computing BM25
scores, the IDF values would change because they depend on corpus statistics.

### Q: "What happens if the re-ranker is slow? How would you optimize?"

**Answer:** The Gemini reranker is the latency bottleneck at ~500-800ms out of a
~1.2-1.5s total pipeline. Here are five concrete optimizations I would consider:

**1. Streaming results.** Return Stage 1 (BM25 + vector + RRF) results
immediately to the UI while the reranker runs in the background. When reranking
completes, update the UI. The user sees results in ~25ms and gets improved
ordering ~500ms later. This is how Google Instant works.

**2. Smaller reranking model.** Use `gemini-2.0-flash-lite` instead of
`gemini-2.0-flash` for reranking. Flash-lite has 2-3x lower latency with
slightly lower quality. For reranking (which is simpler than generation), the
quality difference is often negligible.

**3. Reduce candidate count.** Our current pipeline sends 20 candidates to the
reranker. Reducing to 10 roughly halves the prompt size and reduces latency by
~30%. The trade-off is lower recall, but for factual queries where only 1-2
passages matter, this is acceptable.

**4. Batch classification and expansion.** Currently, query classification and
expansion are two separate Gemini API calls (~300ms each). Combining them into
a single prompt saves one round-trip, cutting ~300ms from the pipeline.

**5. Cross-encoder distillation.** Train a small cross-encoder model (like
TinyBERT or MiniLM) on Gemini's reranking judgments. This gives you a local
reranker that runs in ~5ms on CPU. You lose some accuracy but eliminate the API
dependency entirely. In production, you can use the local model for 95% of
queries and fall back to Gemini for high-stakes queries.

**6. Caching.** Cache reranker scores keyed on `(query_hash, chunk_id)`. Academic
papers do not change, so the same query on the same paper always gets the same
relevance score. This turns repeated queries from ~500ms to ~0ms.

Our current design already supports all of these: the constructor-injected
dependencies mean you can swap the reranker implementation without changing any
other code, and the graceful degradation means skipping reranking entirely still
produces useful (RRF-ordered) results.
