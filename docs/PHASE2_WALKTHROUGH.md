# Phase 2 Walkthrough: Intelligent Chunking & Embedding

This document explains every file, every class, every method, and every design
decision in Phase 2. It covers *why* section-aware chunking outperforms naive
splitting, how the chunk optimizer works, how embeddings are generated and
stored, and how Phase 2 connects to Phase 1 (input) and Phase 3 (output).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 2 Pipeline Diagram](#2-phase-2-pipeline-diagram)
3. [Why Section-Aware Chunking Beats Naive Fixed-Size Chunking](#3-why-section-aware-chunking-beats-naive-fixed-size-chunking)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
   - [src/models/schemas.py (Phase 2 additions)](#41-srcmodelsschemaspy-phase-2-additions)
   - [src/chunking/chunker.py](#42-srcchunkingchunkerpy)
   - [src/chunking/chunk_optimizer.py](#43-srcchunkingchunk_optimizerpy)
   - [src/retrieval/embedding_service.py](#44-srcretrievalembedding_servicepy)
   - [src/retrieval/vector_store.py](#45-srcretrievalvector_storepy)
   - [tests/conftest.py (Phase 2 fixtures)](#46-testsconftestpy-phase-2-fixtures)
   - [tests/unit/test_chunker.py](#47-testsunittest_chunkerpy)
   - [tests/unit/test_embedding_service.py](#48-testsunittest_embedding_servicepy)
   - [tests/unit/test_vector_store.py](#49-testsunittest_vector_storepy)
5. [How Phase 2 Connects to Phase 1 (Input) and Phase 3 (Output)](#5-how-phase-2-connects-to-phase-1-input-and-phase-3-output)
6. [Cross-Cutting Design Decisions](#6-cross-cutting-design-decisions)
7. [Edge Cases and Error Handling](#7-edge-cases-and-error-handling)
8. [Interview Questions and Answers](#8-interview-questions-and-answers)

---

## 1. Architecture Overview

Phase 2 is the **chunking, embedding, and storage layer**. It takes the
structured `PaperStructure` produced by Phase 1 and transforms it into
searchable vector embeddings stored in ChromaDB.

```
Layer           Responsibility
─────────────   ──────────────────────────────────────────────────
Chunking        Split parsed papers into enriched, section-respecting chunks
Optimization    Grid search over chunk-size/overlap to maximize retrieval quality
Embedding       Convert text chunks into dense vectors via Gemini text-embedding-004
Storage         Persist embeddings in ChromaDB with metadata-filtered cosine search
Schemas         EnrichedChunk and PaperStructure bridge Phase 1 → Phase 2
```

Key architectural principles:

- **Structure-preserving chunking** — Chunks never cross section boundaries.
  A methodology chunk never contains introduction text. This means every chunk
  has a single, unambiguous `section_type`, which enables precise metadata
  filtering at retrieval time.
- **Content-type-specific rules** — Tables, equations, and references each get
  dedicated handling. A table is always one chunk. An equation is never split
  from its context. References are batched 5-10 per chunk. These rules exist
  because each content type has unique retrieval semantics.
- **Offline-first optimization** — The chunk optimizer uses term-overlap scoring
  (no API calls) to simulate retrieval quality. This makes grid search fast,
  free, and reproducible.
- **Separation of embedding and storage** — The `EmbeddingService` produces
  vectors. The `VectorStore` persists and searches them. Neither knows about
  the other. This means you can swap Gemini for OpenAI embeddings or ChromaDB
  for Pinecone without touching the other component.

---

## 2. Phase 2 Pipeline Diagram

```
                   Phase 1 Output
                        │
                        ▼
              ┌───────────────────┐
              │  PaperStructure   │
              │                   │
              │  .sections[]      │  DetectedSection objects from LayoutAnalyzer
              │  .tables[]        │  ExtractedTable objects from TableExtractor
              │  .references[]    │  Raw reference strings
              │  .metadata        │  DocumentMetadataSchema from MetadataExtractor
              └────────┬──────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
         ▼             ▼              ▼
   ┌───────────┐ ┌───────────┐ ┌───────────┐
   │ Section   │ │ Table     │ │ Reference │
   │ Chunking  │ │ Chunking  │ │ Chunking  │
   │           │ │           │ │           │
   │ _recursive│ │ Render as │ │ Batch     │
   │ _split()  │ │ pipe-     │ │ 5-10 per  │
   │ with      │ │ delimited │ │ chunk     │
   │ equation  │ │ text +    │ │           │
   │ protection│ │ caption   │ │           │
   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         │             │              │
         └─────────────┼──────────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ list[EnrichedChunk│
              │ ]                 │
              │                   │
              │ .chunk_id         │  Deterministic SHA-256 hash
              │ .text             │  The chunk's text content
              │ .section_type     │  From the source section
              │ .page_numbers     │  From the source section
              │ .paper_id         │  From the paper
              │ .token_count      │  len(text) // 4
              │ .chunk_index      │  0, 1, 2, ...
              │ .total_chunks     │  Total across paper
              └────────┬──────────┘
                       │
          ┌────────────┼──────────────────┐
          │            │                  │
          │   (Optional: ChunkOptimizer)  │
          │   Grid search over            │
          │   [256,512,768,1024] ×        │
          │   [0,25,50,100]               │
          │   Measures Precision@5        │
          │                               │
          └────────────┼──────────────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ EmbeddingService  │
              │                   │
              │ embed_chunks()    │  task_type="retrieval_document"
              │ embed_query()     │  task_type="retrieval_query"
              │                   │
              │ Batches of 100    │
              │ 0.5s delay        │
              │ Retry with        │
              │ exponential       │
              │ backoff           │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ VectorStore       │
              │ (ChromaDB)        │
              │                   │
              │ store()           │  Upsert chunks + embeddings
              │ search()          │  Cosine similarity + metadata filters
              │ delete_paper()    │  Remove all chunks for a paper
              │                   │
              │ Filters:          │
              │   paper_id        │
              │   section_type    │
              │   combined ($and) │
              └────────┬──────────┘
                       │
                       ▼
                 Phase 3 Input
              (RAG retrieval, Q&A)
```

---

## 3. Why Section-Aware Chunking Beats Naive Fixed-Size Chunking

This is one of the most important design decisions in the entire system and
a frequently asked interview question. Here is the full argument.

### 3.1 What naive chunking does

Naive chunking treats the document as a flat string. It splits every N
characters (or tokens), optionally with overlap. LangChain's
`RecursiveCharacterTextSplitter` is the canonical example.

```
┌──────────────────────────────────────────────────────────────┐
│ ...end of Abstract. 1. Introduction  Document understanding  │
│ is a critical task...                                        │  ← Chunk 3
└──────────────────────────────────────────────────────────────┘
```

The problem: **Chunk 3 mixes Abstract and Introduction text.** It has no
single section label. A metadata filter for `section_type=introduction`
would either miss this chunk (losing relevant text) or include it (adding
irrelevant abstract text as noise).

### 3.2 What section-aware chunking does

Section-aware chunking splits *within* sections, never *across* them:

```
┌──────────────────────────────────────────┐
│ ...novel approach to document            │
│ understanding using deep neural          │
│ networks. We demonstrate state-of-the-   │
│ art results...                           │  ← Abstract chunk (complete)
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Document understanding is a critical     │
│ task in information extraction. Prior    │
│ work has focused on rule-based...        │  ← Introduction chunk 1
└──────────────────────────────────────────┘
```

Every chunk belongs to exactly one section. Metadata is unambiguous.

### 3.3 Five concrete advantages

| # | Advantage | Why it matters |
|---|-----------|----------------|
| 1 | **Clean metadata** | Every chunk has a single `section_type`. Filters work perfectly — "show me only methodology chunks" returns exactly what the user expects. |
| 2 | **Better retrieval precision** | When a user asks "what loss function did they use?", the query should match methodology chunks. If a chunk mixes methodology + results, the retrieval model sees diluted signal. |
| 3 | **Explainable citations** | In a RAG system, you need to tell the user *where* an answer came from. "Page 3, Methodology section" is meaningful. "Characters 2048-4096" is not. |
| 4 | **Table and equation integrity** | Naive chunking can split `$$L = L_{cls} + \lambda L_{layout}$$` across two chunks, destroying it. Section-aware chunking uses equation-protection regex to keep equations whole. |
| 5 | **Reference deduplication** | Naive chunking might put half a reference in one chunk and half in the next. Section-aware chunking batches 5-10 complete references per chunk. |

### 3.4 The cost

Section-aware chunking produces **variable-size chunks**. A short abstract
might be 80 tokens. A long methodology section might produce five 512-token
chunks. This is fine for cosine similarity search (embedding models handle
variable lengths), but it means you cannot assume a fixed token budget per
chunk when building LLM prompts. The `token_count` field on each
`EnrichedChunk` exists precisely for this reason — downstream code sums
`token_count` to stay within the LLM context window.

### 3.5 When naive chunking is acceptable

Naive chunking works when:
- The document has no meaningful structure (e.g., a legal contract with no
  headings).
- You need speed over quality (e.g., indexing millions of web pages).
- The retrieval pipeline does not use metadata filters.

For research papers — which are *highly* structured — section-aware chunking
is strictly better.

---

## 4. File-by-File Deep Dive

---

### 4.1 `src/models/schemas.py` (Phase 2 additions)

Phase 2 adds two new schemas at the bottom of the existing file. These
are the data contracts that connect Phase 1 output to Phase 2 processing.

#### `EnrichedChunk` (lines 166-179)

```python
class EnrichedChunk(BaseModel):
    chunk_id: str                                    # Deterministic SHA-256 hash
    text: str                                        # The actual chunk text
    token_count: int = Field(ge=0)                   # Approximate token count
    section_type: SectionType                        # From the source section
    section_title: str | None = None                 # e.g. "2. Methodology"
    page_numbers: list[int] = Field(default_factory=list)
    paper_id: str                                    # Links chunk to source paper
    paper_title: str | None = None                   # For display purposes
    chunk_index: int = Field(ge=0)                   # 0-based position in paper
    total_chunks: int = Field(ge=0)                  # Total chunks for this paper
    metadata: dict = Field(default_factory=dict)     # Extensible key-value bag
```

**Why every field exists:**

- `chunk_id` — Deterministic (SHA-256 of `paper_id:text`), not random UUID.
  This makes re-indexing idempotent: chunking the same paper twice produces
  the same IDs, so ChromaDB upsert replaces rather than duplicates.
- `token_count` — Approximate (`len(text) // 4`). Used by RAG prompt-builders
  to pack chunks into the LLM context window without exceeding token limits.
  Using a rough heuristic (4 chars per token for English) avoids requiring
  a tokenizer dependency.
- `section_type` — A `SectionType` enum (not a free-form string). This
  enables type-safe metadata filtering in the vector store.
- `page_numbers` — List, not a single int, because a section can span pages.
  Used for source citations in RAG answers.
- `paper_id` + `paper_title` — Denormalized from the paper level onto each
  chunk. This avoids joins at retrieval time: when ChromaDB returns a chunk,
  you immediately know which paper it belongs to without a second lookup.
- `chunk_index` + `total_chunks` — Enables ordered reconstruction ("show me
  all chunks from this paper in order") and progress tracking.
- `metadata` — An open `dict` for type-specific data (e.g., `{"type": "table",
  "table_index": 0}` for table chunks). Avoids schema bloat for fields that
  only apply to certain chunk types.

#### `PaperStructure` (lines 182-193)

```python
class PaperStructure(BaseModel):
    paper_id: str
    sections: list[DetectedSection] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    metadata: DocumentMetadataSchema = Field(default_factory=DocumentMetadataSchema)
```

**Why this exists:** Phase 1 produces four separate outputs (pages, sections,
tables, metadata) from four independent processors. `PaperStructure` bundles
them into a single object that serves as the **contract** between Phase 1 and
Phase 2. Without it, the chunker would need four separate parameters, and
any change to Phase 1 outputs would require changing every chunker call site.

**Why `references` is a separate field:** References appear both inside the
`REFERENCES` section (as raw text) and as a parsed `list[str]` from the
`LayoutAnalyzer`. The chunker uses the parsed list for intelligent batching
(5-10 per chunk). If no parsed references exist, it falls back to chunking
the raw section text.

**Why `metadata` has a default factory:** A paper might not have metadata
(e.g., if heuristic extraction failed and no LLM was available). The
`default_factory=DocumentMetadataSchema` creates an empty metadata object
with all `None` fields, avoiding `NoneType` errors downstream.

---

### 4.2 `src/chunking/chunker.py`

This is the heart of Phase 2 — 319 lines that implement all five chunking
rules (section boundaries, tables, equations, references, overlap).

#### Module-Level Constants (lines 32-43)

```python
_DEFAULT_TARGET_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 50
_REFS_PER_CHUNK_MIN = 5
_REFS_PER_CHUNK_MAX = 10
_CHARS_PER_TOKEN = 4
_DISPLAY_EQUATION_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_EQUATION_RE = re.compile(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)")
```

**Why 512 tokens:** This is the sweet spot for embedding models. Smaller
chunks have more precise semantics but lose context. Larger chunks have
richer context but diluted signal for similarity search. 512 tokens is the
default in most production RAG systems (LlamaIndex, LangChain).

**Why 50 tokens overlap:** Overlap prevents information loss at chunk
boundaries. If a sentence straddles two chunks, the overlap ensures it
appears in both. 50 tokens (~200 characters, ~2-3 sentences) is enough to
preserve sentence-level context without excessive duplication.

**Why 4 chars per token:** This is a well-known approximation for English
text. OpenAI's tokenizer averages 4.0 characters per token for English
prose. Using a constant avoids depending on a tokenizer library (tiktoken,
sentencepiece) just for sizing decisions.

**The equation regexes:**

- `_DISPLAY_EQUATION_RE`: Matches `$$...$$` (display math). The `re.DOTALL`
  flag makes `.` match newlines, because multi-line equations like
  `$$\sum_{i=1}^{n}\n  x_i$$` are common.
- `_INLINE_EQUATION_RE`: Matches `$...$` (inline math). Uses negative
  lookbehind `(?<!\$)` and negative lookahead `(?!\$)` to avoid matching
  display equation delimiters (`$$`).

#### `class SectionAwareChunker` (lines 46-319)

##### `__init__` (lines 57-67)

```python
def __init__(
    self,
    target_tokens: int = _DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
    refs_per_chunk_min: int = _REFS_PER_CHUNK_MIN,
    refs_per_chunk_max: int = _REFS_PER_CHUNK_MAX,
) -> None:
```

All parameters have sensible defaults but are configurable. The chunk
optimizer exercises this by creating chunkers with different sizes/overlaps
during grid search.

##### `chunk()` — The main entry point (lines 73-105)

```python
def chunk(self, paper: PaperStructure) -> list[EnrichedChunk]:
    raw_chunks: list[EnrichedChunk] = []

    # 1. Chunk each section
    for section in paper.sections:
        if section.section_type == SectionType.REFERENCES:
            raw_chunks.extend(
                self._chunk_references(paper.references, section, paper)
            )
        else:
            raw_chunks.extend(self._chunk_section(section, paper))

    # 2. Add tables as standalone chunks
    for table in paper.tables:
        raw_chunks.append(self._table_to_chunk(table, paper))

    # 3. Assign final indices and totals
    total = len(raw_chunks)
    result: list[EnrichedChunk] = []
    for idx, chunk in enumerate(raw_chunks):
        chunk.chunk_index = idx
        chunk.total_chunks = total
        result.append(chunk)

    return result
```

**Three-phase process:**

1. **Section dispatch (lines 85-91):** Iterates over sections in document
   order. References get special handling (`_chunk_references`); all other
   sections go through `_chunk_section`. This is a **strategy pattern** —
   the section type determines which chunking strategy to use.

2. **Table injection (lines 94-95):** Tables are added *after* sections.
   Each table becomes exactly one chunk. Tables are not interleaved with
   section chunks because a table's position in the document is tracked via
   `page_number`, not document order.

3. **Index assignment (lines 98-103):** All chunks get sequential indices
   `0, 1, 2, ...` and the total chunk count. This must happen *after* all
   chunks are collected because `total_chunks` depends on the final count.
   The `_enrich()` method sets placeholder values (`chunk_index=0`,
   `total_chunks=0`) that get overwritten here.

**Why this order matters:** Sections are processed in `order_index` order
(the order they appear in the document). This means chunk indices follow
the natural reading order of the paper: title → abstract → introduction →
methodology → results → conclusion → references → tables.

##### `_chunk_section()` — Section splitting (lines 111-141)

```python
def _chunk_section(self, section: DetectedSection, paper: PaperStructure) -> list[EnrichedChunk]:
    text = section.text.strip()
    if not text:
        return []

    target_chars = self.target_tokens * _CHARS_PER_TOKEN
    overlap_chars = self.overlap_tokens * _CHARS_PER_TOKEN

    fragments = self._recursive_split(text, target_chars)

    chunks: list[EnrichedChunk] = []
    for i, fragment in enumerate(fragments):
        if i > 0 and overlap_chars > 0:
            prev_tail = fragments[i - 1][-overlap_chars:]
            fragment = prev_tail + " " + fragment

        chunks.append(self._enrich(text=fragment.strip(), section=section, paper=paper))

    return chunks
```

**Line-by-line:**

- **Line 117:** Empty sections produce no chunks (e.g., a heading with no body).
- **Lines 121-122:** Convert token counts to character counts using the 4
  chars/token approximation. All splitting happens in character space.
- **Line 124:** Delegate the actual text splitting to `_recursive_split()`.
- **Lines 129-131:** Apply overlap by prepending the tail of the previous
  fragment. This is a **trailing overlap** strategy: chunk N+1 starts with
  the last `overlap_chars` characters of chunk N, then continues with its
  own content. This ensures continuity — a sentence that ends chunk N
  appears at the start of chunk N+1.
- **Line 133-138:** Each fragment is wrapped into an `EnrichedChunk` with
  full provenance metadata.

**Why overlap is applied after splitting, not during:** The recursive
splitter produces non-overlapping fragments. Overlap is added in a second
pass. This separation of concerns makes the splitter simpler (it does not
need to track overlap state) and makes overlap configurable without changing
the split logic.

##### `_recursive_split()` — Hierarchical text splitting (lines 143-172)

```python
def _recursive_split(self, text: str, target_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]

    # Protect equation blocks from splitting
    equations: dict[str, str] = {}
    protected = text
    for match in _DISPLAY_EQUATION_RE.finditer(text):
        placeholder = f"__EQ{len(equations)}__"
        equations[placeholder] = match.group()
        protected = protected.replace(match.group(), placeholder, 1)

    # Try splitting on paragraph boundaries first
    paragraphs = re.split(r"\n\s*\n", protected)
    if len(paragraphs) > 1:
        return self._merge_splits(paragraphs, target_chars, equations)

    # Fall back to sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    if len(sentences) > 1:
        return self._merge_splits(sentences, target_chars, equations)

    # Last resort: split on whitespace at target boundary
    words = protected.split()
    return self._merge_splits(words, target_chars, equations, join_char=" ")
```

**This is the most algorithmically interesting method in Phase 2.**

**Step 1: Base case (line 149).** If the text fits within `target_chars`,
return it as-is. No splitting needed.

**Step 2: Equation protection (lines 153-158).** Before any splitting,
display equations (`$$...$$`) are replaced with short placeholders like
`__EQ0__`, `__EQ1__`. The original equation text is stored in a dictionary.
After splitting, placeholders are restored. This guarantees that no split
point ever falls inside an equation.

**Why this works:** Placeholders are short strings (8-10 chars) that contain
no paragraph breaks, sentence-ending punctuation, or whitespace. None of
the three splitting strategies (paragraph, sentence, word) will split inside
a placeholder.

**Step 3: Hierarchical splitting.** Try three strategies in order of
semantic quality:

1. **Paragraphs (`\n\s*\n`):** The highest-quality split point. Paragraphs
   are natural semantic boundaries. If the text has paragraph breaks, split
   there first.

2. **Sentences (`(?<=[.!?])\s+`):** If the text is a single long paragraph,
   split on sentence boundaries. The regex matches whitespace following a
   sentence-ending punctuation mark.

3. **Words (`.split()`):** Last resort. If a single sentence exceeds
   `target_chars` (unlikely but possible for very small target sizes), split
   on word boundaries.

**Why this hierarchy matters:** Splitting on paragraph boundaries preserves
the most context. A paragraph usually contains a complete thought. Splitting
on sentence boundaries is the next best — each fragment at least contains
complete sentences. Word-level splitting is destructive but prevents any
fragment from exceeding the target size.

##### `_merge_splits()` — Greedy bin-packing (lines 174-206)

```python
def _merge_splits(self, parts, target_chars, equations, join_char="\n\n"):
    fragments: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part) + len(join_char)
        if current and current_len + part_len > target_chars:
            fragments.append(join_char.join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += part_len

    if current:
        fragments.append(join_char.join(current))

    # Restore equation placeholders
    restored: list[str] = []
    for frag in fragments:
        for placeholder, original in equations.items():
            frag = frag.replace(placeholder, original)
        restored.append(frag)

    return restored
```

**This is a greedy bin-packing algorithm.** It iterates over parts (paragraphs,
sentences, or words) and accumulates them into the current bin until adding
the next part would exceed `target_chars`. Then it starts a new bin.

**Why greedy, not optimal:** Optimal bin-packing (minimizing the number of
bins) is NP-hard. Greedy first-fit is O(n) and produces near-optimal results
for this use case. The slight variation in chunk sizes is actually beneficial
— it means chunks end on natural boundaries (paragraph or sentence) rather
than at an arbitrary character count.

**Lines 200-204: Equation restoration.** After splitting is complete,
placeholders are replaced with the original equation text. This is the
second half of the equation-protection strategy.

##### `_chunk_references()` — Reference batching (lines 212-235)

```python
def _chunk_references(self, references, section, paper):
    if not references:
        if section.text.strip():
            return self._chunk_section(section, paper)
        return []

    chunks: list[EnrichedChunk] = []
    batch_size = min(self.refs_per_chunk_max, max(self.refs_per_chunk_min, 5))

    for i in range(0, len(references), batch_size):
        batch = references[i : i + batch_size]
        text = "\n".join(batch)
        chunks.append(self._enrich(text=text, section=section, paper=paper))

    return chunks
```

**Why references need special handling:** A typical references section has
20-50 entries. Naive chunking would either (a) put all references in one
giant chunk (bad for retrieval — searching for "Smith et al." matches a
2000-token chunk) or (b) put one reference per chunk (wasteful — 50 tiny
chunks with minimal context). Batching 5-10 references per chunk is the
Goldilocks zone.

**Fallback (lines 219-222):** If Phase 1 did not extract individual
reference strings (the `references` list is empty), but the References
section has raw text, fall back to standard section chunking. This graceful
degradation means the chunker never silently drops references.

**Batch size clamping (line 226):** `min(max, max(min, 5))` ensures the
batch size is always within the configured bounds, even if the caller
passes unusual values.

##### `_table_to_chunk()` — Table rendering (lines 241-272)

```python
def _table_to_chunk(self, table: ExtractedTable, paper: PaperStructure) -> EnrichedChunk:
    lines: list[str] = []
    if table.caption:
        lines.append(table.caption)
    if table.headers:
        lines.append(" | ".join(table.headers))
        lines.append("-" * (len(" | ".join(table.headers))))
    for row in table.rows:
        lines.append(" | ".join(row))

    text = "\n".join(lines)
    ...
```

**Why tables are single chunks:** A table is a single semantic unit. Its
headers define columns, and rows provide data. Splitting a table across
chunks would separate headers from rows, making each chunk incomprehensible.

**The rendering format:** Tables are converted to pipe-delimited text:

```
Table 1: Performance comparison on benchmarks.
Method | F1 | mAP
-----------------------
Ours | 94.2 | 91.8
LayoutLM | 89.1 | 86.3
```

This is similar to Markdown table syntax, which embedding models handle
well because they have seen millions of Markdown tables during training.

**Metadata (lines 267-271):** Table chunks get `metadata={"type": "table",
"table_index": 0, "extraction_method": "pdfplumber"}`. The `type: "table"`
field allows downstream code to identify table chunks without parsing the
text (used in tests and in the RAG prompt builder).

**Section type (line 260):** Tables get `SectionType.UNKNOWN` because they
do not belong to a specific logical section. They are structural elements
that float in the document.

##### `_enrich()` — Chunk enrichment (lines 278-303)

```python
def _enrich(self, text, section, paper):
    page_numbers = [section.page_start]
    if section.page_end is not None and section.page_end != section.page_start:
        page_numbers = list(range(section.page_start, section.page_end + 1))

    return EnrichedChunk(
        chunk_id=self._make_id(text, paper.paper_id),
        text=text,
        token_count=self._count_tokens(text),
        section_type=section.section_type,
        section_title=section.title,
        page_numbers=page_numbers,
        paper_id=paper.paper_id,
        paper_title=paper.metadata.title,
        chunk_index=0,      # set later by chunk()
        total_chunks=0,     # set later by chunk()
        metadata={"section_order": section.order_index},
    )
```

**Page number computation (lines 285-287):** If a section spans multiple
pages (e.g., Methodology starts on page 1 and ends on page 2), the chunk
gets `page_numbers=[1, 2]`. This is computed as `range(page_start,
page_end + 1)`. If the section is on a single page, it gets a single-element
list like `[1]`.

**Placeholder values (lines 298-299):** `chunk_index=0` and `total_chunks=0`
are placeholders. The `chunk()` method overwrites them in its final pass
(lines 100-102). This two-pass approach is necessary because `total_chunks`
is not known until all chunks have been generated.

##### `_count_tokens()` — Token approximation (lines 310-312)

```python
@staticmethod
def _count_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)
```

`max(1, ...)` ensures even empty text gets a token count of 1. This prevents
division-by-zero errors in any downstream code that divides by token count.

##### `_make_id()` — Deterministic chunk IDs (lines 314-318)

```python
@staticmethod
def _make_id(text: str, paper_id: str) -> str:
    raw = f"{paper_id}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

**Why SHA-256, not UUID:** UUIDs are random. If you re-chunk the same paper,
you get different IDs, and ChromaDB treats them as new documents (doubling
storage). SHA-256 of `paper_id:text` is deterministic: same input always
produces the same ID. This makes `store()` idempotent via ChromaDB's upsert.

**Why truncate to 16 hex chars:** 16 hex characters = 64 bits = 2^64
possible IDs. The probability of collision is negligible for any realistic
number of chunks (birthday problem: ~4 billion chunks before 50% collision
probability). Shorter IDs reduce ChromaDB storage overhead.

**Why `paper_id:text`, not just `text`:** Two different papers might contain
identical text (e.g., "We use the standard experimental setup."). Including
`paper_id` in the hash ensures these produce different chunk IDs.

---

### 4.3 `src/chunking/chunk_optimizer.py`

175 lines that implement grid search over chunking parameters to find the
configuration that maximizes retrieval quality.

#### Why grid search matters

Chunk size and overlap are hyperparameters that significantly affect retrieval
quality. Consider the extremes:

- **Too small (64 tokens):** Each chunk is 1-2 sentences. High precision
  (each chunk is specific) but low recall (the answer might span two chunks).
  Also produces hundreds of chunks per paper, inflating storage and search
  latency.
- **Too large (2048 tokens):** Each chunk is a full page. High recall (the
  answer is probably in there somewhere) but low precision (lots of irrelevant
  text dilutes the embedding, and the LLM wastes context on noise).
- **Zero overlap:** Information at chunk boundaries is lost. A key sentence
  might be split across two chunks, appearing in neither.
- **Too much overlap (overlap = chunk_size / 2):** Massive duplication. Every
  sentence appears in two chunks. Storage doubles. Search returns duplicate
  content.

Grid search explores the `(chunk_size, overlap)` space systematically and
picks the combination that maximizes Precision@5 on known test queries.

#### `TestQuery` dataclass (lines 24-28)

```python
@dataclass
class TestQuery:
    query: str
    relevant_section_types: list[SectionType]
```

A test query has a natural-language question and a list of section types
where the answer should be found. For example:

```python
TestQuery(
    query="What loss function was used?",
    relevant_section_types=[SectionType.METHODOLOGY]
)
```

This is the ground truth for evaluation. The optimizer checks whether the
top-5 retrieved chunks come from the expected section types.

#### `OptimizationResult` dataclass (lines 31-38)

```python
@dataclass
class OptimizationResult:
    best_chunk_size: int
    best_overlap: int
    best_precision: float
    all_results: list[dict] = field(default_factory=list)
```

`all_results` contains every `(size, overlap, num_chunks, precision@5)`
entry from the grid search. This is useful for visualization: you can plot
a heatmap of precision vs. chunk_size and overlap to understand the
sensitivity surface.

#### `ChunkOptimizer.optimize()` (lines 66-126)

```python
def optimize(self, paper, test_queries):
    if not test_queries:
        return OptimizationResult(best_chunk_size=512, best_overlap=50, ...)

    best_precision = -1.0
    best_size = self.chunk_sizes[0]
    best_overlap = self.overlaps[0]
    all_results: list[dict] = []

    for size in self.chunk_sizes:
        for overlap in self.overlaps:
            if overlap >= size:
                continue

            chunker = SectionAwareChunker(target_tokens=size, overlap_tokens=overlap)
            chunks = chunker.chunk(paper)

            if not chunks:
                continue

            avg_p5 = self._evaluate(chunks, test_queries)
            entry = {
                "chunk_size": size, "overlap": overlap,
                "num_chunks": len(chunks), "precision_at_5": round(avg_p5, 4),
            }
            all_results.append(entry)

            if avg_p5 > best_precision:
                best_precision = avg_p5
                best_size = size
                best_overlap = overlap

    return OptimizationResult(...)
```

**Line-by-line:**

- **Lines 80-86:** If no test queries are provided, return sensible defaults
  (512 tokens, 50 overlap) with `best_precision=0.0`. This avoids crashing
  when the optimizer is called without evaluation data.

- **Lines 93-96:** Nested loop over all `(size, overlap)` pairs. The
  `overlap >= size` guard (line 95) skips invalid configurations where the
  overlap is larger than the chunk itself. Default grid:
  `[256, 512, 768, 1024] × [0, 25, 50, 100]` = 16 combinations, minus
  invalid ones = ~13 evaluations.

- **Lines 98-101:** For each configuration, create a fresh `SectionAwareChunker`
  and chunk the paper. This is the inner loop's most expensive operation,
  but chunking is CPU-only and fast (~1ms per paper).

- **Line 106:** Evaluate this configuration's chunks against all test queries.
  `_evaluate()` returns the average Precision@5.

- **Lines 116-119:** Track the best configuration seen so far. Simple
  argmax over the grid.

#### `_evaluate()` — Precision@5 computation (lines 132-150)

```python
def _evaluate(self, chunks, queries):
    total_p5 = 0.0

    for query in queries:
        scored = self._score_chunks(chunks, query.query)
        top5 = scored[:5]
        relevant = sum(
            1 for chunk in top5
            if chunk.section_type in query.relevant_section_types
        )
        total_p5 += relevant / min(5, len(top5)) if top5 else 0.0

    return total_p5 / len(queries)
```

**Precision@5** = (number of relevant results in top 5) / 5.

For each test query:
1. Score all chunks by relevance (term-overlap proxy).
2. Take the top 5.
3. Count how many come from a relevant section type.
4. Divide by 5 (or fewer, if there are fewer than 5 chunks total).

Average across all queries to get the configuration's score.

**Why Precision@5 and not Recall or NDCG:** Precision@5 directly measures
what the user sees — the top 5 search results. In a RAG system, these are
the chunks fed to the LLM. If 4 out of 5 are relevant, the answer quality
is high. Recall requires knowing *all* relevant chunks (expensive to
annotate). NDCG requires relevance grades (we only have binary relevance).

#### `_score_chunks()` — Term-overlap proxy (lines 152-174)

```python
@staticmethod
def _score_chunks(chunks, query):
    query_terms = set(query.lower().split())

    scored: list[tuple[float, int, EnrichedChunk]] = []
    for idx, chunk in enumerate(chunks):
        chunk_terms = set(chunk.text.lower().split())
        if not query_terms:
            overlap = 0.0
        else:
            overlap = len(query_terms & chunk_terms) / len(query_terms)
        scored.append((overlap, -idx, chunk))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _, _, c in scored]
```

**Why term-overlap instead of real embeddings:** The optimizer runs during
development/tuning, not production. Using the real embedding API would:
- Cost money (each grid search evaluation embeds all chunks + queries).
- Be slow (API round-trips per evaluation).
- Be non-deterministic (API responses can vary).

Term-overlap (`|query ∩ chunk| / |query|`) is a reasonable proxy: chunks
about methodology will share terms with a methodology query. It is not
perfect, but it correlates with embedding-based retrieval well enough
to identify the optimal chunk size.

**Tie-breaking (line 171):** `-idx` as the secondary sort key ensures
that when two chunks have the same term-overlap score, the one that
appears earlier in the document wins. This makes results deterministic.

---

### 4.4 `src/retrieval/embedding_service.py`

178 lines that wrap Google's Gemini text-embedding-004 API with batching,
rate-limit handling, and exponential-backoff retry.

#### Module-Level Constants (lines 20-23)

```python
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_DELAY = 0.5      # seconds between batches
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0       # seconds
```

**Why batch size 100:** The Gemini embedding API accepts up to 100 texts
per request. Larger batches reduce HTTP overhead (one round-trip for 100
texts vs. 100 round-trips). The batch size matches the API's limit.

**Why 0.5s delay:** Gemini has rate limits (1,500 requests per minute
for the free tier). A 0.5s delay between batches means ~120 batches per
minute × 100 texts per batch = 12,000 texts per minute, which stays
well within the limit while still being fast.

**Why 5 retries with 1.0s base backoff:** Transient API failures (rate
limits, network timeouts) are common. Exponential backoff (1s, 2s, 4s,
8s, 16s) gives the API time to recover. 5 retries with exponential
backoff means the system waits up to 31 seconds before giving up.

#### `EmbeddingService.__init__()` (lines 36-46)

```python
def __init__(
    self,
    model_name: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    delay: float = _DEFAULT_DELAY,
    max_retries: int = _MAX_RETRIES,
) -> None:
    self.model_name = model_name or settings.embedding_model
    self.batch_size = batch_size
    self.delay = delay
    self.max_retries = max_retries
```

**`model_name or settings.embedding_model`:** If the caller does not specify
a model, use the one from application settings (`models/text-embedding-004`
by default). This makes the service configurable at two levels: per-instance
and globally via environment variables.

#### `embed_chunks()` (lines 52-70)

```python
def embed_chunks(self, chunks: list[EnrichedChunk]) -> list[list[float]]:
    if not chunks:
        raise ValueError("No chunks to embed")
    texts = [c.text for c in chunks]
    return self._batch_embed(texts, task_type="retrieval_document")
```

**Two critical design decisions here:**

1. **`task_type="retrieval_document"`:** Google's embedding API supports
   task-type-aware embeddings. When indexing documents, you use
   `retrieval_document`. When searching, you use `retrieval_query`. The
   API optimizes the embedding vector for the specified task. This is a
   Google-specific feature that improves retrieval quality by ~5-10%
   compared to using the same task type for both.

2. **Extracting `.text` from chunks:** The embedding service only sees raw
   text. It does not know about sections, pages, or metadata. This keeps
   the embedding logic decoupled from the domain model.

#### `embed_query()` (lines 72-88)

```python
def embed_query(self, query: str) -> list[float]:
    if not query or not query.strip():
        raise ValueError("Query must not be empty")
    result = self._batch_embed([query], task_type="retrieval_query")
    return result[0]
```

**Returns `list[float]`, not `list[list[float]]`:** A query is a single
text, so the method returns a single vector. The caller does not need to
unwrap a list.

**`task_type="retrieval_query"`:** Matches the indexing task type. When
ChromaDB computes cosine similarity between a query vector (produced with
`retrieval_query`) and a document vector (produced with
`retrieval_document`), the asymmetric encoding improves relevance ranking.

#### `_batch_embed()` — Internal batching loop (lines 94-126)

```python
def _batch_embed(self, texts, task_type):
    self._ensure_api_key()

    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), self.batch_size):
        batch = texts[batch_start : batch_start + self.batch_size]
        embeddings = self._embed_with_retry(genai, batch, task_type)
        all_embeddings.extend(embeddings)

        if batch_start + self.batch_size < len(texts):
            time.sleep(self.delay)

    return all_embeddings
```

**Line-by-line:**

- **Line 109:** Validate the API key *before* any API calls. Fail fast.
- **Lines 111-112:** Lazy import of `google.generativeai`. This avoids
  import-time errors when the library is not installed (e.g., during
  unit tests that mock the API).
- **Line 113:** Configure the API key globally. This must happen before
  any `embed_content` call.
- **Lines 117-120:** Slice texts into batches of `batch_size` and embed
  each batch. Results are accumulated in `all_embeddings`.
- **Lines 123-124:** Sleep between batches *except after the last batch*.
  The `if` guard avoids an unnecessary 0.5s delay at the end.

#### `_embed_with_retry()` — Exponential backoff (lines 128-164)

```python
def _embed_with_retry(self, genai, batch, task_type):
    last_exc: Exception | None = None

    for attempt in range(self.max_retries):
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type=task_type,
            )
            embedding = result["embedding"]
            if isinstance(embedding[0], float):
                return [embedding]
            return embedding
        except Exception as exc:
            last_exc = exc
            wait = _BASE_BACKOFF * (2 ** attempt)
            logger.warning(
                "Embedding attempt %d/%d failed (%s), retrying in %.1fs",
                attempt + 1, self.max_retries, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {self.max_retries} retries: {last_exc}"
    ) from last_exc
```

**Retry strategy:**

| Attempt | Wait time | Total elapsed |
|---------|-----------|---------------|
| 1 | 1.0s | 1.0s |
| 2 | 2.0s | 3.0s |
| 3 | 4.0s | 7.0s |
| 4 | 8.0s | 15.0s |
| 5 | 16.0s | 31.0s |

After 5 failures, raise `RuntimeError`. The `from last_exc` chain preserves
the original exception for debugging.

**Single-text response handling (lines 146-148):** When the API receives a
list with one text, it may return a flat vector `[0.1, 0.2, ...]` instead
of a nested list `[[0.1, 0.2, ...]]`. The `isinstance(embedding[0], float)`
check detects this and wraps it in a list for consistent return type.

#### `_ensure_api_key()` (lines 170-177)

```python
@staticmethod
def _ensure_api_key() -> None:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Set it in .env or as an environment variable."
        )
```

**Fail-fast pattern:** Check the API key *once*, before any batching starts.
Without this check, the first batch would fail with a cryptic Google API
error. This raises a clear, actionable error message instead.

---

### 4.5 `src/retrieval/vector_store.py`

269 lines that wrap ChromaDB for persistent, metadata-filtered semantic search.

#### `SearchResult` dataclass (lines 22-32)

```python
@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float               # 0.0 to 1.0 (cosine similarity)
    paper_id: str
    section_type: str          # String, not SectionType enum
    section_title: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

**Why `section_type` is a string, not `SectionType`:** ChromaDB stores
metadata as plain strings. When results come back from ChromaDB, the
section type is a string like `"methodology"`. Converting it back to a
`SectionType` enum would add complexity and risk `ValueError` if the
stored string does not match any enum member. Keeping it as a string
makes the search result a faithful representation of what ChromaDB returns.

**Why `score` is 0.0-1.0:** ChromaDB returns cosine *distance* (0 = identical,
2 = opposite). We convert to *similarity* (`1.0 - distance`) because
humans intuit "higher = better". This conversion happens in `_parse_results`.

#### `VectorStore.__init__()` (lines 44-52)

```python
def __init__(self, persist_dir=None, collection_name=_COLLECTION_NAME):
    self.persist_dir = persist_dir or settings.chroma_persist_dir
    self.collection_name = collection_name
    self._client = None
    self._collection = None
```

**Lazy initialization:** `_client` and `_collection` are `None` until the
first operation. This means creating a `VectorStore` instance is free — no
disk I/O, no network. The ChromaDB client is only created when `store()`,
`search()`, or `delete_paper()` is called. This is important for fast
application startup and for tests that create many `VectorStore` instances.

#### `_get_collection()` — Lazy ChromaDB setup (lines 58-75)

```python
def _get_collection(self):
    if self._collection is not None:
        return self._collection

    import chromadb

    self._client = chromadb.PersistentClient(path=self.persist_dir)
    self._collection = self._client.get_or_create_collection(
        name=self.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' ready (%d items)", ...)
    return self._collection
```

**`PersistentClient`:** Data is stored on disk in `persist_dir`. Unlike
the in-memory `Client`, this survives application restarts. The directory
is created automatically by ChromaDB.

**`get_or_create_collection`:** Idempotent — creates the collection on
first run, reuses it on subsequent runs.

**`hnsw:space: "cosine"`:** Configures the HNSW (Hierarchical Navigable
Small World) index to use cosine distance. Without this, ChromaDB defaults
to L2 (Euclidean) distance, which does not account for vector magnitude.
Cosine similarity only considers the angle between vectors, making it
invariant to embedding scale — which is what you want for text similarity.

**Lazy `import chromadb`:** The import is inside the method, not at the
module top. This means `from src.retrieval.vector_store import VectorStore`
does not fail if chromadb is not installed. Tests mock at the method level
rather than needing chromadb as a hard dependency.

#### `store()` (lines 81-127)

```python
def store(self, chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
    if not chunks:
        return 0

    collection = self._get_collection()

    ids = []
    documents = []
    metadatas = []
    embedding_list = []

    for chunk, emb in zip(chunks, embeddings):
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append(self._chunk_to_metadata(chunk))
        embedding_list.append(emb)

    collection.upsert(
        ids=ids, documents=documents,
        embeddings=embedding_list, metadatas=metadatas,
    )
    return len(chunks)
```

**Why `upsert` instead of `add`:** `add` raises an error if an ID already
exists. `upsert` replaces existing documents and adds new ones. This makes
re-indexing safe: re-chunking and re-storing a paper updates its chunks
in-place rather than failing or creating duplicates.

**Mismatch validation (lines 98-101):** Every chunk must have exactly one
embedding vector. If the caller accidentally passes mismatched lists (e.g.,
due to a batching bug in the embedding service), we fail immediately with
a clear error rather than silently storing partial data.

**Four parallel lists:** ChromaDB's API expects `ids`, `documents`,
`embeddings`, and `metadatas` as separate lists of equal length. The `zip`
loop (lines 112-116) ensures they stay synchronized.

#### `search()` (lines 129-165)

```python
def search(self, query_embedding, n_results=10, paper_id=None, section_type=None):
    collection = self._get_collection()

    where = self._build_where(paper_id, section_type)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        raw = collection.query(**kwargs)
    except Exception:
        logger.error("ChromaDB query failed", exc_info=True)
        return []

    return self._parse_results(raw)
```

**Metadata filtering:** The `where` parameter tells ChromaDB to only
consider chunks matching the filter *before* computing similarity. This
is more efficient than post-filtering: if you only want methodology chunks,
ChromaDB searches only the methodology subset of the index.

**Why `include` is explicit (line 154):** By default, ChromaDB returns
everything including the full embeddings. We only need documents (text),
metadatas, and distances. Excluding embeddings from the response reduces
memory usage and network transfer.

**Error handling (lines 159-163):** A failed ChromaDB query returns an
empty list rather than crashing the application. This is critical for
production robustness — if the index is corrupted or the query is malformed,
the search endpoint returns zero results rather than a 500 error.

#### `delete_paper()` (lines 167-191)

```python
def delete_paper(self, paper_id):
    collection = self._get_collection()

    results = collection.get(where={"paper_id": paper_id}, include=[])
    ids = results.get("ids", [])

    if not ids:
        return 0

    collection.delete(ids=ids)
    return len(ids)
```

**Two-step delete:** ChromaDB does not support `DELETE WHERE paper_id = X`.
Instead, we first `get` all matching IDs, then `delete` by ID. The
`include=[]` optimization means the `get` call only returns IDs, not the
full documents and embeddings — much faster.

**Return value:** Returns the count of deleted chunks. This lets the caller
verify the operation ("I expected 15 chunks to be deleted, but only 10
were. Something is wrong.").

#### `_chunk_to_metadata()` — Metadata flattening (lines 198-212)

```python
@staticmethod
def _chunk_to_metadata(chunk):
    return {
        "paper_id": chunk.paper_id,
        "paper_title": chunk.paper_title or "",
        "section_type": chunk.section_type.value,
        "section_title": chunk.section_title or "",
        "page_numbers": ",".join(str(p) for p in chunk.page_numbers),
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "token_count": chunk.token_count,
    }
```

**Why flatten:** ChromaDB metadata values must be `str`, `int`, `float`,
or `bool`. Lists and nested objects are not supported. So:

- `page_numbers: [1, 2, 3]` becomes `"1,2,3"` (comma-joined string).
- `section_type: SectionType.METHODOLOGY` becomes `"methodology"` (enum value).
- `None` values become empty strings (`""`) because ChromaDB does not support
  `None` in metadata.

**Why store `paper_title` in metadata:** Denormalization. When a search
result is returned, you want to display "From: Deep Learning for Document
Understanding, Methodology section, pages 1-2" without making a second
database query.

#### `_build_where()` — Filter construction (lines 214-230)

```python
@staticmethod
def _build_where(paper_id, section_type):
    conditions = []
    if paper_id:
        conditions.append({"paper_id": paper_id})
    if section_type:
        conditions.append({"section_type": section_type.value})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
```

**ChromaDB filter syntax:**

- No filters: `None` (search everything).
- Single filter: `{"paper_id": "paper-A"}`.
- Combined: `{"$and": [{"paper_id": "paper-A"}, {"section_type": "methodology"}]}`.

**Why the `$and` wrapper for combined filters:** ChromaDB does not support
multiple keys in a single dict as an implicit AND. You must use the `$and`
operator explicitly. The method handles the 0, 1, and 2+ cases to produce
the minimal correct filter.

#### `_parse_results()` — Result conversion (lines 232-268)

```python
@staticmethod
def _parse_results(raw):
    results = []

    ids_list = raw.get("ids", [[]])[0]
    docs_list = raw.get("documents", [[]])[0]
    metas_list = raw.get("metadatas", [[]])[0]
    dists_list = raw.get("distances", [[]])[0]

    for i, chunk_id in enumerate(ids_list):
        meta = metas_list[i] if i < len(metas_list) else {}
        distance = dists_list[i] if i < len(dists_list) else 1.0
        score = 1.0 - distance

        page_str = meta.get("page_numbers", "")
        page_numbers = (
            [int(p) for p in page_str.split(",") if p.strip()]
            if page_str else []
        )

        results.append(SearchResult(
            chunk_id=chunk_id,
            text=docs_list[i] if i < len(docs_list) else "",
            score=round(score, 4),
            paper_id=meta.get("paper_id", ""),
            section_type=meta.get("section_type", "unknown"),
            section_title=meta.get("section_title") or None,
            page_numbers=page_numbers,
            metadata=meta,
        ))

    return results
```

**`raw["ids"][0]` — why `[0]`?** ChromaDB's `query` method supports
multiple query embeddings simultaneously, returning results as a list of
lists. We always pass a single query embedding, so results are in `[0]`.

**Distance-to-similarity conversion (line 246):** `score = 1.0 - distance`.
Cosine distance ranges from 0 (identical) to 2 (opposite). Similarity
ranges from 1.0 (identical) to -1.0 (opposite). For normalized embeddings,
distance is always 0-1, so similarity is 0-1.

**Page number reconstruction (lines 248-252):** The comma-joined string
`"1,2,3"` is split back into `[1, 2, 3]`. The `if p.strip()` guard handles
edge cases like trailing commas or empty strings.

**Default values for missing data (lines 243-244):** If ChromaDB returns
fewer metadata entries than IDs (should not happen, but defensive), we
default to empty dict and distance 1.0 (similarity 0.0).

---

### 4.6 `tests/conftest.py` (Phase 2 fixtures)

Phase 2 adds two fixtures to the shared conftest.

#### `sample_paper` fixture (lines 186-315)

```python
@pytest.fixture()
def sample_paper(sample_pages: list[PageResult]) -> PaperStructure:
    return PaperStructure(
        paper_id="test-paper-001",
        sections=[
            DetectedSection(section_type=SectionType.TITLE, ...),
            DetectedSection(section_type=SectionType.ABSTRACT, ...),
            DetectedSection(section_type=SectionType.INTRODUCTION, text=... * 20),
            DetectedSection(section_type=SectionType.METHODOLOGY, ...),
            DetectedSection(section_type=SectionType.RESULTS, ...),
            DetectedSection(section_type=SectionType.CONCLUSION, ...),
            DetectedSection(section_type=SectionType.REFERENCES, ...),
        ],
        tables=[ExtractedTable(headers=["Method", "F1", "mAP"], ...)],
        references=["[1] Smith et al. ...", ... "[12] Wu et al. ..."],
        metadata=DocumentMetadataSchema(title="Deep Learning for Document Understanding"),
    )
```

**Why 7 sections:** Covers every `SectionType` that the chunker handles
differently. The Introduction has `text=... * 20` (repeated text) to ensure
it exceeds the target chunk size and triggers splitting. The Methodology
section contains an equation (`$$L = L_{cls} + \lambda L_{layout}$$`) to
test equation protection. The References section has 12 entries to test
batching (12 / 5 = 2-3 chunks).

**Why 12 references:** Tests the boundary behavior of 5-10 batching.
With `batch_size=5`, 12 references produce 3 chunks (5 + 5 + 2). With
`batch_size=10`, 12 references produce 2 chunks (10 + 2).

#### `sample_chunks` fixture (lines 318-324)

```python
@pytest.fixture()
def sample_chunks(sample_paper: PaperStructure) -> list[EnrichedChunk]:
    from src.chunking.chunker import SectionAwareChunker
    chunker = SectionAwareChunker(target_tokens=512, overlap_tokens=50)
    return chunker.chunk(sample_paper)
```

**Depends on `sample_paper`:** The fixture chain is
`sample_pages → sample_paper → sample_chunks`. This means any test that
uses `sample_chunks` gets a fully-processed paper with realistic chunks.

**Why this exists:** Embedding and vector store tests need pre-chunked data.
Creating chunks in every test would be repetitive and would couple those
tests to the chunker implementation.

---

### 4.7 `tests/unit/test_chunker.py`

29 tests organized into 8 test groups. Every chunking rule has at least
two tests.

#### Test groups and what they verify:

| Group | Tests | What it verifies |
|-------|-------|-----------------|
| Basic chunking | 7 tests | Chunks are `EnrichedChunk`, sequential indices, unique IDs, deterministic IDs, `paper_id`, `paper_title`, `token_count` > 0, empty paper → empty list |
| Section boundary | 2 tests | Short section = 1 chunk, long section splits into multiple chunks |
| Table rule | 3 tests | Each table = 1 chunk, chunk includes headers/rows, chunk includes caption |
| Reference grouping | 2 tests | 12 refs produce 2-3 chunks, each chunk has multiple `[N]` entries |
| Equation rule | 1 test | `$$...$$` stays with surrounding context text |
| Overlap | 2 tests | With overlap > 0, chunk N+1 starts with tail of chunk N. With overlap = 0, no duplication |
| Page numbers | 2 tests | Chunks have page numbers; multi-page section has range |
| Token counting | 2 tests | `_count_tokens` approximation matches `len // 4`, minimum 1 |
| Chunk ID | 2 tests | `_make_id` is deterministic, different inputs produce different IDs |
| Recursive split | 4 tests | Short text returns as-is, splits on paragraphs, splits on sentences, preserves equations |

**Notable test: `test_equations_stay_with_context` (lines 197-211):**

```python
def test_equations_stay_with_context(self, sample_paper):
    chunker = SectionAwareChunker(target_tokens=512)
    chunks = chunker.chunk(sample_paper)

    method_chunks = [c for c in chunks if c.section_type == SectionType.METHODOLOGY]
    eq_chunks = [c for c in method_chunks if "$$" in c.text]
    assert len(eq_chunks) >= 1
    eq_chunk = eq_chunks[0]
    assert "loss function" in eq_chunk.text.lower() or "L_{cls}" in eq_chunk.text
```

This test verifies the equation-protection mechanism end-to-end. The
methodology section contains `$$L = L_{cls} + \lambda L_{layout}$$` and
surrounding text about the loss function. The test asserts that both the
equation and its context appear in the same chunk.

**Notable test: `test_recursive_split_preserves_equations` (lines 348-360):**

```python
def test_recursive_split_preserves_equations(self):
    text = (
        "Before equation. "
        "$$E = mc^2 + \\sum_{i=1}^{n} x_i$$ "
        "After equation. More text here to make it longer."
    )
    chunker = SectionAwareChunker()
    result = chunker._recursive_split(text, 60)
    all_text = " ".join(result)
    assert "$$E = mc^2" in all_text
    assert "x_i$$" in all_text
```

This tests the placeholder mechanism directly. Even with a very small
target (60 chars), the equation is never split. The assertion checks that
the full `$$...$$` block survives the splitting process.

---

### 4.8 `tests/unit/test_embedding_service.py`

14 tests covering configuration validation, successful embedding, batching,
task types, retry behavior, and constructor defaults.

#### Test groups:

| Group | Tests | What it verifies |
|-------|-------|-----------------|
| Config/validation | 5 tests | Missing API key raises `RuntimeError`, empty chunks raises `ValueError`, empty/whitespace query raises `ValueError` |
| Successful embedding | 2 tests | `embed_chunks` returns one vector per chunk, `embed_query` returns a single vector |
| Batching | 1 test | 7 chunks with `batch_size=3` produces 3 API calls (3+3+1) |
| Task types | 2 tests | `embed_chunks` uses `retrieval_document`, `embed_query` uses `retrieval_query` |
| Retry | 2 tests | Retries on transient failure with backoff, raises `RuntimeError` after max retries |
| Constructor | 2 tests | Default model from settings, custom parameters stored correctly |

**How mocking works in these tests:**

Every test that calls the embedding API uses `@patch` decorators:

```python
@patch("src.retrieval.embedding_service.settings")
def test_embed_chunks_returns_vectors(self, mock_settings):
    mock_settings.gemini_api_key = "fake-key"
    mock_settings.embedding_model = "models/text-embedding-004"

    with patch("google.generativeai.configure"):
        with patch("google.generativeai.embed_content",
                   return_value={"embedding": [[0.1, 0.2], [0.4, 0.5]]}):
            service = EmbeddingService(batch_size=10, delay=0.0)
            result = service.embed_chunks(chunks)
```

Three patches are needed:
1. `settings` — Provides a fake API key.
2. `genai.configure` — Prevents the real configuration call.
3. `genai.embed_content` — Returns fake embedding vectors.

**Notable test: `test_batch_splitting` (lines 117-140):**

Uses a `side_effect` function to capture batch sizes:

```python
def mock_embed(model, content, task_type):
    nonlocal call_count
    call_count += 1
    batch_sizes.append(len(content))
    return {"embedding": [[0.1] * 3 for _ in content]}
```

With 7 chunks and `batch_size=3`, the test asserts:
- `call_count == 3` (three API calls)
- `batch_sizes == [3, 3, 1]` (3 + 3 + 1 = 7)

**Notable test: `test_retry_on_transient_failure` (lines 190-216):**

```python
def mock_embed(model, content, task_type):
    nonlocal attempt
    attempt += 1
    if attempt < 3:
        raise ConnectionError("transient failure")
    return {"embedding": [[0.1] * 3 for _ in content]}
```

Fails twice, succeeds on the third attempt. Verifies:
- `attempt == 3` (retried correctly)
- `mock_sleep.call_count >= 2` (backoff sleeps between retries)

The `@patch("src.retrieval.embedding_service.time.sleep")` prevents actual
sleeping during tests — they run in milliseconds, not seconds.

---

### 4.9 `tests/unit/test_vector_store.py`

21 tests organized into 5 groups.

#### Test groups:

| Group | Tests | What it verifies |
|-------|-------|-----------------|
| Store | 5 tests | Returns count, stores multiple chunks, empty returns 0, mismatched lengths raises `ValueError`, upsert is idempotent |
| Search | 5 tests | Returns `SearchResult` objects, score is 0.0-1.0, respects `n_results`, empty store returns `[]`, metadata is populated |
| Metadata filters | 3 tests | Filter by `paper_id`, filter by `section_type`, combined filter |
| Delete | 3 tests | Removes chunks for a paper, nonexistent paper returns 0, search after delete returns empty |
| Helper methods | 5 tests | `_build_where` returns correct filters, `_chunk_to_metadata` flattens correctly |

**How these tests avoid needing a real embedding model:**

The tests create fake embeddings with `_fake_embedding()`:

```python
def _fake_embedding(dim: int = 8, seed: float = 0.1) -> list[float]:
    return [seed + i * 0.01 for i in range(dim)]
```

These are just sequences of floats — not meaningful embeddings, but
ChromaDB does not care. It computes cosine similarity on whatever vectors
you give it. The tests verify storage, retrieval, filtering, and deletion
logic, not embedding quality.

**Notable test: `test_store_is_idempotent` (lines 91-101):**

```python
def test_store_is_idempotent(self):
    chunks = [_make_chunk("c1", "hello")]
    embeddings = [_fake_embedding()]

    self.store.store(chunks, embeddings)
    self.store.store(chunks, embeddings)

    results = self.store.search(_fake_embedding(), n_results=10)
    assert len(results) == 1  # Not 2
```

Stores the same chunk twice. Verifies that upsert replaces rather than
duplicates — the search returns 1 result, not 2.

**Notable test: `test_search_combined_filters` (lines 197-215):**

```python
def test_search_combined_filters(self):
    chunks = [
        _make_chunk("c1", "A intro", paper_id="A", section_type=SectionType.INTRODUCTION),
        _make_chunk("c2", "A methods", paper_id="A", section_type=SectionType.METHODOLOGY),
        _make_chunk("c3", "B intro", paper_id="B", section_type=SectionType.INTRODUCTION),
    ]
    ...
    results = self.store.search(
        _fake_embedding(), n_results=10,
        paper_id="A", section_type=SectionType.INTRODUCTION,
    )
    assert len(results) == 1
    assert results[0].paper_id == "A"
    assert results[0].section_type == "introduction"
```

Three chunks, only one matches both `paper_id="A"` AND
`section_type=introduction`. Verifies the `$and` filter logic end-to-end.

---

## 5. How Phase 2 Connects to Phase 1 (Input) and Phase 3 (Output)

### 5.1 Phase 1 → Phase 2 connection

Phase 1 produces four outputs from its four processors:

```
PDFParser         →  list[PageResult]
TableExtractor    →  list[ExtractedTable]
LayoutAnalyzer    →  list[DetectedSection] + list[str] references
MetadataExtractor →  DocumentMetadataSchema
```

The `PaperStructure` schema aggregates all four into a single object:

```python
paper = PaperStructure(
    paper_id=document.id,
    sections=layout_result.sections,
    tables=table_result.tables,
    references=layout_result.references,
    metadata=metadata_result,
)
```

This is the **adapter pattern** — `PaperStructure` adapts Phase 1's
scattered outputs into Phase 2's expected input format. If Phase 1 adds
a new output (e.g., figure captions), you add a field to `PaperStructure`
without changing the chunker's method signature.

**Data flow:**

```
Phase 1:  PDF → Pages → Sections, Tables, References, Metadata
              ↓
          PaperStructure (bridge schema)
              ↓
Phase 2:  PaperStructure → Chunks → Embeddings → ChromaDB
```

### 5.2 Phase 2 → Phase 3 connection

Phase 2 produces two assets that Phase 3 (RAG / Q&A) will consume:

1. **ChromaDB index** — Contains all chunks with embeddings and metadata.
   Phase 3's retrieval pipeline will call:

   ```python
   # Phase 3 query flow:
   query_embedding = embedding_service.embed_query("What loss function was used?")
   results = vector_store.search(
       query_embedding,
       n_results=5,
       section_type=SectionType.METHODOLOGY,  # optional filter
   )
   # results is a list of SearchResult with .text, .score, .paper_id, etc.
   ```

2. **`SearchResult` objects** — Each result carries everything the LLM
   needs to generate an answer:
   - `.text` — The chunk text to include in the prompt.
   - `.score` — To rank/filter low-quality matches.
   - `.paper_id` / `.paper_title` — For citation.
   - `.section_type` / `.section_title` — For source attribution.
   - `.page_numbers` — For precise references ("see page 3").

**What Phase 3 will build on top:**

```
User Question
     │
     ▼
EmbeddingService.embed_query()
     │
     ▼
VectorStore.search()  →  top-5 SearchResults
     │
     ▼
Prompt Builder (assembles chunks into LLM context)
     │
     ▼
LLM (Gemini) generates answer with citations
     │
     ▼
Response with answer + source references
```

---

## 6. Cross-Cutting Design Decisions

### 6.1 Why ChromaDB and not Pinecone/Weaviate/FAISS?

| Criterion | ChromaDB | Pinecone | FAISS |
|-----------|----------|----------|-------|
| Deployment | Embedded (no server) | Cloud-hosted | Embedded (no server) |
| Persistence | Built-in disk storage | Managed | Manual save/load |
| Metadata filtering | Native | Native | Manual post-filtering |
| Setup | `pip install chromadb` | API key + cloud | C++ build deps |
| Cost | Free | Paid | Free |
| Scale | ~1M vectors | Billions | Billions |

ChromaDB wins for this project because:
- It requires zero infrastructure (no server, no cloud account).
- It has built-in persistence and metadata filtering.
- Scale to ~1M vectors is more than enough for a research paper system.

### 6.2 Why Gemini text-embedding-004 and not OpenAI/Cohere?

- **Task-type-aware embeddings:** Gemini distinguishes `retrieval_document`
  and `retrieval_query`, optimizing the vector for each role. OpenAI added
  this later with `text-embedding-3-*`, but Gemini had it first.
- **Cost:** Gemini's embedding API has a generous free tier (1,500 requests
  per minute, each up to 2048 tokens).
- **Consistency with Phase 1:** Phase 1 already uses Gemini (for Vision
  table extraction and JSON metadata extraction). Using the same provider
  for embeddings reduces the number of API keys and billing accounts.

### 6.3 Why deterministic chunk IDs (SHA-256) instead of UUIDs?

UUIDs are random: chunking the same paper twice produces different IDs.
This breaks upsert idempotency — ChromaDB sees the new IDs as new
documents and you get duplicates.

SHA-256 of `paper_id:text` is deterministic: same input → same ID.
Re-indexing a paper cleanly replaces its chunks via upsert. This is
critical for a system where users might re-upload or re-process papers.

### 6.4 Why approximate token counting (chars / 4)?

Options considered:
1. **tiktoken (OpenAI tokenizer):** Accurate for GPT models but adds a
   ~15MB dependency and is not compatible with Gemini's tokenizer.
2. **sentencepiece (Google tokenizer):** Accurate for Gemini but adds a
   heavy C++ dependency.
3. **chars / 4 heuristic:** Free, zero dependencies, ~90% accurate for
   English text.

For chunk *sizing* decisions, 90% accuracy is sufficient. The target is
512 tokens ± 10%. A chunk that is actually 480 or 560 tokens performs
identically in practice.

### 6.5 Why separate EmbeddingService and VectorStore?

Single Responsibility Principle. The embedding service knows how to call
the API. The vector store knows how to persist and query. Benefits:

- **Testability:** You can test the vector store with fake embeddings
  (no API calls). You can test the embedding service without a database.
- **Swappability:** Replace Gemini embeddings with OpenAI by changing
  one class. Replace ChromaDB with Pinecone by changing one class.
- **Reusability:** The embedding service can be used for non-RAG tasks
  (e.g., clustering, deduplication) without involving ChromaDB.

---

## 7. Edge Cases and Error Handling

| Scenario | Component | Handling |
|----------|-----------|---------|
| Empty paper (no sections, no tables) | `chunker.chunk()` | Returns empty list |
| Section with empty text | `_chunk_section()` | Returns empty list (line 119) |
| No parsed references but References section has text | `_chunk_references()` | Falls back to `_chunk_section()` (line 221) |
| No parsed references and empty section text | `_chunk_references()` | Returns empty list (line 223) |
| Equation with newlines inside `$$...$$` | `_DISPLAY_EQUATION_RE` | `re.DOTALL` flag makes `.` match newlines |
| Single huge sentence exceeding target | `_recursive_split()` | Falls to word-level splitting (line 171) |
| Overlap larger than chunk size | `ChunkOptimizer.optimize()` | Skipped with `if overlap >= size: continue` (line 95) |
| No test queries for optimizer | `ChunkOptimizer.optimize()` | Returns defaults (512, 50) with precision 0.0 |
| Missing Gemini API key | `EmbeddingService._ensure_api_key()` | Raises `RuntimeError` with actionable message |
| Empty chunk list | `embed_chunks()` | Raises `ValueError("No chunks to embed")` |
| Empty or whitespace query | `embed_query()` | Raises `ValueError("Query must not be empty")` |
| Transient API failure | `_embed_with_retry()` | Retries up to 5 times with exponential backoff |
| Persistent API failure | `_embed_with_retry()` | Raises `RuntimeError` after max retries |
| Single text returns flat vector | `_embed_with_retry()` | `isinstance(embedding[0], float)` detects and wraps |
| Mismatched chunk/embedding counts | `VectorStore.store()` | Raises `ValueError("Mismatch: ...")` |
| ChromaDB query failure | `VectorStore.search()` | Catches exception, logs, returns empty list |
| Delete nonexistent paper | `VectorStore.delete_paper()` | Returns 0, does not raise |
| Page numbers stored as comma-string | `_parse_results()` | Splits on comma, converts to `int`, handles empty |

---

## 8. Interview Questions and Answers

### Q1: Why did you choose section-aware chunking over LangChain's RecursiveCharacterTextSplitter?

**Answer:** LangChain's splitter treats the document as a flat string. It has
no concept of sections, tables, or equations. For research papers — which
are highly structured — this means a chunk might mix Abstract and
Introduction text, making metadata filtering impossible. Our chunker
respects section boundaries, so every chunk has a single, unambiguous
`section_type`. This enables precise metadata-filtered search: "show me
only methodology chunks" works perfectly. Additionally, we have special
handling for tables (kept as single chunks), equations (never split from
context), and references (batched 5-10 per chunk). None of this is
possible with a generic text splitter.

### Q2: How do you prevent equations from being split across chunks?

**Answer:** Before splitting, display equations (`$$...$$`) are replaced
with short placeholder strings like `__EQ0__`. These placeholders contain
no whitespace, paragraph breaks, or sentence-ending punctuation, so none
of our three splitting strategies (paragraph, sentence, word) will cut
inside them. After splitting, placeholders are restored to the original
equation text. This is effectively a **protect-then-restore** pattern —
the same approach used for handling quoted strings in parsers.

### Q3: Why 512 tokens as the default chunk size?

**Answer:** 512 tokens is the empirical sweet spot for dense retrieval.
Smaller chunks (128-256) have high precision but lose context — the
embedding model does not have enough text to build a rich semantic
representation. Larger chunks (1024-2048) have richer embeddings but
diluted signal — irrelevant text in the chunk pulls the embedding away
from the query. Studies from LlamaIndex and LangChain both converge on
512 ± 128 as the default for RAG systems. Our chunk optimizer can
confirm this for specific datasets via grid search.

### Q4: Explain how the chunk optimizer works without using real embeddings.

**Answer:** The optimizer uses term-overlap scoring as a proxy for semantic
similarity. For a query "What loss function was used?", it counts how
many query words appear in each chunk (`|query ∩ chunk| / |query|`) and
ranks chunks by this score. Then it checks whether the top-5 chunks come
from the expected section types (Methodology, in this case). This is
Precision@5. The grid search iterates over `[256, 512, 768, 1024] ×
[0, 25, 50, 100]` and picks the `(size, overlap)` pair with the highest
average Precision@5. Term-overlap is not as good as embedding similarity,
but it correlates well enough to identify the optimal hyperparameters —
and it costs zero API calls, runs in milliseconds, and is fully
reproducible.

### Q5: Why do you use `retrieval_document` and `retrieval_query` task types?

**Answer:** Google's embedding model is trained to produce *asymmetric*
embeddings. When you embed a document chunk with `retrieval_document`, the
model emphasizes the chunk's topical content. When you embed a query with
`retrieval_query`, the model emphasizes the query's intent. This asymmetry
improves retrieval quality by 5-10% compared to using the same task type
for both. It is similar to what bi-encoders do in cross-encoder reranking,
but built into the embedding model itself.

### Q6: How does ChromaDB's cosine similarity search work under the hood?

**Answer:** ChromaDB uses an HNSW (Hierarchical Navigable Small World)
graph index. HNSW is an approximate nearest neighbor algorithm that builds
a multi-layer graph where each node is a vector. The top layers have few
connections (coarse navigation), and the bottom layers have many connections
(fine-grained search). A query traverses the graph from the top layer down,
greedily moving to the nearest neighbor at each step. For cosine distance,
"nearest" means the smallest angle between vectors. HNSW provides ~95%
recall at millisecond latencies, which is why it is the default index for
most vector databases. We configure `hnsw:space: "cosine"` to use cosine
distance instead of the default L2 (Euclidean).

### Q7: Why is `VectorStore.store()` using upsert instead of add?

**Answer:** Upsert is idempotent: storing the same chunk twice replaces
it rather than creating a duplicate. This is critical because chunk IDs
are deterministic (SHA-256 of `paper_id:text`). When a user re-processes
a paper, the chunker produces the same IDs. With `add`, the second store
would fail with a duplicate ID error. With `upsert`, it silently updates
the existing records. This makes the entire pipeline re-runnable: you can
re-ingest, re-chunk, and re-store a paper at any time without manual
cleanup.

### Q8: How would you handle very large papers (100+ pages)?

**Answer:** The current system handles this naturally:
- The chunker processes sections sequentially, so memory usage is
  proportional to the largest section, not the total document.
- The embedding service batches texts (100 per API call) with rate-limit
  delays, so a 500-chunk paper would make 5 API calls over ~2 seconds.
- ChromaDB stores chunks on disk via its persistent client, so even
  10,000 chunks for a massive paper do not exhaust memory.

The main bottleneck would be the embedding API. For truly massive
documents, I would add a Celery task queue (already in docker-compose
from Phase 1) to process embeddings asynchronously and report progress.

### Q9: What would you change to support multilingual papers?

**Answer:** Three changes:
1. **Token counting:** The `4 chars/token` heuristic works for English
   but not for CJK languages (1-2 chars per token) or Arabic (different
   character frequencies). I would add a language-detection step and use
   a language-specific chars-per-token constant.
2. **Sentence splitting:** The regex `(?<=[.!?])\s+` assumes English
   sentence boundaries. For Chinese/Japanese, sentence endings use `。`
   and `！`. The recursive splitter would need a locale-aware sentence
   boundary detector.
3. **Embedding model:** `text-embedding-004` supports multilingual text,
   but some languages have lower quality than English. For non-English
   papers, I might switch to a multilingual-specific model like
   `multilingual-e5-large`.

### Q10: How does the overlap mechanism work and why is it applied after splitting?

**Answer:** Overlap is applied in a second pass after the recursive
splitter produces non-overlapping fragments. For each fragment after the
first, the tail of the previous fragment (last `overlap_chars` characters)
is prepended. So if chunk 1 ends with "...controls the layout weight"
and chunk 2 starts with "The evaluation protocol...", the actual chunk 2
text becomes "...controls the layout weight The evaluation protocol...".

It is applied *after* splitting because it is a separate concern. The
splitter's job is to divide text at natural boundaries (paragraphs,
sentences). Overlap's job is to add redundancy at those boundaries.
Mixing the two would make the splitter more complex and harder to test.
This separation also makes overlap trivially configurable — changing
`overlap_tokens` does not affect the split logic at all.

### Q11: Why did you store page_numbers as a comma-separated string in ChromaDB metadata?

**Answer:** ChromaDB metadata values must be `str`, `int`, `float`, or
`bool`. Lists are not supported. So `[1, 2, 3]` must be serialized.
Options:
1. **JSON string:** `'[1, 2, 3]'` — requires `json.loads()` to deserialize.
2. **Comma-separated:** `'1,2,3'` — requires `.split(',')` to deserialize.

I chose comma-separated because it is simpler, more readable in ChromaDB's
admin tools, and does not require importing `json`. The deserialization in
`_parse_results` is a one-liner: `[int(p) for p in page_str.split(",")
if p.strip()]`.

### Q12: What happens if the embedding API is down for 60 seconds?

**Answer:** The `_embed_with_retry` method uses exponential backoff:
1s, 2s, 4s, 8s, 16s (total: 31 seconds). After 5 failures, it raises
`RuntimeError`. In production, this would bubble up to the API layer,
which would return a 500 error with a clear message. The user can retry
later. If 60-second outages are common, I would increase `max_retries`
to 7 (1s, 2s, 4s, 8s, 16s, 32s, 64s = 127 seconds total) or add a
jitter factor to avoid thundering herd problems when multiple workers
retry simultaneously.

### Q13: How would you evaluate chunk quality in production (not just offline)?

**Answer:** I would track three metrics:
1. **Retrieval precision:** Log every search query and the chunks returned.
   When a user thumbs-up/down a RAG answer, attribute the signal to the
   retrieved chunks. Over time, compute precision per chunk-size
   configuration.
2. **LLM answer quality:** A/B test different chunk sizes by randomly
   assigning users to configurations. Measure answer quality via human
   evaluation or automated metrics (faithfulness, relevance).
3. **Chunk utilization:** Track how often each chunk is retrieved. If a
   chunk is never retrieved after N queries, it may be too generic or too
   specific. This helps identify pathological chunking.

### Q14: Why is `_count_tokens` a static method and not a module-level function?

**Answer:** It is a static method because it logically belongs to the
`SectionAwareChunker` class (it is part of the chunking logic) but does
not need access to instance state. Making it `@staticmethod` signals this:
"this function is conceptually part of the class but does not read or
modify `self`." It also makes it easy to call from tests:
`SectionAwareChunker._count_tokens("hello")` without creating an instance.
A module-level function would work too, but would pollute the module
namespace and lose the conceptual grouping.

### Q15: Walk me through what happens when a user searches "What F1 score did the model achieve?"

**Answer:** End-to-end flow:

1. **Query embedding:** `EmbeddingService.embed_query("What F1 score did
   the model achieve?")` sends the query to Gemini with
   `task_type="retrieval_query"`. Returns a 768-dimensional vector.

2. **Vector search:** `VectorStore.search(query_embedding, n_results=5)`
   passes the vector to ChromaDB. ChromaDB's HNSW index finds the 5
   nearest vectors by cosine distance.

3. **Result parsing:** `_parse_results()` converts ChromaDB's raw output
   into `SearchResult` objects. The cosine distance is converted to
   similarity (1.0 - distance). Page numbers are reconstructed from
   comma-separated strings.

4. **Expected top results:** The table chunk ("Method | F1 | mAP...")
   would score high because it contains "F1" and actual scores. The
   Results section chunk ("94.2% F1 on DocBank") would also score high.

5. **Phase 3 would then:** Assemble the top-5 chunk texts into an LLM
   prompt, ask Gemini to answer the question, and return something like:
   "The model achieved an F1 score of 94.2% on DocBank (Table 1, page 2)."

### Q16: How does the greedy bin-packing in `_merge_splits` compare to optimal packing?

**Answer:** Optimal bin-packing (minimize the number of bins such that no
bin exceeds the capacity) is NP-hard. Our greedy first-fit algorithm is
O(n) and makes a single pass: keep adding parts to the current bin until
it would overflow, then start a new bin. For text chunking, greedy
produces near-optimal results because:
- Parts (paragraphs/sentences) are much smaller than the bin capacity.
- We care more about semantic coherence (end on natural boundaries) than
  about minimizing the number of chunks.
- A slightly uneven distribution (one chunk at 400 tokens, the next at
  500) is actually beneficial — it means chunks end on paragraph
  boundaries rather than at arbitrary positions.

### Q17: What is the tradeoff between storing more metadata in ChromaDB vs. keeping it in PostgreSQL?

**Answer:** ChromaDB metadata is denormalized and flat (str/int/float only).
PostgreSQL metadata is normalized and rich (JSON, arrays, foreign keys).

**Store in ChromaDB when:** You need the data at query time for filtering
or display. `paper_id`, `section_type`, and `page_numbers` are stored in
ChromaDB because they are used in every search result.

**Store in PostgreSQL when:** You need relational queries, complex
aggregations, or data that is not needed at search time. Full author
lists, DOIs, publication dates, and processing history stay in PostgreSQL.

Our approach stores the minimum necessary metadata in ChromaDB for
zero-join retrieval, and keeps the full data in PostgreSQL for
administrative queries.

### Q18: How would you migrate from ChromaDB to Pinecone?

**Answer:** Because `EmbeddingService` and `VectorStore` are separate
classes:
1. Create a `PineconeVectorStore` class with the same `store()`,
   `search()`, and `delete_paper()` methods.
2. Map `_chunk_to_metadata()` to Pinecone's metadata format (Pinecone
   supports lists natively, so `page_numbers` would not need comma-joining).
3. Replace the `VectorStore` import in the application code.
4. The `EmbeddingService` does not change at all — it produces the same
   vectors regardless of where they are stored.

The migration would also require a one-time re-indexing: read all chunks
from ChromaDB, re-embed them (or export the existing embeddings), and
write to Pinecone. The deterministic chunk IDs ensure no duplication.

### Q19: Why does the chunker process sections in order but tables separately?

**Answer:** Sections have a natural reading order (abstract → introduction
→ methodology → ...) defined by `order_index`. Chunks from sections
preserve this order, which matters for sequential reading and for overlap
(chunk N+1's overlap comes from chunk N, which should be the preceding
text).

Tables do not have a clear position in the reading order. A table on
page 2 might be referenced from text on page 1 or page 3. Instead of
trying to interleave tables into the section order, we append them after
all section chunks. Each table chunk has `page_number` metadata, so
retrieval can still find tables relevant to a query.

### Q20: If you had to optimize this system for latency (sub-100ms search), what would you change?

**Answer:** Three changes in order of impact:
1. **Pre-load the ChromaDB collection:** Currently, `_get_collection()` is
   lazy (first call initializes). In production, I would call it at startup
   to avoid cold-start latency on the first query.
2. **Cache query embeddings:** Many users ask similar questions. A TTL
   cache (e.g., `@lru_cache` or Redis) for `embed_query()` would
   eliminate the ~200ms API call for repeated queries.
3. **Use a GPU-accelerated index:** ChromaDB's HNSW is CPU-only. For
   millions of vectors, switching to a GPU-accelerated index (FAISS with
   IVF-PQ) would reduce search latency from ~50ms to ~5ms.

For our scale (thousands of chunks per paper), ChromaDB is already
sub-100ms, so these optimizations are not yet needed.

---

*End of Phase 2 walkthrough. With this document and the source code, you
can explain every line, every design decision, and every tradeoff in a
technical interview.*
