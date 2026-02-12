# Phase 1 Walkthrough: Intelligent Document Processing Platform

This document explains every file, every class, every method, and every design
decision in Phase 1. Read this before a technical interview and you will be able
to talk through any line of code with confidence.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Ingestion Pipeline Diagram](#2-ingestion-pipeline-diagram)
3. [File-by-File Deep Dive](#3-file-by-file-deep-dive)
   - [pyproject.toml](#31-pyprojecttoml)
   - [.env.example](#32-envexample)
   - [src/config.py](#33-srcconfigpy)
   - [src/models/database.py](#34-srcmodelsdatabasepy)
   - [src/models/schemas.py](#35-srcmodelsschemaspy)
   - [src/ingestion/pdf_parser.py](#36-srcingestionpdf_parserpy)
   - [src/ingestion/table_extractor.py](#37-srcingestiontable_extractorpy)
   - [src/ingestion/layout_analyzer.py](#38-srcingestionlayout_analyzerpy)
   - [src/ingestion/metadata_extractor.py](#39-srcingestionmetadata_extractorpy)
   - [src/main.py](#310-srcmainpy)
   - [Dockerfile](#311-dockerfile)
   - [docker-compose.yml](#312-docker-composeyml)
   - [tests/conftest.py](#313-testsconftestpy)
   - [tests/unit/test_pdf_parser.py](#314-testsunittest_pdf_parserpy)
   - [tests/unit/test_table_extractor.py](#315-testsunittest_table_extractorpy)
   - [tests/unit/test_layout_analyzer.py](#316-testsunittest_layout_analyzerpy)
   - [tests/unit/test_metadata_extractor.py](#317-testsunittest_metadata_extractorpy)
4. [Cross-Cutting Design Decisions](#4-cross-cutting-design-decisions)
5. [Edge Cases and Error Handling](#5-edge-cases-and-error-handling)
6. [Interview Questions and Answers](#6-interview-questions-and-answers)

---

## 1. Architecture Overview

Phase 1 is the **ingestion layer** of a document processing platform. It takes a
raw PDF of a research paper and produces structured output: extracted text,
detected tables, logical sections, and bibliographic metadata.

```
Layer           Responsibility
─────────────   ──────────────────────────────────────────────────
API             FastAPI — upload endpoint, processing trigger
Ingestion       Four independent processors, each with a single job
Models          SQLAlchemy ORM (persistence) + Pydantic schemas (validation)
Config          Single-source-of-truth for every tunable parameter
Infrastructure  Docker + Compose for Postgres, Redis, the app, Celery
```

Key architectural principles:

- **Separation of concerns** — Each ingestion module does one thing. The PDF
  parser does not know about tables. The table extractor does not know about
  section headings. This makes each component independently testable and
  replaceable.
- **Graceful degradation** — Every component has a primary strategy and a
  fallback. PDF parser tries native text, then OCR. Table extractor tries
  pdfplumber, then Gemini Vision. Metadata extractor tries heuristics, then
  LLM. If the fallback also fails, the system returns partial results rather
  than crashing.
- **Confidence scoring** — Every extraction result carries a float confidence
  score in [0, 1]. Downstream consumers (search, summarization in later
  phases) can use these scores to weight or filter results.

---

## 2. Ingestion Pipeline Diagram

```
                         ┌──────────────┐
                         │   Upload PDF │
                         │   (POST /    │
                         │   documents/ │
                         │   upload)    │
                         └──────┬───────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Save to disk        │
                    │   SHA-256 hash        │
                    │   Return doc_id       │
                    └───────────┬───────────┘
                                │
               POST /documents/{doc_id}/process
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               │
  ┌──────────────────┐  ┌──────────────────┐    │
  │   PDFParser      │  │  TableExtractor  │    │
  │                  │  │                  │    │
  │  For each page:  │  │  For each page:  │    │
  │  ┌────────────┐  │  │  ┌────────────┐  │    │
  │  │ Has native │  │  │  │ pdfplumber │  │    │
  │  │ text?      │  │  │  │ finds      │  │    │
  │  └─┬──────┬───┘  │  │  │ tables?    │  │    │
  │ Yes│      │No    │  │  └─┬──────┬───┘  │    │
  │    ▼      ▼      │  │Yes │      │No/   │    │
  │  PyMuPDF  OCR    │  │   ▼      │Low   │    │
  │  native  Tesse-  │  │ Parse    │qual. │    │
  │  text    ract    │  │ headers  │      │    │
  │                  │  │ + rows   ▼      │    │
  │  ┌────────────┐  │  │       Gemini    │    │
  │  │ _clean_text│  │  │       Vision    │    │
  │  │ confidence │  │  │       fallback  │    │
  │  └────────────┘  │  │                 │    │
  └────────┬─────────┘  └────────┬────────┘    │
           │                     │              │
           ▼                     │              │
  ┌──────────────────┐           │              │
  │  LayoutAnalyzer  │           │              │
  │                  │           │              │
  │  Takes pages[] ──┤           │              │
  │  from PDFParser  │           │              │
  │                  │           │              │
  │  For each line:  │           │              │
  │  ┌────────────┐  │           │              │
  │  │ Is heading?│  │           │              │
  │  │ Classify   │  │           │              │
  │  │ section    │  │           │              │
  │  └────────────┘  │           │              │
  │                  │           │              │
  │  Extract refs    │           │              │
  │  from References │           │              │
  │  section         │           │              │
  └────────┬─────────┘           │              │
           │                     │              │
           ▼                     │              │
  ┌──────────────────┐           │              │
  │MetadataExtractor │           │              │
  │                  │           │              │
  │  Takes pages[] ──┤           │              │
  │  from PDFParser  │           │              │
  │                  │           │              │
  │  ┌────────────┐  │           │              │
  │  │ Heuristic  │  │           │              │
  │  │ regex on   │  │           │              │
  │  │ first 3 pg │  │           │              │
  │  └─────┬──────┘  │           │              │
  │        │         │           │              │
  │  ┌─────▼──────┐  │           │              │
  │  │ LLM Gemini │  │           │              │
  │  │ JSON mode  │  │           │              │
  │  └─────┬──────┘  │           │              │
  │        │         │           │              │
  │  ┌─────▼──────┐  │           │              │
  │  │ _merge()   │  │           │              │
  │  │ LLM > heur │  │           │              │
  │  └────────────┘  │           │              │
  └────────┬─────────┘           │              │
           │                     │              │
           ▼                     ▼              │
  ┌───────────────────────────────────────────┐ │
  │              JSON Response                │ │
  │  {                                        │ │
  │    document_id, status, total_pages,      │ │
  │    avg_confidence, sections_found,        │ │
  │    tables_found, references_found,        │ │
  │    metadata: { title, authors, ... }      │ │
  │  }                                        │ │
  └───────────────────────────────────────────┘ │
```

**Data flow summary:**
1. User uploads a PDF via `POST /documents/upload`. File is saved to disk with a UUID filename. SHA-256 hash is computed for deduplication.
2. User triggers `POST /documents/{doc_id}/process`. This runs four processors sequentially:
   - **PDFParser** reads the PDF and produces a `list[PageResult]` — one per page, each with text, extraction method, and confidence score.
   - **TableExtractor** reads the same PDF independently and produces a `list[ExtractedTable]` — one per detected table.
   - **LayoutAnalyzer** consumes `PDFParser`'s `pages[]` output and segments them into `list[DetectedSection]` plus a `list[str]` of individual reference strings.
   - **MetadataExtractor** also consumes `PDFParser`'s `pages[]` output and returns `DocumentMetadataSchema` with title, authors, abstract, DOI, keywords, etc.
3. All results are combined into a JSON response.

**Why sequential?** In Phase 1, processing is synchronous for simplicity. In
Phase 2, this becomes a Celery task chain. The important thing is that
`PDFParser` runs first because `LayoutAnalyzer` and `MetadataExtractor` depend
on its output. `TableExtractor` is independent and could run in parallel with
the others.

---

## 3. File-by-File Deep Dive

---

### 3.1 `pyproject.toml`

**What it does:** Defines the project metadata, all dependencies, dev
dependencies, and tool configurations for the entire project.

**Why it exists:** Modern Python projects use `pyproject.toml` as the single
configuration file, replacing the old `setup.py` + `setup.cfg` +
`requirements.txt` pattern. PEP 621 standardized this.

**Line-by-line:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```
- **Hatchling** is the build backend. It's lighter than setuptools and designed
  for modern Python. The build system section tells `pip` how to build the
  project when you run `pip install -e .`

```toml
[tool.hatch.build.targets.wheel]
packages = ["src"]
```
- Tells hatchling that the importable package lives in the `src/` directory.
  Without this line, hatchling can't find the code and `pip install -e .` fails
  with a metadata-generation error. This is a common gotcha with hatchling
  when your package directory doesn't match the project name.

```toml
dependencies = [
    "fastapi>=0.115.0",      # Web framework
    "uvicorn[standard]>=0.32.0",  # ASGI server; [standard] adds uvloop + httptools
    "pydantic>=2.10.0",      # Data validation
    "pydantic-settings>=2.6.0",   # .env file loading
    "sqlalchemy>=2.0.36",    # ORM
    "asyncpg>=0.30.0",       # Async Postgres driver
    "alembic>=1.14.0",       # DB migrations
    "redis>=5.2.0",          # Redis client
    "celery[redis]>=5.4.0",  # Task queue; [redis] pulls in the Redis transport
    "PyMuPDF>=1.25.0",       # PDF text extraction (native)
    "pytesseract>=0.3.13",   # OCR wrapper
    "pdfplumber>=0.11.0",    # Table extraction
    "Pillow>=11.0.0",        # Image handling for OCR
    "google-generativeai>=0.8.0",  # Gemini API client
    "python-multipart>=0.0.17",    # File upload parsing for FastAPI
    "python-dotenv>=1.0.1",  # .env loading
]
```
- Every dependency is pinned with `>=` minimum versions. This balances
  reproducibility (won't install ancient broken versions) with flexibility
  (won't block minor updates).
- `uvicorn[standard]` is important — the `standard` extra installs `uvloop`
  (faster event loop) and `httptools` (faster HTTP parsing), which roughly
  doubles throughput compared to vanilla uvicorn.

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]
```
- Dev dependencies are separate so production containers don't carry testing
  tools. Install with `pip install -e ".[dev]"`.
- `httpx` is the recommended async HTTP client for testing FastAPI apps with
  `TestClient`.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```
- `asyncio_mode = "auto"` means every `async def test_*` is automatically
  treated as an async test without needing `@pytest.mark.asyncio` on every one.

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]
```
- E=pycodestyle errors, F=pyflakes, I=isort, N=naming conventions, W=warnings,
  UP=pyupgrade, B=bugbear (common pitfalls), SIM=simplify.

```toml
[tool.mypy]
strict = true
```
- Strict mode enables all optional type-checking flags. In an interview, you
  can say: "We use mypy strict because catching type errors statically is
  cheaper than catching them in production."

**Design decisions:**
- Chose hatchling over setuptools because it's faster and has better defaults
  for modern Python.
- Pinned minimum versions instead of exact versions to allow security patches
  while avoiding known-broken releases.

**Connections:** Every other file in the project depends on the packages listed
here. `pip install -e .` uses this file to install the project.

---

### 3.2 `.env.example`

**What it does:** Documents every environment variable the application reads.

**Why it exists:** Developers copy this to `.env` and fill in their own values.
It serves as living documentation — if a new env var is added to `config.py`,
it must also appear here.

```
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=models/text-embedding-004
DATABASE_URL=postgresql+asyncpg://docuser:docpass@postgres:5432/intelligent_doc
REDIS_URL=redis://redis:6379/0
CHROMA_PERSIST_DIR=./data/chroma
UPLOAD_DIR=./data/uploads
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

**Design decisions:**
- Defaults point to Docker Compose service names (`postgres`, `redis`) so the
  file works out-of-the-box with `docker compose up`.
- `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=200` are standard RAG chunking values.
  1000 characters is roughly a paragraph. 200 overlap prevents losing context at
  chunk boundaries.
- API keys default to empty string rather than failing on import, so the app
  starts without Gemini configured (it just skips LLM-powered features).

**Connections:** `src/config.py` reads these values.

---

### 3.3 `src/config.py`

**What it does:** Centralizes all application configuration into a single
validated object.

**Why it exists:** Scattering `os.getenv()` calls across 10 files creates bugs.
A central Settings class with type annotations catches configuration errors at
startup, not at 3am when a Celery worker hits a codepath that reads a missing
env var.

**Line-by-line:**

```python
from __future__ import annotations
```
- Enables PEP 604 union syntax (`str | None`) at runtime on Python 3.11+.
  Without this, type annotations are evaluated eagerly and `str | None` can fail
  in some edge cases with older Pydantic.

```python
from functools import lru_cache
```
- Used to create a singleton. `lru_cache(maxsize=1)` means the Settings object
  is constructed once and cached forever.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
```
- `BaseSettings` from pydantic-settings automatically reads values from
  environment variables. Each field name (e.g., `gemini_api_key`) maps to the
  env var `GEMINI_API_KEY`.
- `case_sensitive=False` means `gemini_api_key`, `GEMINI_API_KEY`, and
  `Gemini_Api_Key` all work. This is forgiving for Docker environments where
  env vars are conventionally UPPER_CASE.
- `env_file=".env"` tells it to also read a `.env` file if one exists. Env vars
  take precedence over the file (standard 12-factor behavior).

```python
    gemini_api_key: str = ""
```
- Defaults to empty string instead of requiring the key. This means the app
  starts without Gemini configured — it just skips LLM features. This is
  critical for local development and testing.

```python
    database_url: str = "postgresql+asyncpg://docuser:docpass@localhost:5432/intelligent_doc"
```
- The `+asyncpg` driver suffix tells SQLAlchemy to use the async Postgres
  driver. If you accidentally use `psycopg2` here, async queries will fail.

```python
    max_upload_size_mb: int = 50
```
- Not in `.env.example` but still configurable via `MAX_UPLOAD_SIZE_MB` env var.
  Pydantic-settings auto-discovers it from the field name.

```python
    chunk_size: int = 1000
    chunk_overlap: int = 200
```
- For Phase 2 RAG chunking. Defined here in Phase 1 so the config schema is
  stable when we add the chunking pipeline later.

```python
    def ensure_dirs(self) -> None:
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
```
- Called on FastAPI startup. `parents=True` creates intermediate directories.
  `exist_ok=True` means it doesn't error if the directory already exists.
  This is idempotent — safe to call multiple times.

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings: Settings = get_settings()
```
- **Singleton pattern** via `lru_cache`. The function is called once, the
  result is cached, and every subsequent call returns the same object.
- The module-level `settings` object means any file can do
  `from src.config import settings` and get the same instance.
- `maxsize=1` is the correct value — we only ever need one Settings object.

**Design decisions:**
- **Why pydantic-settings instead of raw os.getenv?** Type validation. If
  `CHUNK_SIZE` is set to `"abc"`, pydantic raises a clear error at startup.
  With `os.getenv`, you'd get a cryptic `ValueError` deep in the chunking code.
- **Why lru_cache instead of a global variable?** `lru_cache` is lazily
  evaluated — the Settings object isn't constructed until first access. This
  matters for testing, where you might want to patch env vars before Settings
  reads them.
- **Why defaults for everything?** The app should start with zero configuration
  for local development. Only `GEMINI_API_KEY` and `DATABASE_URL` are truly
  required in production, and even those have defaults so tests don't need a
  running database.

**Connections:**
- Imported by `pdf_parser.py` (no, actually it's not — PDFParser is
  self-contained)
- Imported by `table_extractor.py` (for `settings.gemini_api_key` and
  `settings.gemini_model`)
- Imported by `metadata_extractor.py` (for `settings.gemini_api_key` and
  `settings.gemini_model`)
- Imported by `main.py` (for `settings.upload_dir`, `settings.max_upload_size_mb`,
  `settings.ensure_dirs()`)

---

### 3.4 `src/models/database.py`

**What it does:** Defines the PostgreSQL table schema via SQLAlchemy ORM models.

**Why it exists:** The ORM models are the contract between the application and
the database. They define what data gets persisted and how it's indexed.

**Line-by-line:**

```python
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
```
- SQLAlchemy 2.0+ uses `DeclarativeBase` instead of the old
  `declarative_base()` function. This is the new-style API and supports
  type-checked relationships.

#### Document model

```python
class Document(Base):
    __tablename__ = "documents"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
```
- **UUID primary key** instead of auto-incrementing integer. UUIDs are globally
  unique, so they're safe for distributed systems, don't leak row counts, and
  can be generated client-side without a database round-trip.
- `as_uuid=True` means Python code works with `uuid.UUID` objects, not raw
  strings.
- `default=uuid.uuid4` — note: this passes the function itself, not
  `uuid.uuid4()`. SQLAlchemy calls it at insert time to generate a fresh UUID
  for each row.

```python
    file_hash: str = Column(String(64), nullable=False, unique=True, index=True)
```
- SHA-256 hash of the file content. 64 chars because SHA-256 produces 32 bytes
  = 64 hex characters.
- `unique=True` prevents uploading the same PDF twice.
- `index=True` because we'll query by hash to check for duplicates.

```python
    status: str = Column(
        Enum("pending", "processing", "completed", "failed", name="document_status"),
        ...
    )
```
- PostgreSQL native ENUM type. More efficient than a VARCHAR with a CHECK
  constraint — the DB stores an integer internally and validates values at the
  database level.
- The `name="document_status"` is the PostgreSQL type name. Without it,
  SQLAlchemy auto-generates an ugly name.

```python
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: datetime = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
```
- `default=datetime.utcnow` — again, passes the function, not its result.
- `onupdate=datetime.utcnow` automatically sets `updated_at` every time the
  row is modified. This is SQLAlchemy-level, not database-level, so it only
  works when updates go through the ORM.

```python
    metadata_record = relationship("DocumentMetadata", back_populates="document", uselist=False)
```
- `uselist=False` makes this a one-to-one relationship. Without it, SQLAlchemy
  returns a list.
- Named `metadata_record` instead of `metadata` to avoid shadowing SQLAlchemy's
  internal `.metadata` attribute on the Base class.

```python
    pages = relationship("Page", back_populates="document", order_by="Page.page_number")
```
- `order_by` ensures pages always come back in page order, not insertion order.

#### DocumentMetadata model

```python
    authors: list | None = Column(JSONB, nullable=True)
```
- **JSONB** is PostgreSQL's binary JSON type. Faster to query than JSON and
  supports GIN indexes. Used here because authors are a variable-length list
  that doesn't need its own table.

```python
    __table_args__ = (
        Index("ix_document_metadata_document_id", "document_id", unique=True),
    )
```
- Composite `__table_args__` tuple. The unique index on `document_id` enforces
  one-to-one at the database level (not just the ORM level).
- Trailing comma inside the tuple is required — without it, Python interprets
  the parentheses as grouping, not a tuple.

#### Page model

```python
    __table_args__ = (
        Index("ix_pages_document_page", "document_id", "page_number", unique=True),
    )
```
- **Composite unique index**: guarantees no document can have two rows for
  the same page number. This prevents bugs in re-processing.

#### TableRecord model

```python
    __table_args__ = (
        Index("ix_tables_document_page", "document_id", "page_number"),
    )
```
- **Non-unique** index on (document_id, page_number). A page can have multiple
  tables, so uniqueness isn't enforced here. The index speeds up "give me all
  tables for this page" queries.

#### Section model

```python
    __table_args__ = (
        Index("ix_sections_document_order", "document_id", "order_index"),
    )
```
- Index on (document_id, order_index) for fast ordered retrieval of a
  document's sections.

**Design decisions:**
- **Why UUIDs over integer IDs?** UUIDs prevent ID enumeration attacks (an
  attacker can't guess `doc/2` exists because they saw `doc/1`). They also
  allow the client to generate IDs without a DB round-trip, which matters for
  distributed task queues.
- **Why JSONB for authors/keywords?** These are variable-length lists that
  don't need relational queries (we won't JOIN on author names). JSONB avoids
  the overhead of a separate `document_authors` table.
- **Why separate Page model?** Per-page storage enables page-level confidence
  tracking, per-page extraction method recording, and efficient re-processing
  of individual pages.
- **Why `ondelete="CASCADE"`?** When a document is deleted, all its pages,
  tables, sections, and metadata are automatically deleted. This prevents
  orphaned rows.

**Connections:**
- Imported by `main.py` (not directly in Phase 1, but will be in Phase 2)
- The schema mirrors the Pydantic schemas in `schemas.py` — they share field
  names and types so conversion is straightforward.

---

### 3.5 `src/models/schemas.py`

**What it does:** Defines all Pydantic models for request/response validation
and for passing data between ingestion components.

**Why it exists:** The ORM models define what's stored. The Pydantic schemas
define what's transmitted — over the API and between functions. Keeping them
separate follows the **DTO pattern** (Data Transfer Object): internal storage
representations can evolve independently of the API contract.

**Line-by-line:**

#### Enums

```python
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```
- Inherits from both `str` and `Enum`. The `str` mixin means the enum value
  serializes to a plain string in JSON (`"pending"`, not `"DocumentStatus.PENDING"`).
- This is the standard pattern for Pydantic-compatible enums. Without `str`,
  JSON serialization breaks.

```python
class SectionType(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    ...
    UNKNOWN = "unknown"
```
- 12 values covering standard research paper sections.
- `UNKNOWN` is a catch-all for numbered headings that don't match any known
  section pattern (e.g., "7. Limitations").

#### PageResult

```python
class PageResult(BaseModel):
    page_number: int
    text: str
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0)
    char_count: int = Field(ge=0)
```
- `Field(ge=0.0, le=1.0)` adds **validation constraints**. If code tries to
  create a PageResult with confidence=1.5, Pydantic raises a `ValidationError`.
  This catches bugs early instead of storing garbage data.
- `ge` = greater-than-or-equal, `le` = less-than-or-equal.
- This is the data contract between `PDFParser` and downstream consumers
  (`LayoutAnalyzer`, `MetadataExtractor`).

#### ExtractedTable

```python
class ExtractedTable(BaseModel):
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None
    extraction_method: str = "pdfplumber"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
```
- `extraction_method` is a plain string, not an enum, because we might add more
  extraction methods (camelot, tabula) without changing the schema.
- `headers` separate from `rows` because headers have special semantic meaning
  (they become column names in downstream analysis).

#### DocumentMetadataSchema

```python
class DocumentMetadataSchema(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
```
- `default_factory=list` creates a new empty list for each instance. If we used
  `default=[]` instead, all instances would share the same list object — a
  classic Python mutable default bug.

#### API response schemas

```python
class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
```
- `from_attributes=True` (formerly `orm_mode=True` in Pydantic v1) allows
  creating this schema from SQLAlchemy ORM objects:
  `DocumentUploadResponse.model_validate(orm_object)`. Pydantic reads attributes
  instead of requiring a dict.

```python
class DocumentDetailResponse(BaseModel):
    ...
    metadata: DocumentMetadataSchema | None = None
    sections: list[DetectedSection] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
```
- Nested Pydantic models. Pydantic handles recursive serialization
  automatically.

**Design decisions:**
- **Why separate schemas from ORM models?** The API might return a subset of
  fields, or compute derived fields. The database might store fields the API
  never exposes. Coupling them together means every DB migration forces an API
  change.
- **Why `str | None` instead of `Optional[str]`?** PEP 604 syntax. Functionally
  identical, but more readable and modern. The `from __future__ import annotations`
  at the top ensures compatibility.
- **Why `default_factory=list` everywhere?** Prevents mutable default sharing
  bugs. This is a Python best practice that Pydantic enforces.

**Connections:**
- Imported by every ingestion module (they return these schemas)
- Imported by `main.py` for API response types
- Imported by `conftest.py` for test fixtures

---

### 3.6 `src/ingestion/pdf_parser.py`

**What it does:** Extracts text from every page of a PDF. Decides per-page
whether to use native text extraction (PyMuPDF) or OCR (Tesseract). Computes
a confidence score for each page.

**Why it exists:** This is the foundation of the entire pipeline. Everything
downstream — layout analysis, metadata extraction — depends on having clean text.
The dual-strategy approach (native + OCR) handles both digital-native PDFs and
scanned papers.

#### Module-level constants

```python
_NATIVE_TEXT_THRESHOLD = 0.3
_MIN_CHAR_COUNT = 20
_OCR_DPI = 300
```
- `_NATIVE_TEXT_THRESHOLD = 0.3`: If the average characters per text block is
  below `0.3 * 100 = 30`, the page is considered image-based. This threshold
  was tuned empirically — most native PDF pages have 100+ chars per block.
- `_MIN_CHAR_COUNT = 20`: Absolute minimum. Pages with fewer than 20 characters
  are definitely image-based (blank pages, cover images, etc.).
- `_OCR_DPI = 300`: Standard DPI for OCR. Higher = more accurate but slower.
  300 is the sweet spot recommended by Tesseract documentation.
- Underscore prefix = module-private. These shouldn't be imported elsewhere.

#### `PDFParser.__init__`

```python
def __init__(
    self,
    ocr_dpi: int = _OCR_DPI,
    native_threshold: float = _NATIVE_TEXT_THRESHOLD,
) -> None:
    self.ocr_dpi = ocr_dpi
    self.native_threshold = native_threshold
```
- Constructor takes tunable parameters with sensible defaults. This makes the
  class testable — tests can override DPI or thresholds.
- **Dependency injection via constructor** rather than hardcoded constants.

#### `PDFParser.extract`

```python
def extract(self, pdf_path: str | Path) -> PDFExtractionResult:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
```
- Accepts both `str` and `Path` for caller convenience. Immediately converts to
  `Path` for consistent handling.
- **Fail-fast validation**: checks existence before attempting to open. This
  gives a clear error message instead of a cryptic fitz error.

```python
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc
```
- Wraps the fitz exception in a `RuntimeError` with a clear message.
- `from exc` preserves the original traceback for debugging.
- **Why RuntimeError instead of letting fitz's exception propagate?** Callers
  shouldn't depend on fitz internals. If we swap to a different PDF library,
  the exception type stays the same.

```python
    pages: list[PageResult] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            if self._needs_ocr(page):
                result = self._extract_ocr_page(page, page_num)
            else:
                result = self._extract_native_page(page, page_num)
            pages.append(result)
    finally:
        doc.close()
```
- **Strategy pattern**: for each page, the `_needs_ocr` method decides which
  extraction strategy to use. The caller doesn't know or care which was chosen.
- `try/finally` ensures the document is closed even if extraction fails on
  page 5 of 20. File handle leaks are a real problem with long-running servers.
- `range(len(doc))` iterates by index. We need the index for `page_number` in
  the results.

```python
    total = len(pages)
    avg_conf = sum(p.confidence for p in pages) / total if total else 0.0
```
- Average confidence across all pages. The `if total else 0.0` guard prevents
  ZeroDivisionError on empty PDFs.

#### `PDFParser._needs_ocr`

```python
def _needs_ocr(self, page: fitz.Page) -> bool:
    text = page.get_text("text") or ""
    if len(text.strip()) < _MIN_CHAR_COUNT:
        return True
```
- First check: absolute character count. If there's almost no text, don't
  bother with more analysis.
- `or ""` handles the case where `get_text` returns `None`.

```python
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]
```
- `get_text("dict")` returns structured data about the page layout. Block
  type 0 = text, type 1 = image.
- `TEXT_PRESERVE_WHITESPACE` flag keeps original spacing, which gives more
  accurate character counts.

```python
    total_chars = sum(
        len(span.get("text", ""))
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )
```
- Triple nested generator expression walks the PyMuPDF document structure:
  blocks → lines → spans. A span is a run of text with the same font/size.
- This is a flat comprehension (one expression with three `for` clauses), not
  three nested loops. Python evaluates them left-to-right.

```python
    chars_per_block = total_chars / len(text_blocks) if text_blocks else 0
    return chars_per_block < self.native_threshold * 100
```
- The heuristic: if average chars per block is below 30 (0.3 * 100), the
  "text" is probably artefacts (stray characters from image compression), not
  real content.

#### `PDFParser._extract_native_page`

```python
def _extract_native_page(self, page: fitz.Page, page_number: int) -> PageResult:
    raw_text = page.get_text("text") or ""
    cleaned = self._clean_text(raw_text)
    confidence = self._compute_confidence(page, cleaned, method="native")
    return PageResult(
        page_number=page_number,
        text=cleaned,
        extraction_method=ExtractionMethod.NATIVE,
        confidence=round(confidence, 4),
        char_count=len(cleaned),
    )
```
- Simple three-step pipeline: extract → clean → score.
- `round(confidence, 4)` prevents floating-point noise like `0.8700000001`.
- `char_count=len(cleaned)` is set after cleaning, so it reflects the final
  text length.

#### `PDFParser._extract_ocr_page`

```python
def _extract_ocr_page(self, page: fitz.Page, page_number: int) -> PageResult:
    import pytesseract
```
- **Lazy import**: pytesseract is only imported when OCR is actually needed.
  This means the app starts faster and doesn't crash if Tesseract isn't
  installed but OCR is never triggered.

```python
    pix = page.get_pixmap(dpi=self.ocr_dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```
- Renders the PDF page to a pixel map at the configured DPI.
- Converts to a PIL Image because pytesseract expects PIL images.
- `"RGB"` format — Tesseract works with color images.

```python
    raw_text: str = pytesseract.image_to_string(img)
```
- Calls Tesseract OCR on the rendered image. Returns a plain string.
- Tesseract uses the language pack installed in the Docker container
  (`tesseract-ocr-eng`).

#### `PDFParser._clean_text`

```python
@staticmethod
def _clean_text(text: str) -> str:
    text = re.sub(r"[^\S \n\t]+", " ", text)
```
- `[^\S \n\t]+` matches any whitespace character that is NOT a regular space,
  newline, or tab. This removes control characters (form feeds, vertical tabs,
  zero-width spaces) that PDF extractors sometimes emit. Replaces them with a
  single regular space.
- `\S` matches non-whitespace. `[^\S]` therefore matches whitespace. Adding
  ` \n\t` to the negated set excludes those three "normal" whitespace chars.

```python
    text = re.sub(r"\n{3,}", "\n\n", text)
```
- Collapses three or more consecutive newlines into exactly two (one blank
  line). This normalizes the excessive spacing that OCR often produces.

```python
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
```
- Strips trailing whitespace from each line individually, then strips leading/
  trailing whitespace from the entire text.

#### `PDFParser._compute_confidence`

```python
@staticmethod
def _compute_confidence(page: fitz.Page, text: str, method: str) -> float:
    char_count = len(text)
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return 0.0
    density = char_count / (page_area / 1000)
```
- `density` = characters per 1000 square PDF units. A typical US Letter page is
  612 x 792 = ~485,000 square units, so dividing by 1000 gives ~485. A
  well-filled page might have 3000 characters, so density ≈ 6.2.

```python
    if method == "native":
        score = min(1.0, 0.85 + density * 0.01)
    else:
        score = min(1.0, 0.5 + density * 0.02)
```
- **Native starts at 0.85**: native extraction is inherently reliable, so the
  baseline is high. Dense text pushes it toward 1.0.
- **OCR starts at 0.5**: OCR is inherently less reliable. Dense text pushes it
  up faster (0.02 per density unit vs 0.01) because dense OCR output suggests
  the page had lots of readable text.
- `min(1.0, ...)` caps at 1.0.

```python
    return max(0.0, min(score, 1.0))
```
- Double-clamping ensures result is always in [0, 1] regardless of edge cases.

**Edge cases handled:**
- Missing file → `FileNotFoundError`
- Corrupted PDF → `RuntimeError`
- Zero-page PDF → returns `PDFExtractionResult(pages=[], total_pages=0, avg_confidence=0.0)`
- Scanned/image page → automatic OCR fallback
- Mixed PDF (some native, some scanned pages) → per-page decision
- Zero page area → confidence 0.0 (not division by zero)
- Control characters in text → cleaned out

**Connections:**
- Returns `PDFExtractionResult` (from `schemas.py`)
- Output feeds into `LayoutAnalyzer.analyze(pages)` and
  `MetadataExtractor.extract(pages)`
- Tests in `test_pdf_parser.py`

---

### 3.7 `src/ingestion/table_extractor.py`

**What it does:** Extracts tables from PDF pages. Uses pdfplumber as the primary
method and falls back to Gemini Vision for complex tables.

**Why it exists:** Research papers are full of results tables. Rule-based
extraction (pdfplumber) is fast and free but breaks on complex layouts. Vision
LLMs handle visual tables but cost money and are slower. The two-tier approach
gives the best of both worlds.

#### Module-level constants

```python
_MIN_CELL_FILL_RATIO = 0.3
_VISION_DPI = 200
```
- `0.3` = if fewer than 30% of cells have content, the extraction is too noisy.
- `_VISION_DPI = 200` — lower than OCR's 300 because we're sending to a vision
  model, not doing character recognition. 200 DPI keeps the image small enough
  for fast API calls while still being readable.

#### `TableExtractor.extract_tables`

```python
def extract_tables(self, pdf_path: str | Path) -> TableExtractionResult:
    ...
    tables: list[ExtractedTable] = []
    page_texts = self._get_page_texts(pdf_path)
```
- `_get_page_texts` does a quick PyMuPDF pass to get raw text for each page.
  This text is used later for caption matching.
- We open the PDF twice (once for text, once for pdfplumber). This is
  intentional — pdfplumber and PyMuPDF use different PDF parsing backends and
  can't share a file handle.

```python
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            raw_tables = self._pdfplumber_extract(page)
            for tbl_idx, raw_tbl in enumerate(raw_tables):
                if self._needs_vision_fallback(raw_tbl):
                    ...
                    vision_result = self._gemini_vision_extract(pdf_path, page_idx)
                    if vision_result:
                        ...
                        break  # vision fallback already handled the page
```
- **Escalation pattern**: try pdfplumber first. If quality is low, escalate to
  Gemini Vision.
- `break` after vision fallback: if we send the page to Gemini, we use its
  results for ALL tables on that page (Gemini returns all tables at once). No
  need to process remaining pdfplumber results for that page.

#### `TableExtractor._pdfplumber_extract`

```python
@staticmethod
def _pdfplumber_extract(page: pdfplumber.page.Page) -> list[list[list[str | None]]]:
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 4,
        "join_tolerance": 4,
    }
```
- `"lines"` strategy: detect tables by finding horizontal and vertical ruled
  lines in the PDF. This works well for academic papers that use bordered
  tables.
- `snap_tolerance=4`: lines within 4 PDF units of each other are snapped
  together. Handles imprecise PDF rendering.
- `join_tolerance=4`: line endpoints within 4 units are joined. Handles
  slightly misaligned grid lines.

```python
    try:
        tables = page.extract_tables(table_settings)
        return tables if tables else []
    except Exception:
        logger.warning("pdfplumber extraction failed, trying text strategy", exc_info=True)
        table_settings["vertical_strategy"] = "text"
        table_settings["horizontal_strategy"] = "text"
```
- **Fallback within pdfplumber**: if line-based detection fails, try text-based
  detection. Text strategy infers table structure from character alignment — it
  doesn't need ruled lines at all.
- `exc_info=True` logs the full traceback at WARNING level for debugging.

#### `TableExtractor._needs_vision_fallback`

```python
def _needs_vision_fallback(self, raw_table: list[list[str | None]]) -> bool:
    if not raw_table:
        return True
    total_cells = sum(len(row) for row in raw_table)
    if total_cells == 0:
        return True
    filled = sum(1 for row in raw_table for cell in row if cell and cell.strip())
    return (filled / total_cells) < self.min_cell_fill_ratio
```
- **Fill ratio heuristic**: count non-empty cells divided by total cells.
  Below 30% = the extraction is garbage.
- `cell and cell.strip()` handles both `None` cells and whitespace-only cells.
- This is the quality gate that decides whether to spend money on a Gemini
  API call.

#### `TableExtractor._gemini_vision_extract`

```python
def _gemini_vision_extract(self, pdf_path: Path, page_number: int) -> list[ExtractedTable] | None:
    if not settings.gemini_api_key:
        logger.warning("No GEMINI_API_KEY configured; skipping vision fallback")
        return None
```
- **Graceful degradation**: if no API key, log and return None. The caller
  handles this by skipping the table.

```python
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai not installed; skipping vision fallback")
        return None
```
- **Double-guarded lazy import**: checks both the API key and whether the
  library is installed. This means the module can be imported even if the
  google-generativeai package is missing (e.g., in a minimal test environment).

```python
    genai.configure(api_key=settings.gemini_api_key)
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number]
        pix = page.get_pixmap(dpi=_VISION_DPI)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()
```
- Renders the specific page as an image. `try/finally` ensures the PDF is
  closed.

```python
    prompt = (
        "Extract ALL tables from this research paper page as JSON.\n"
        "Return a JSON array where each element has:\n"
        '  - "headers": list of column header strings\n'
        '  - "rows": list of lists of cell value strings\n'
        "If no tables exist, return an empty array [].\n"
        "Return ONLY valid JSON, no markdown fences."
    )
```
- Structured prompt designed for Gemini's JSON mode. Key elements:
  - Explicit output format specification
  - "Return ONLY valid JSON" — prevents markdown wrapping
  - "If no tables exist, return []" — handles the empty case

```python
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(
        [prompt, img],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
```
- `response_mime_type="application/json"` enables Gemini's **constrained
  decoding** mode. The model is forced to output valid JSON, dramatically
  reducing parsing failures.
- `temperature=0.0` for deterministic output. Table extraction should be
  reproducible.
- `[prompt, img]` — multimodal input: text prompt + image.

```python
    if not isinstance(data, list):
        data = [data]
```
- Safety check: Gemini sometimes returns a single object instead of an array.
  This normalizes the response.

```python
    results.append(
        ExtractedTable(
            ...
            confidence=0.75,
        )
    )
```
- Vision-extracted tables get a fixed 0.75 confidence. Lower than pdfplumber's
  typical 0.9+ because vision models can hallucinate table content.

#### `TableExtractor._parse_raw_table`

```python
@staticmethod
def _parse_raw_table(raw_table: list[list[str | None]]) -> tuple[list[str], list[list[str]]]:
    headers = [cell.strip() if cell else "" for cell in raw_table[0]]
    rows = [
        [cell.strip() if cell else "" for cell in row]
        for row in raw_table[1:]
    ]
```
- First row = headers, everything else = data rows.
- `cell.strip() if cell else ""` handles `None` cells from pdfplumber by
  replacing them with empty strings.

#### `TableExtractor._find_caption`

```python
pattern = rf"(?i)table\s+{table_index + 1}[\s:.]\s*(.+)"
```
- Searches for "Table N" where N = table_index + 1 (tables are 1-indexed in
  papers but 0-indexed in code).
- `[\s:.]` after the number matches "Table 1:", "Table 1.", or "Table 1 ".
- `(?i)` makes it case-insensitive.
- Fallback: any line starting with "Table" longer than 6 characters.

#### `TableExtractor._get_page_texts`

```python
@staticmethod
def _get_page_texts(pdf_path: Path) -> dict[int, str]:
    texts: dict[int, str] = {}
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            texts[i] = doc[i].get_text("text") or ""
    finally:
        doc.close()
    return texts
```
- Quick single pass to grab text for caption matching. Uses PyMuPDF (not
  pdfplumber) because it's faster for text-only extraction.
- Returns a dict indexed by page number for O(1) lookup.

**Edge cases handled:**
- No tables on any page → returns `TableExtractionResult(tables=[], total_tables=0)`
- pdfplumber line strategy fails → falls back to text strategy
- pdfplumber text strategy also fails → returns empty list for that page
- Gemini API key missing → skips vision fallback, logs warning
- google-generativeai not installed → skips vision fallback
- Gemini returns invalid JSON → catches exception, returns None
- Gemini returns single object instead of array → wraps in list
- All cells are None → triggers vision fallback
- Caption not found → returns None (field is optional)

**Connections:**
- Imports `settings` from `config.py` (for Gemini API key and model)
- Returns `TableExtractionResult` (from `schemas.py`)
- Independent of `PDFParser` — reads the PDF directly
- Tests in `test_table_extractor.py`

---

### 3.8 `src/ingestion/layout_analyzer.py`

**What it does:** Takes the extracted page texts and segments them into logical
sections (Abstract, Introduction, Methodology, etc.). Also extracts individual
reference entries from the References section.

**Why it exists:** Raw page text is just a blob. Section detection adds
structure. Knowing that paragraph X is in the "Methodology" section enables
downstream features like section-specific search, automatic summarization, and
structured comparison between papers.

#### Module-level pattern registry

```python
_SECTION_PATTERNS: dict[SectionType, list[re.Pattern[str]]] = {
    SectionType.ABSTRACT: [
        re.compile(r"^\s*abstract\s*$", re.IGNORECASE),
    ],
    SectionType.INTRODUCTION: [
        re.compile(r"^\s*\d*\.?\s*introduction\s*$", re.IGNORECASE),
    ],
    ...
}
```
- **Pattern registry**: a dict mapping each `SectionType` to a list of regex
  patterns that match heading lines for that section.
- Patterns are pre-compiled at module load time (`re.compile`), not on each
  call. This avoids recompiling the same regex on every line of every page.
- Each pattern uses `^\s*\d*\.?\s*` to optionally match a section number prefix
  like "1." or "3.". This handles both numbered ("1. Introduction") and
  unnumbered ("Introduction") headings.
- `re.IGNORECASE` handles both "Abstract" and "ABSTRACT".
- Multiple patterns per type: "Related Work", "Literature Review", "Background",
  and "Prior Work" all map to `RELATED_WORK`. Academic papers are not consistent
  with naming.

```python
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+\.?\s+"        # Arabic: "1. " or "1 "
    r"|[IVXLC]+\.?\s+"  # Roman: "IV. " or "IV "
    r")"
    r"[A-Z]",           # Heading text starts with uppercase
)
```
- Matches any numbered heading, whether Arabic (1, 2, 3) or Roman (I, II, III).
- Requires the heading text to start with uppercase — "1. introduction" would
  NOT match, but "1. Introduction" or "IV. EXPERIMENTS" would.
- Used as a fallback: if a numbered heading doesn't match any known section, it
  gets classified as `UNKNOWN`.

```python
_REFERENCE_ENTRY_RE = re.compile(r"^\s*\[(\d+)\]\s+(.+)")
```
- Matches reference entries like `[1] Smith et al. ...`.
- Captures the number and the text separately.

#### `_SectionAccumulator` dataclass

```python
@dataclass
class _SectionAccumulator:
    section_type: SectionType
    title: str | None
    lines: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int | None = None
```
- **Internal mutable container** used during the section-building pass. Not
  exposed outside the module.
- `field(default_factory=list)` avoids the mutable default gotcha with
  dataclasses (same reason as Pydantic's `default_factory`).
- `page_start` and `page_end` track which pages this section spans.
- Leading underscore marks it as private to the module.

#### `LayoutAnalyzer.analyze`

```python
def analyze(self, pages: list[PageResult]) -> LayoutAnalysisResult:
    if not pages:
        return LayoutAnalysisResult(sections=[], references=[])
    sections = self._build_sections(pages)
    references = self._extract_references(sections)
    return LayoutAnalysisResult(sections=sections, references=references)
```
- Two-phase analysis:
  1. `_build_sections`: walk through text line-by-line, detect headings, split
     into sections.
  2. `_extract_references`: from the detected References section, pull out
     individual reference entries.
- Empty pages → empty result. No exceptions.

#### `LayoutAnalyzer._build_sections`

```python
def _build_sections(self, pages: list[PageResult]) -> list[DetectedSection]:
    accumulators: list[_SectionAccumulator] = []
    current = _SectionAccumulator(
        section_type=SectionType.TITLE,
        title=None,
        page_start=pages[0].page_number,
    )
```
- Starts with a TITLE section accumulator. Everything before the first detected
  heading is assumed to be the title/header area of the paper.

```python
    for page in pages:
        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped:
                current.lines.append("")
                continue

            detected = self._classify_section(stripped)
            if detected is not None and self._detect_headers(stripped):
                current.page_end = page.page_number
                accumulators.append(current)
                current = _SectionAccumulator(
                    section_type=detected,
                    title=stripped,
                    page_start=page.page_number,
                )
            else:
                current.lines.append(line)
```
- **Linear scan** through all lines of all pages.
- For each non-blank line, two checks must BOTH pass:
  1. `_classify_section` returns a SectionType (not None)
  2. `_detect_headers` returns True
- **Both checks are needed** because `_classify_section` might match body text
  that happens to contain the word "Abstract", while `_detect_headers` verifies
  it looks like a heading (short, uppercase, or numbered).
- When a heading is detected: close the current section (set its page_end,
  append to accumulators), start a new section.
- When it's not a heading: append the line to the current section's content.
- Empty lines are preserved (for paragraph separation in the final output).

```python
        current.page_end = page.page_number
    accumulators.append(current)
```
- After the loop: `page_end` is updated for every page (so the current section
  always knows its last page), and the final section is appended.

```python
    detected: list[DetectedSection] = []
    for idx, acc in enumerate(accumulators):
        text = "\n".join(acc.lines).strip()
        if not text and acc.section_type == SectionType.TITLE and not acc.title:
            continue
```
- Converts accumulators to `DetectedSection` objects.
- Skips the initial TITLE accumulator if it's completely empty (can happen if
  the paper starts with "Abstract" on line 1).

#### `LayoutAnalyzer._detect_headers`

```python
@staticmethod
def _detect_headers(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
```
- Empty lines and very long lines (>120 chars) are never headings. Real section
  headings are short.

```python
    if _NUMBERED_HEADING_RE.match(stripped):
        return True
```
- Numbered headings ("1. Introduction", "III. Methods") are always headings.

```python
    alpha_chars = [c for c in stripped if c.isalpha()]
    if alpha_chars and all(c.isupper() for c in alpha_chars) and len(alpha_chars) >= 3:
        return True
```
- ALL-CAPS detection. Extracts only alphabetic characters, checks if they're all
  uppercase, and requires at least 3 to avoid matching things like "A" or "OK".

```python
    words = stripped.split()
    if 1 <= len(words) <= 6:
        for patterns in _SECTION_PATTERNS.values():
            for pat in patterns:
                if pat.match(stripped):
                    return True
```
- Short lines (1-6 words) that match a known section pattern are headings.
  The word count filter prevents matching "introduction" inside a sentence like
  "This paper provides an introduction to...".

#### `LayoutAnalyzer._classify_section`

```python
@staticmethod
def _classify_section(line: str) -> SectionType | None:
    stripped = line.strip()
    for section_type, patterns in _SECTION_PATTERNS.items():
        for pat in patterns:
            if pat.match(stripped):
                return section_type
    if _NUMBERED_HEADING_RE.match(stripped):
        return SectionType.UNKNOWN
    return None
```
- Tries all patterns in order. First match wins.
- If nothing matches but it's a numbered heading → UNKNOWN.
- If nothing matches and it's not numbered → None (not a heading).

#### `LayoutAnalyzer._extract_references`

```python
@staticmethod
def _extract_references(sections: list[DetectedSection]) -> list[str]:
    references: list[str] = []
    ref_sections = [s for s in sections if s.section_type == SectionType.REFERENCES]
    if not ref_sections:
        return references
    ref_text = ref_sections[0].text
```
- Finds the References section. Uses the first one found (a paper shouldn't
  have multiple).

```python
    current_ref: list[str] = []
    for line in ref_text.split("\n"):
        match = _REFERENCE_ENTRY_RE.match(line)
        if match:
            if current_ref:
                references.append(" ".join(current_ref).strip())
            current_ref = [line.strip()]
        elif current_ref and line.strip():
            current_ref.append(line.strip())
    if current_ref:
        references.append(" ".join(current_ref).strip())
```
- **Accumulator pattern** (same as section building): when a new `[N]` marker
  is found, flush the previous reference and start a new one.
- Multi-line references are joined with spaces. This handles references that
  wrap across lines.
- `if current_ref` before appending: don't flush an empty accumulator.
- After the loop: flush the last reference.

**Edge cases handled:**
- Empty pages → empty result
- No headings detected → entire text becomes a single TITLE section
- Unknown numbered headings → classified as UNKNOWN
- No References section → empty references list
- Multi-line references → joined correctly
- ALL-CAPS headings → detected
- Roman numeral headings → detected

**Connections:**
- Consumes `list[PageResult]` from `PDFParser`
- Returns `LayoutAnalysisResult` (from `schemas.py`)
- Tests in `test_layout_analyzer.py`

---

### 3.9 `src/ingestion/metadata_extractor.py`

**What it does:** Extracts bibliographic metadata (title, authors, abstract,
DOI, keywords) from the first few pages of a paper. Uses two strategies:
fast heuristic regex and optional LLM extraction via Gemini.

**Why it exists:** Metadata is needed for search, citation, and cataloguing.
The dual-strategy approach provides fast free extraction (heuristics) with
an optional accuracy boost (LLM).

#### Module-level constants

```python
_LLM_CONTEXT_CHARS = 4000
_HEURISTIC_PAGES = 3
```
- `4000` characters ≈ 1 page of text. Enough for title, authors, abstract, and
  keywords, which are always on the first 1-2 pages of a paper.
- `3` pages for heuristics — broader scan for edge cases where the abstract
  spills onto page 2.

#### `MetadataExtractor.extract`

```python
def extract(self, pages: list[PageResult], *, use_llm: bool = True) -> DocumentMetadataSchema:
    if not pages:
        return DocumentMetadataSchema()
    heuristic = self._heuristic_extract(pages)
    if use_llm and settings.gemini_api_key:
        llm = self._llm_extract(pages)
        return self._merge(heuristic, llm)
    return heuristic
```
- `use_llm` is a **keyword-only argument** (the `*` before it forces callers to
  write `use_llm=True`, not just `True`). This prevents accidental positional
  errors.
- Three execution paths:
  1. `use_llm=True` and API key set → heuristic + LLM, merged
  2. `use_llm=True` but no API key → heuristic only
  3. `use_llm=False` → heuristic only (for tests)

#### `MetadataExtractor._heuristic_extract`

```python
def _heuristic_extract(self, pages: list[PageResult]) -> DocumentMetadataSchema:
    text = "\n".join(p.text for p in pages[:_HEURISTIC_PAGES])
    lines = [l.strip() for l in text.split("\n") if l.strip()]
```
- Joins the first 3 pages into one text block for regex searching.
- `lines` is the same text but split into non-empty lines (for title/author
  extraction which works line-by-line).

#### `MetadataExtractor._extract_title`

```python
@staticmethod
def _extract_title(lines: list[str]) -> str | None:
    skip_patterns = re.compile(
        r"(?i)^(proceedings|journal|volume|issue|copyright|arxiv|preprint|doi\s*:)"
    )
    for line in lines[:15]:
        if skip_patterns.match(line):
            continue
        if len(line) < 10:
            continue
        if 10 <= len(line) <= 300:
            return line
    return None
```
- **Boilerplate filter**: skips lines starting with journal names, proceedings
  headers, copyright notices, etc. These often appear before the title in PDFs.
- Scans the first 15 non-empty lines.
- Length filter: titles are typically 10-300 characters. Too short = page
  numbers or section labels. Too long = body text.
- Returns the FIRST line that passes all filters. In most papers, this is the
  title.

#### `MetadataExtractor._extract_authors`

```python
@staticmethod
def _extract_authors(lines: list[str], title: str | None) -> list[str]:
    if not title:
        return []
```
- Can't find authors without knowing where the title is (authors come right
  after).

```python
    title_idx = None
    for i, line in enumerate(lines):
        if line == title:
            title_idx = i
            break
```
- Finds the title's position in the line list.

```python
    for line in lines[title_idx + 1: title_idx + 5]:
        if re.search(r"@|university|department|institute|lab\b", line, re.IGNORECASE):
            continue
        if re.match(r"^\d+\s|^abstract|^keyword", line, re.IGNORECASE):
            break
```
- Looks at the 4 lines after the title.
- **Skips affiliations**: lines containing emails (@), university, department,
  institute, or lab are likely institutional affiliations, not author names.
- **Stops at structural markers**: if we hit a numbered section, "abstract", or
  "keyword", we've gone past the author block.

```python
        parts = re.split(r"\s*[,;]\s*|\s+and\s+", line)
        for part in parts:
            part = part.strip()
            if part and re.match(r"^[A-Z][a-z]+([\s-][A-Z][a-z]+)*$", part):
                authors.append(part)
```
- Splits on commas, semicolons, or " and ".
- Name validation regex: `^[A-Z][a-z]+([\s-][A-Z][a-z]+)*$` matches patterns
  like "Alice Smith", "Jean-Pierre Martin", "Bob Jones". Each word must start
  with uppercase followed by lowercase.
- This filters out non-name fragments like superscript numbers ("1,2") or
  partial affiliations.

#### `MetadataExtractor._extract_abstract`

```python
match = re.search(
    r"(?i)\babstract\b[:\s\-]*\n?(.*?)(?=\n\s*\n\s*(?:\d+\.?\s+)?[A-Z][a-z]|\n\s*(?:keywords?|introduction)\b)",
    text,
    re.DOTALL,
)
```
- **Complex regex** with three parts:
  1. `\babstract\b[:\s\-]*\n?` — matches "Abstract", "ABSTRACT:", "Abstract -",
     etc., followed by an optional newline.
  2. `(.*?)` — non-greedy capture of the abstract body. `re.DOTALL` makes `.`
     match newlines too.
  3. Lookahead `(?=...)` — stops capturing when it sees:
     - A blank line followed by a capitalized line (probably next section heading)
     - A "Keywords" or "Introduction" line

```python
    if len(abstract) > 50:
        return abstract
```
- Minimum length sanity check. An abstract shorter than 50 characters is
  probably a false match.

#### `MetadataExtractor._extract_doi`

```python
match = re.search(r"(?i)\b(?:doi\s*:?\s*)(10\.\d{4,}/[^\s]+)", text)
```
- DOI format: always starts with `10.` followed by a registrant code (4+
  digits), a slash, and a suffix.
- Two patterns tried: with "DOI:" prefix first, then bare DOI.
- `.rstrip(".")` removes trailing periods that are often part of the sentence
  but not the DOI.

#### `MetadataExtractor._extract_keywords`

```python
match = re.search(r"(?i)\bkeywords?\s*[:\-]\s*(.+?)(?:\n\n|\n\s*\d)", text, re.DOTALL)
```
- Matches "Keywords:", "Keyword:", "Keywords -", etc.
- Stops at a blank line or a numbered section heading.

```python
parts = re.split(r"\s*[;,•·]\s*", raw)
```
- Splits on commas, semicolons, and bullet characters (•·). Academic papers use
  all of these as keyword separators.

#### `MetadataExtractor._llm_extract`

```python
def _llm_extract(self, pages: list[PageResult]) -> DocumentMetadataSchema:
    context = "\n".join(p.text for p in pages[:_HEURISTIC_PAGES])[:_LLM_CONTEXT_CHARS]
```
- Truncates to 4000 characters to control API costs and latency.

```python
    prompt = (
        "You are a metadata extraction system for academic research papers.\n"
        ...
        "Return ONLY valid JSON. If a field is not found, use null for strings "
        "and empty arrays for lists.\n\n"
        f"---\n{context}\n---"
    )
```
- System role + explicit output format + null handling instructions.
- The `---` fences clearly delineate the paper text from the instructions.

```python
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    data = json.loads(response.text)
```
- JSON mode + zero temperature. Same rationale as table extraction.
- `json.loads` is wrapped in the outer `try/except` to handle malformed
  responses.

```python
    return DocumentMetadataSchema(
        ...
        confidence=0.85,
    )
```
- LLM extraction gets 0.85 confidence (higher than heuristic's 0.5).

#### `MetadataExtractor._merge`

```python
@staticmethod
def _merge(heuristic: DocumentMetadataSchema, llm: DocumentMetadataSchema) -> DocumentMetadataSchema:
    return DocumentMetadataSchema(
        title=llm.title or heuristic.title,
        authors=llm.authors if llm.authors else heuristic.authors,
        ...
        confidence=max(heuristic.confidence, llm.confidence),
    )
```
- **LLM-preferred merge**: for each field, use the LLM value if it's non-empty,
  otherwise fall back to heuristic.
- `llm.title or heuristic.title` — Python's `or` returns the first truthy
  value. If `llm.title` is None or empty string, `heuristic.title` is used.
- `llm.authors if llm.authors else heuristic.authors` — uses explicit
  conditional instead of `or` because an empty list `[]` is falsy in Python,
  and we want to prefer LLM only when it actually found authors.
- `confidence=max(...)` — the merged result is at least as confident as the
  best individual source.

**Edge cases handled:**
- Empty pages → empty metadata
- No API key → heuristic only
- LLM returns empty fields → heuristic values used as fallback
- LLM fails entirely → returns empty schema (merged with heuristic, so
  heuristic values survive)
- DOI with trailing period → stripped
- Boilerplate before title → skipped
- Abstract too short → returned as None
- No keywords section → empty list

**Connections:**
- Imports `settings` from `config.py`
- Consumes `list[PageResult]` from `PDFParser`
- Returns `DocumentMetadataSchema` (from `schemas.py`)
- Tests in `test_metadata_extractor.py`

---

### 3.10 `src/main.py`

**What it does:** FastAPI application with three endpoints: health check,
document upload, and document processing.

**Why it exists:** This is the HTTP interface to the ingestion pipeline. It ties
all four processors together into a single synchronous workflow.

**Line-by-line:**

```python
app = FastAPI(
    title="Intelligent Document Processing",
    description="AI-powered research paper analysis platform",
    version="0.1.0",
)
```
- These fields populate the auto-generated OpenAPI/Swagger docs at `/docs`.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- CORS middleware allows any origin. In production, you'd restrict this to your
  frontend domain. Wide-open CORS is acceptable for Phase 1 development.

```python
@app.on_event("startup")
async def startup() -> None:
    settings.ensure_dirs()
```
- Creates upload and chroma directories on first boot. Runs once when uvicorn
  starts the application.

#### `upload_document`

```python
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
```
- `UploadFile` is FastAPI's file upload handler. It streams large files to a
  temporary file on disk instead of loading everything into memory.
- `response_model=DocumentUploadResponse` tells FastAPI to validate and
  serialize the response using this Pydantic model. It also generates the
  correct OpenAPI schema.

```python
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
```
- Input validation: reject non-PDF files immediately.
- `file.filename` can be None if the client doesn't send a filename header.

```python
    content = await file.read()
    file_size = len(content)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(status_code=413, ...)
```
- **Size limit enforcement**. HTTP 413 = "Payload Too Large", the semantically
  correct status code.
- `await file.read()` loads the entire file into memory. For Phase 1 this is
  fine (50MB limit). In production, you'd stream to disk in chunks.

```python
    file_hash = hashlib.sha256(content).hexdigest()
    doc_id = uuid.uuid4()
```
- SHA-256 hash for deduplication. Same file → same hash → unique constraint in
  the DB would prevent re-upload.
- UUID generated client-side (in the API handler), not by the database.

```python
    save_path = upload_dir / f"{doc_id}.pdf"
    save_path.write_bytes(content)
```
- Files are named by their UUID, not the original filename. This prevents path
  traversal attacks (a filename like `../../etc/passwd.pdf`) and filename
  collisions.

#### `process_document`

```python
@app.post("/documents/{doc_id}/process")
async def process_document(doc_id: str) -> dict:
```
- Path parameter `doc_id` is a string (not UUID) for simplicity.
- Returns a raw dict — in Phase 2, this would return a proper Pydantic model.

```python
    parser = PDFParser()
    extraction: PDFExtractionResult = parser.extract(pdf_path)

    table_extractor = TableExtractor()
    tables: TableExtractionResult = table_extractor.extract_tables(pdf_path)

    layout_analyzer = LayoutAnalyzer()
    layout: LayoutAnalysisResult = layout_analyzer.analyze(extraction.pages)

    meta_extractor = MetadataExtractor()
    metadata: DocumentMetadataSchema = meta_extractor.extract(
        extraction.pages, use_llm=bool(settings.gemini_api_key)
    )
```
- **Sequential pipeline**. Four processors, one after another.
- `PDFParser` runs first because `LayoutAnalyzer` and `MetadataExtractor` need
  its output.
- `TableExtractor` is independent — it reads the PDF directly. Could run in
  parallel with `PDFParser` in Phase 2.
- `use_llm=bool(settings.gemini_api_key)` — only use LLM if key is configured.
  Empty string → `bool("")` → `False`.

```python
    return {
        "document_id": doc_id,
        "status": "completed",
        "total_pages": extraction.total_pages,
        "avg_confidence": extraction.avg_confidence,
        "sections_found": len(layout.sections),
        "tables_found": tables.total_tables,
        "references_found": len(layout.references),
        "metadata": metadata.model_dump(),
    }
```
- `model_dump()` converts the Pydantic model to a dict for JSON serialization.

**Edge cases handled:**
- Non-PDF upload → 400
- Oversized file → 413
- Missing document → 404
- Missing API key → LLM features gracefully disabled

**Connections:**
- Imports all four ingestion modules
- Imports schemas for response types
- Imports settings for configuration
- This is the single entry point that ties everything together

---

### 3.11 `Dockerfile`

**What it does:** Builds a Docker image with Python 3.11, Tesseract OCR, and
all dependencies.

**Why it exists:** Reproducible deployments. Every developer and every CI
environment runs the same OS, same Python version, same system libraries.

```dockerfile
FROM python:3.11-slim AS base
```
- `python:3.11-slim` is a Debian-based image with only the essential packages.
  Roughly 150MB vs 900MB for the full image.
- `AS base` names this build stage for potential multi-stage builds later.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```
- `tesseract-ocr` + `tesseract-ocr-eng`: OCR engine and English language pack.
- `poppler-utils`: PDF rendering utilities (used by pdfplumber internally).
- `libgl1` + `libglib2.0-0`: OpenGL and GLib libraries required by OpenCV/PIL
  for image processing.
- `--no-install-recommends` prevents pulling in unnecessary suggested packages.
- `rm -rf /var/lib/apt/lists/*` cleans the apt cache to shrink the image.
- Single `RUN` command with `&&` chains — each `RUN` creates a Docker layer.
  Combining them into one layer means the apt cache cleanup actually reduces the
  image size.

```dockerfile
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .
```
- **Two-stage copy** for layer caching. `pyproject.toml` is copied first and
  dependencies are installed. Only then is the source code copied. This means
  if you change a `.py` file but not `pyproject.toml`, Docker uses the cached
  dependency layer and only re-copies the code. Saves minutes on rebuilds.

```dockerfile
RUN mkdir -p /app/data/uploads /app/data/chroma
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Data directories created at build time.
- `--host 0.0.0.0` binds to all interfaces so the container is reachable from
  outside.

---

### 3.12 `docker-compose.yml`

**What it does:** Defines the complete development environment: app server,
Celery worker, PostgreSQL, and Redis.

**Why it exists:** `docker compose up` should start the entire system with zero
manual setup.

```yaml
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      - DATABASE_URL=postgresql+asyncpg://docuser:docpass@postgres:5432/intelligent_doc
      - REDIS_URL=redis://redis:6379/0
```
- `env_file: .env` loads the `.env` file.
- The `environment` block overrides DATABASE_URL and REDIS_URL with
  Docker-internal hostnames (`postgres`, `redis`) instead of `localhost`.
  Docker Compose creates a network where services are reachable by their
  service name.

```yaml
    volumes:
      - upload_data:/app/data/uploads
      - chroma_data:/app/data/chroma
```
- **Named volumes** persist data across container restarts. Without these,
  uploaded PDFs and ChromaDB data would be lost every time you restart.

```yaml
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
```
- `service_healthy` means the app waits until Postgres and Redis are not just
  started but actually ready to accept connections. This prevents the app from
  crashing on startup because the database isn't ready yet.

```yaml
  postgres:
    image: postgres:16-alpine
    ...
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U docuser -d intelligent_doc"]
      interval: 5s
      timeout: 5s
      retries: 5
```
- `pg_isready` is Postgres's built-in readiness check.
- 5 retries × 5s interval = waits up to 25 seconds for Postgres to start.

```yaml
  celery-worker:
    command: celery -A src.worker worker --loglevel=info --concurrency=2
```
- Points to `src.worker` which doesn't exist yet (Phase 2). The worker is
  defined now so the infrastructure is ready when the Celery tasks are
  implemented.
- `--concurrency=2` = two worker processes. Low for development, would be
  higher in production.

```yaml
volumes:
  pg_data:
  redis_data:
  upload_data:
  chroma_data:
```
- Four named volumes at the bottom. Docker manages their lifecycle. They persist
  even if you `docker compose down` (but not `docker compose down -v`).

---

### 3.13 `tests/conftest.py`

**What it does:** Defines shared pytest fixtures used across all test files.

**Why it exists:** DRY principle. Three fixtures are used by multiple test
modules. Defining them in `conftest.py` makes them automatically available to
all tests in the `tests/` directory and below.

#### `tmp_pdf` fixture

```python
@pytest.fixture()
def tmp_pdf(tmp_path: Path) -> Path:
    import fitz
    doc = fitz.open()
    page0 = doc.new_page(width=612, height=792)
    page0.insert_text((72, 80), "Deep Learning for Document Understanding", fontsize=16)
    ...
```
- Creates a **real 2-page PDF** in a temporary directory.
- `tmp_path` is a built-in pytest fixture that provides a unique temporary
  directory for each test.
- Page dimensions 612×792 = US Letter size in PDF points (72 points/inch).
- Known text is inserted at specific positions: title, authors, abstract,
  keywords, DOI, introduction, references.
- This fixture is used by `test_pdf_parser.py` and `test_table_extractor.py`.
- The PDF is **real** — not mocked. Tests that use this fixture exercise the
  actual PyMuPDF and pdfplumber code paths.

#### `sample_pages` fixture

```python
@pytest.fixture()
def sample_pages() -> list[PageResult]:
    return [
        PageResult(page_number=0, text="...", ...),
        PageResult(page_number=1, text="...", ...),
        PageResult(page_number=2, text="...", ...),
    ]
```
- Pre-built `PageResult` objects with known content: a 3-page paper with
  abstract, introduction, methodology, experiments, results, conclusion, and
  references.
- Used by `test_layout_analyzer.py` and `test_metadata_extractor.py`. These
  tests don't need a real PDF — they operate on already-extracted text.
- **Design decision**: decouple layout/metadata tests from the PDF parser.
  If the PDF parser has a bug, it shouldn't break layout analyzer tests.

#### `mock_fitz_page` fixture

```python
@pytest.fixture()
def mock_fitz_page() -> MagicMock:
    page = MagicMock()
    page.get_text.side_effect = lambda fmt="text", **kw: (
        {"blocks": [...]} if fmt == "dict" else "Sample text..."
    )
```
- A `MagicMock` that behaves like a `fitz.Page` for the two `get_text` call
  signatures the parser uses.
- `side_effect` with a lambda: returns dict structure when called with
  `fmt="dict"` and plain text when called with `fmt="text"`.
- `page.rect.width = 612` — mock the page dimensions for confidence scoring.

---

### 3.14 `tests/unit/test_pdf_parser.py`

**What it does:** 14 tests covering PDFParser's extraction, OCR fallback, text
cleaning, and error handling.

**Why these specific tests exist:**

| Test | What it verifies |
|------|------------------|
| `test_extract_returns_all_pages` | Parser processes every page, not just the first |
| `test_extract_contains_expected_text` | Text extraction actually works (integration) |
| `test_confidence_is_valid_range` | Confidence stays in [0, 1] — catches math bugs |
| `test_native_extraction_method` | Known-text PDF uses native, not OCR |
| `test_char_count_matches_text_length` | Internal consistency check |
| `test_file_not_found_raises` | Error handling: missing file |
| `test_invalid_pdf_raises_runtime_error` | Error handling: corrupt file |
| `test_clean_text_collapses_blank_lines` | Text normalization: blank lines |
| `test_clean_text_strips_trailing_whitespace` | Text normalization: trailing spaces |
| `test_clean_text_empty_input` | Edge case: empty string |
| `test_needs_ocr_with_rich_text` | OCR decision: rich text → no OCR |
| `test_needs_ocr_with_sparse_text` | OCR decision: sparse text → OCR |
| `test_ocr_fallback_called` | OCR actually triggers when needed |
| `test_avg_confidence_calculation` | Math correctness |

**Notable testing pattern:**

```python
@patch("pytesseract.image_to_string", return_value="OCR extracted text")
def test_ocr_fallback_called(self, mock_ocr: MagicMock, tmp_path: Path) -> None:
```
- Patches `pytesseract.image_to_string` directly (not through the module path)
  because `pdf_parser.py` uses a lazy `import pytesseract` inside the method.
  Lazy imports mean the module attribute doesn't exist until the method runs,
  so `@patch("src.ingestion.pdf_parser.pytesseract")` would fail with
  `AttributeError`.

---

### 3.15 `tests/unit/test_table_extractor.py`

**What it does:** 17 tests covering table parsing, quality scoring, caption
finding, vision fallback gating, and end-to-end extraction.

**Notable tests:**

| Test | What it verifies |
|------|------------------|
| `test_needs_vision_fallback_*` (4 tests) | Quality gate at various fill ratios |
| `test_parse_raw_table_with_none_cells` | None → empty string conversion |
| `test_score_table_partial` | Score = 0.5 for half-filled table |
| `test_find_caption_with_matching_table` | Regex finds "Table 1: ..." |
| `test_find_caption_fallback` | Falls back to any "Table" line |
| `test_gemini_vision_skipped_without_api_key` | No API key → None |
| `test_extract_tables_with_table_pdf` | Real PDF with drawn grid borders |

**Testing pattern for settings:**

```python
@patch("src.ingestion.table_extractor.settings")
def test_gemini_vision_skipped_without_api_key(self, mock_settings, ...):
    mock_settings.gemini_api_key = ""
```
- Patches the `settings` object at the module level. Unlike the pytesseract
  case, `settings` is imported at the top of the module, so it exists as a
  module attribute and can be patched with the standard path.

---

### 3.16 `tests/unit/test_layout_analyzer.py`

**What it does:** 26 tests — the most comprehensive test file. Covers section
detection, header heuristics, classification for all 12 section types, reference
extraction, ordering, and page ranges.

**Why so many tests?** The layout analyzer is entirely heuristic-based. Unlike
the PDF parser (which delegates to a PDF library) or the metadata extractor
(which delegates to an LLM), every line of logic in the layout analyzer is our
code. More custom logic = more tests needed.

**Notable patterns:**

```python
def test_analyze_detects_sections(self, sample_pages):
    section_types = {s.section_type for s in result.sections}
    assert SectionType.ABSTRACT in section_types
```
- Uses a **set comprehension** for membership testing. Cleaner than looping and
  asserting.

```python
def test_classify_methodology(self) -> None:
    assert LayoutAnalyzer._classify_section("3. Methodology") == SectionType.METHODOLOGY
    assert LayoutAnalyzer._classify_section("Methods") == SectionType.METHODOLOGY
    assert LayoutAnalyzer._classify_section("Approach") == SectionType.METHODOLOGY
```
- Tests multiple heading variations per section type because academic papers use
  inconsistent naming.

---

### 3.17 `tests/unit/test_metadata_extractor.py`

**What it does:** 21 tests covering heuristic extraction, merge logic, and
individual regex helpers.

**Notable patterns:**

```python
def test_merge_prefers_llm_title(self) -> None:
    heuristic = DocumentMetadataSchema(title="Heuristic Title", confidence=0.5)
    llm = DocumentMetadataSchema(title="LLM Title", confidence=0.85)
    result = MetadataExtractor._merge(heuristic, llm)
    assert result.title == "LLM Title"
```
- Tests merge logic with **synthetic data** — no need for actual extraction.
  This isolates the merge logic from extraction bugs.

```python
def test_extract_title_skips_boilerplate(self) -> None:
    lines = [
        "Proceedings of ICML 2024",
        "Volume 42, Issue 3",
        "A Novel Method for Text Classification",
        "John Doe, Jane Smith",
    ]
    title = MetadataExtractor._extract_title(lines)
    assert title == "A Novel Method for Text Classification"
```
- Directly tests the static helper method. No need to construct PageResult
  objects or run the full pipeline.

---

## 4. Cross-Cutting Design Decisions

### Why `from __future__ import annotations` everywhere?

Every source file starts with this import. It enables two things:
1. **PEP 604 union syntax**: `str | None` instead of `Optional[str]`
2. **Forward references**: class A can reference class B before B is defined
3. **Performance**: annotations are stored as strings, not evaluated at import time

### Why static methods for helper functions?

Methods like `_clean_text`, `_parse_raw_table`, `_extract_doi` are all
`@staticmethod`. They don't access `self` or class state. Benefits:
- **Testability**: can be called directly in tests without instantiating the
  class
- **Documentation**: signals "this is a pure function with no side effects"
- **Minor performance**: no `self` argument overhead

### Why confidence scores everywhere?

Every extraction result carries a confidence float in [0, 1]. This enables:
- Downstream filtering: "only use text with confidence > 0.8"
- Quality monitoring: track avg confidence over time
- User feedback: show low-confidence sections with a warning
- Hybrid decisions: weight search results by confidence

### Why lazy imports for optional dependencies?

`pytesseract` and `google.generativeai` are imported inside methods, not at the
top of the file. This means:
- The app starts even if Tesseract isn't installed (as long as OCR isn't
  triggered)
- Tests run without a Gemini API key
- Import errors are caught at the point of use with a helpful log message

### Why `try/finally` for PDF handles?

`fitz.open()` returns a document handle that must be closed. `try/finally`
guarantees cleanup even if processing crashes mid-page. In a long-running
server, leaked file handles eventually cause "too many open files" errors.

---

## 5. Edge Cases and Error Handling

### PDFParser

| Edge Case | What Happens |
|-----------|--------------|
| File doesn't exist | `FileNotFoundError` with clear message |
| File exists but isn't a PDF | `RuntimeError("Failed to open PDF: ...")` |
| PDF has zero pages | Returns `PDFExtractionResult(pages=[], total_pages=0, avg_confidence=0.0)` |
| Page has no text at all | `_needs_ocr` returns True → OCR fallback |
| Page has tiny artefact text | `_needs_ocr` detects low density → OCR |
| OCR produces garbage | Still returned, but with low confidence score |
| Page has zero area | `_compute_confidence` returns 0.0 (no division by zero) |
| Text has control characters | `_clean_text` strips them |
| 50+ consecutive blank lines | Collapsed to one blank line |

### TableExtractor

| Edge Case | What Happens |
|-----------|--------------|
| PDF has no tables | Returns `TableExtractionResult(tables=[], total_tables=0)` |
| pdfplumber line strategy fails | Falls back to text strategy |
| Both pdfplumber strategies fail | Returns empty list for that page |
| Table has all None cells | Triggers vision fallback |
| Table has sparse data (<30% filled) | Triggers vision fallback |
| No Gemini API key | Skips vision, logs warning, returns None |
| google-generativeai not installed | Skips vision, logs warning, returns None |
| Gemini returns invalid JSON | Catches exception, returns None |
| Gemini returns single object (not array) | Wraps in list |
| No caption found | `caption` field is None |

### LayoutAnalyzer

| Edge Case | What Happens |
|-----------|--------------|
| Empty page list | Returns `LayoutAnalysisResult(sections=[], references=[])` |
| No section headings detected | Entire text = one TITLE section |
| Unknown numbered heading | Classified as `SectionType.UNKNOWN` |
| Heading word inside body text | Rejected by `_detect_headers` (too long, not uppercase) |
| ALL-CAPS heading like "ABSTRACT" | Detected by uppercase check |
| Roman numeral heading "IV. METHODS" | Detected by `_NUMBERED_HEADING_RE` |
| No References section | `references` list is empty |
| Multi-line reference entry | Joined with spaces |
| Paper starts with Abstract (no title preamble) | Empty TITLE section is skipped |

### MetadataExtractor

| Edge Case | What Happens |
|-----------|--------------|
| Empty page list | Returns empty `DocumentMetadataSchema` |
| No API key configured | Heuristic-only mode |
| `use_llm=False` | Heuristic-only mode |
| Boilerplate before title | Skipped by `skip_patterns` regex |
| Abstract shorter than 50 chars | Returned as None |
| No DOI in paper | `doi` is None |
| No keywords section | `keywords` is empty list |
| LLM returns empty fields | Heuristic values used as fallback |
| LLM call fails entirely | Returns empty schema; merge preserves heuristic |
| DOI has trailing period | Stripped by `.rstrip(".")` |

---

## 6. Interview Questions and Answers

### Architecture & Design

**Q: Why did you separate the ingestion pipeline into four independent modules instead of one big class?**

A: Single Responsibility Principle. Each module does one thing and can be tested,
replaced, and scaled independently. If we need to swap pdfplumber for Camelot,
we only touch `table_extractor.py`. If OCR performance is a bottleneck, we can
scale just the PDF parser. It also means four developers can work on four
modules in parallel without merge conflicts.

---

**Q: Why use pydantic-settings for configuration instead of just reading environment variables?**

A: Type safety and fail-fast behavior. With `os.getenv`, a typo like
`CHUNKSIZE` (missing underscore) silently returns None, and you don't discover
it until runtime. Pydantic-settings validates all fields at startup. If
`CHUNK_SIZE` is set to `"abc"`, you get a clear validation error immediately,
not a `ValueError` deep in the chunking code at 3am. It also provides
centralized documentation — one file lists every configurable parameter with its
type and default.

---

**Q: Why did you choose UUIDs for primary keys instead of auto-incrementing integers?**

A: Three reasons. (1) Security: integers are sequential, so knowing document
ID 42 exists lets you guess 41 and 43. UUIDs are unguessable. (2) Distributed
generation: UUIDs can be generated client-side without a database round-trip.
This matters when uploads go through a load balancer to multiple app servers.
(3) Merge safety: if we ever need to merge data from multiple database
instances, UUIDs won't collide.

---

**Q: Why do you have both SQLAlchemy models AND Pydantic schemas? Isn't that duplication?**

A: They serve different purposes. SQLAlchemy models define storage — what goes
in the database, with indexes, constraints, and relationships. Pydantic schemas
define the API contract — what gets sent over HTTP, with validation rules. They
can evolve independently. For example, the database might add a `processing_log`
column that's never exposed to the API. Or the API might add a computed
`processing_time` field that doesn't exist in the database. Coupling them
together means every database migration forces an API change and vice versa.

---

**Q: How would you make the processing pipeline asynchronous?**

A: The code is already designed for it. In Phase 2, `process_document` becomes a
Celery task. The `POST /documents/{doc_id}/process` endpoint would
`task.delay(doc_id)` and immediately return a 202 Accepted. The Celery worker
(already defined in `docker-compose.yml`) runs the four processors. A
`GET /documents/{doc_id}` endpoint returns the current status. We'd also add a
webhook or WebSocket for completion notifications. The key insight is that
`PDFParser`, `TableExtractor`, `LayoutAnalyzer`, and `MetadataExtractor` are
already pure functions with no HTTP dependencies — they take a path or a list of
pages and return a result. They don't know or care whether they're running in a
web request or a Celery worker.

---

### PDF Parser

**Q: How does your code decide whether to use native extraction or OCR?**

A: The `_needs_ocr` method uses a two-tier heuristic. First, it checks the
absolute character count — if there are fewer than 20 characters on the page,
it's definitely image-based. Second, it looks at the PyMuPDF block structure.
It counts the total characters across all text spans and divides by the number
of text blocks. If the average characters per block is below 30 (the threshold
0.3 × 100), the page is likely a scan with artefact text rather than real
content. This catches pages that have a few stray characters from image
compression but no actual readable text.

---

**Q: Why import pytesseract lazily inside `_extract_ocr_page` instead of at the top of the file?**

A: Two reasons. (1) Startup speed: if a PDF has only native text, pytesseract
is never imported, and the Tesseract binary is never invoked. (2) Fault
tolerance: if Tesseract isn't installed on the system, the app still starts and
works fine for native PDFs. The ImportError only surfaces if OCR is actually
needed. This is especially important for development — you don't want to force
every developer to install Tesseract just to run the app.

---

**Q: What's the `from exc` in `raise RuntimeError(...) from exc`?**

A: Exception chaining. It sets the new exception's `__cause__` attribute to the
original exception. When the traceback is printed, you see both: "RuntimeError:
Failed to open PDF: ..." followed by "The above exception was the direct cause
of the following exception: ...". This preserves debugging context. Without
`from exc`, the original traceback is lost.

---

**Q: Why does `_clean_text` use `[^\S \n\t]+` instead of just removing specific characters?**

A: The character class `[^\S \n\t]+` is a double-negative that reads as "match
any whitespace character that is NOT a space, newline, or tab." This catches all
non-standard whitespace: form feeds (`\f`), vertical tabs (`\v`), zero-width
spaces, non-breaking spaces, and Unicode whitespace variants. Rather than
maintaining a list of specific characters to remove (which is fragile and
incomplete), the regex defines what to KEEP (space, newline, tab) and replaces
everything else. This is more robust against unusual PDF encodings.

---

### Table Extractor

**Q: Why use pdfplumber for tables instead of Camelot or Tabula?**

A: pdfplumber has the best balance of accuracy and simplicity for ruled tables
in academic papers. Camelot is more accurate for some complex layouts but
requires GhostScript and has a heavier dependency tree. Tabula requires Java.
pdfplumber is pure Python, integrates well with our stack, and handles the
most common case — bordered result tables — very well. For the cases it can't
handle, we fall back to Gemini Vision, which handles almost anything.

---

**Q: How does the quality gate for vision fallback work?**

A: `_needs_vision_fallback` computes the fill ratio: non-empty cells divided by
total cells. If less than 30% of cells have content, the pdfplumber extraction
is considered too noisy. This threshold was chosen empirically — a well-parsed
table typically has >80% fill ratio. Below 30% usually means pdfplumber
detected a "table" where there isn't one, or misaligned the grid so badly that
cells are empty or duplicated. The threshold is configurable via the constructor
for tuning.

---

**Q: Why render the page as an image for Gemini Vision instead of sending the PDF?**

A: Gemini's vision models accept images, not PDFs. Even if we could send a PDF,
rendering a specific page as a PNG gives us control over the resolution and
ensures the model sees exactly what a human would see. We use 200 DPI — lower
than the 300 DPI for OCR — because we're not doing character recognition; we
just need the model to see the table structure, and lower DPI means a smaller
image and faster API calls.

---

### Layout Analyzer

**Q: Why do you require BOTH `_classify_section` and `_detect_headers` to pass?**

A: Defence in depth. `_classify_section` checks if the text MATCHES a known
section pattern (e.g., contains the word "Abstract"). `_detect_headers` checks
if the line LOOKS like a heading (short, uppercase, or numbered). Requiring both
prevents false positives like a body sentence that says "We follow the approach
of the introduction section" — it matches the pattern "introduction" but
`_detect_headers` rejects it because it's a long sentence, not a heading. The
two checks are complementary: classification catches the WHAT, header detection
catches the HOW.

---

**Q: How does `_build_sections` handle page boundaries?**

A: It processes all pages sequentially in a single pass. The accumulator tracks
`page_start` (set when a new section begins) and `page_end` (updated at the end
of every page and when a new heading is detected). So a section that starts on
page 2 and continues to page 4 will have `page_start=2, page_end=4`. The key
insight is that sections DON'T align with page boundaries — a heading on page 3
starts a new section mid-page, and the accumulator handles this naturally because
it processes lines, not pages.

---

**Q: How do you handle papers with non-standard section names?**

A: Multiple patterns per SectionType. "Methodology" matches "Methodology",
"Methods", "Method", "Approach", "Proposed Method", "Proposed Approach",
"Proposed System", and "Model". Each pattern is a separate regex in the list.
For headings that don't match any known pattern, we check if it's a numbered
heading (like "7. Limitations") and classify it as UNKNOWN. The UNKNOWN type
preserves the section structure even when we can't name the section.

---

### Metadata Extractor

**Q: Why two extraction strategies (heuristic + LLM) instead of just using the LLM?**

A: (1) Cost: heuristics are free; every LLM call costs money. For 10,000 papers,
that adds up. (2) Speed: regex runs in microseconds; an LLM API call takes 1-3
seconds. (3) Availability: if the Gemini API is down or the quota is exhausted,
heuristics still work. (4) Testability: heuristics are deterministic and testable
without network access. The LLM is an accuracy boost on top of a solid baseline,
not a crutch.

---

**Q: How does the merge strategy work, and why prefer the LLM?**

A: For each field, the merge checks: if the LLM returned a non-empty value, use
it; otherwise, fall back to the heuristic value. LLM is preferred because it
understands context — it can identify a title even if it doesn't match any
heuristic pattern, and it can correctly parse author names in non-Western formats.
The confidence score is the max of both sources, reflecting that the merged
result is at least as good as the best individual source. Importantly, the
heuristic always runs first, so even if the LLM fails completely (returns all
empty fields), the heuristic results survive the merge.

---

**Q: Why `_LLM_CONTEXT_CHARS = 4000`?**

A: Metadata (title, authors, abstract, keywords, DOI) is always in the first
1-2 pages of a research paper, which is roughly 2000-4000 characters. Sending
more text wastes tokens and money without improving accuracy. Sending less risks
truncating a long abstract. 4000 characters is the sweet spot — it fits
comfortably within Gemini's context window, costs minimal tokens, and captures
all metadata fields for 99%+ of papers.

---

### Testing

**Q: Why create a real PDF in the test fixture instead of just mocking fitz?**

A: Real PDFs test the actual code path including PyMuPDF's text extraction and
pdfplumber's table detection. Mocking fitz would only test our wrapper logic,
not whether our code works with actual PDF data. The `tmp_pdf` fixture generates
a known PDF with known text, so we can assert on exact content while still
exercising the real library code. The PDF is tiny (two pages, text only) so it
adds negligible time to the test suite.

---

**Q: Why do layout analyzer and metadata extractor tests use `sample_pages` instead of `tmp_pdf`?**

A: Decoupling. If the PDF parser has a bug in text extraction, it shouldn't
cause layout analyzer tests to fail. `sample_pages` provides pre-built
`PageResult` objects with known text, isolating the layout analyzer and metadata
extractor from PDF parsing concerns. This follows the unit testing principle of
testing one thing at a time. The integration between PDF parsing and layout
analysis would be covered by integration tests (Phase 2).

---

**Q: How did you handle testing the OCR path without Tesseract installed?**

A: `@patch("pytesseract.image_to_string", return_value="OCR extracted text")`.
This patches the pytesseract function at the library level (not the module
level) because our code uses a lazy import. The test creates a PDF with a drawn
rectangle (no text layer), which forces `_needs_ocr` to return True. The mock
then intercepts the Tesseract call and returns a known string. We verify that
the mock was called, confirming the OCR path was triggered.

---

### Docker & Infrastructure

**Q: Why `python:3.11-slim` instead of Alpine?**

A: Alpine uses musl libc instead of glibc. Many Python packages with C
extensions (PyMuPDF, Pillow, psycopg2) require glibc and either fail to install
or need to be compiled from source on Alpine, which is slow and fragile. The
slim Debian image adds ~50MB over Alpine but saves hours of debugging broken
builds.

---

**Q: Why does docker-compose.yml use `condition: service_healthy` instead of just `depends_on`?**

A: Plain `depends_on` only waits for the container to START, not for the service
inside it to be READY. PostgreSQL can take several seconds to initialize its
data directory on first boot. Without healthchecks, the app would start, try to
connect to Postgres, and crash. `service_healthy` waits until `pg_isready`
returns success, guaranteeing the database is actually accepting connections.

---

**Q: Why separate named volumes for uploads and chroma data?**

A: Lifecycle management. Upload data and vector store data have different
retention needs. You might want to `docker volume rm chroma_data` to rebuild
the vector index without losing uploaded PDFs, or vice versa. Named volumes
also survive `docker compose down` (unlike anonymous volumes), so your data
persists across development sessions.
