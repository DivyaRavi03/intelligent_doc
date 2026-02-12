# Phase 5 Walkthrough: API Layer, Async Processing & Real-Time Updates

Phase 5 transforms the processing pipeline built in Phases 1-4 into a complete, production-shaped REST API. Everything a user can do — upload a paper, ask a question, compare findings, watch processing happen in real time — goes through the endpoints defined here. This walkthrough explains every file, every class, every endpoint, and every design decision line by line.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Request Lifecycle Diagram](#2-request-lifecycle-diagram)
3. [Full API Design](#3-full-api-design)
   - [Why RESTful](#why-restful)
   - [Why Versioned (/api/v1/)](#why-versioned-apiv1)
   - [Why These Specific Endpoints](#why-these-specific-endpoints)
   - [Complete Endpoint Map](#complete-endpoint-map)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
   - [src/main.py — Application Entry Point](#srcmainpy--application-entry-point)
   - [src/api/stores.py — In-Memory Data Layer](#srcapistorespy--in-memory-data-layer)
   - [src/api/auth.py — API Key Authentication](#srcapiauthpy--api-key-authentication)
   - [src/api/rate_limiter.py — Rate Limiting](#srcapirate_limiterpy--rate-limiting)
   - [src/api/routes_documents.py — Document CRUD](#srcapiroutes_documentspy--document-crud)
   - [src/api/routes_query.py — Query and Search](#srcapiroutes_querypy--query-and-search)
   - [src/api/routes_extract.py — Extraction and Feedback](#srcapiroutes_extractpy--extraction-and-feedback)
   - [src/api/routes_admin.py — Admin and Monitoring](#srcapiroutes_adminpy--admin-and-monitoring)
   - [src/api/websocket.py — Real-Time Processing Updates](#srcapiwebsocketpy--real-time-processing-updates)
   - [src/workers/celery_app.py — Celery Configuration](#srcworkerscelery_apppy--celery-configuration)
   - [src/workers/tasks.py — Processing Pipeline Task](#srcworkerstaskspy--processing-pipeline-task)
   - [src/models/schemas.py — Phase 5 Schemas](#srcmodelsschemaspyphase-5-additions)
5. [How the Async Celery Pipeline Works](#5-how-the-async-celery-pipeline-works)
6. [How Authentication and Rate Limiting Work Together](#6-how-authentication-and-rate-limiting-work-together)
7. [How the WebSocket Streams Real-Time Status](#7-how-the-websocket-streams-real-time-status)
8. [How the Celery Task Orchestrates All Phases](#8-how-the-celery-task-orchestrates-all-phases)
9. [Why Swagger Docs Matter for Portfolio Presentation](#9-why-swagger-docs-matter-for-portfolio-presentation)
10. [How Phase 5 Ties Together All Previous Phases](#10-how-phase-5-ties-together-all-previous-phases)
11. [Testing Deep Dive](#11-testing-deep-dive)
12. [Interview Questions and Ideal Answers](#12-interview-questions-and-ideal-answers)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                        │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ CORS     │  │ Rate     │  │ Auth     │  │ Lifespan │           │
│  │ Middle-  │  │ Limiter  │  │ (X-API-  │  │ Manager  │           │
│  │ ware     │  │ (slowapi)│  │  Key)    │  │          │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       API Routers                            │   │
│  │                                                              │   │
│  │  /api/v1/documents/*    /api/v1/query     /api/v1/search    │   │
│  │  /api/v1/compare        /api/v1/feedback  /api/v1/admin/*   │   │
│  │  /ws/processing/{id}    /health                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    In-Memory Stores (MVP)                    │   │
│  │                                                              │   │
│  │  DocumentStore         APIKeyStore         Metrics           │   │
│  │  (thread-safe)         (pre-populated)     (thread-safe)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Celery Workers                              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  _process_document_impl()  — 9-stage pipeline               │  │
│  │                                                              │  │
│  │  Phase 1: PDFParser → TableExtractor → LayoutAnalyzer →     │  │
│  │           MetadataExtractor                                  │  │
│  │  Phase 2: SectionAwareChunker                                │  │
│  │  Phase 3: EmbeddingService → VectorStore                     │  │
│  │  Phase 4: PaperExtractor (structured extraction)             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│                    Redis (broker + results)                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Key architectural principles:**

- **Separation of concerns** — Each router handles one domain (documents, queries, extraction, admin). Authentication and rate limiting are cross-cutting middleware, not embedded in business logic.
- **Dependency injection everywhere** — Stores, auth, and metrics flow through FastAPI's `Depends()`. Tests swap implementations with `app.dependency_overrides` instead of monkey-patching.
- **Graceful degradation** — If Redis is down, uploads still work (synchronous fallback). If no API key is configured, embedding and extraction stages are skipped with warnings instead of crashes.
- **Testable by design** — The core pipeline is a plain function (`_process_document_impl`), not a Celery task. Celery is a thin wrapper. Every endpoint is testable with `httpx.AsyncClient` and no external infrastructure.

---

## 2. Request Lifecycle Diagram

What happens when a request hits the server:

```
Client Request
     │
     ▼
┌─────────────┐
│ CORS Check   │  ← Adds Access-Control headers
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Rate Limiter │  ← slowapi checks IP against limits
│ (slowapi)    │    Returns 429 if exceeded
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Route Match  │  ← FastAPI matches URL + method to handler
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Dependency   │  ← Resolves Depends() chain:
│ Injection    │    verify_api_key → get_document_store → etc.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Request      │  ← Pydantic validates body against schema
│ Validation   │    Returns 422 if invalid
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Handler      │  ← Your endpoint function runs
│ Function     │    Business logic + service calls
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Response     │  ← Pydantic validates response against model
│ Serialization│    Serializes to JSON
└──────┬──────┘
       │
       ▼
  JSON Response
```

---

## 3. Full API Design

### Why RESTful

REST is the right choice for this API because:

1. **Resource-oriented domain.** The system manages documents — discrete resources with clear CRUD operations. REST maps directly: `POST /documents/upload` creates, `GET /documents/{id}` reads, `DELETE /documents/{id}` deletes.

2. **Statelessness.** Each request carries everything needed to process it (the API key in the header, the query in the body). No server-side sessions to manage. This makes horizontal scaling trivial — any server instance can handle any request.

3. **HTTP semantics for free.** Status codes (201 Created, 204 No Content, 404 Not Found, 413 Payload Too Large, 429 Too Many Requests) communicate outcomes without custom error schemas. Clients already know what 404 means.

4. **Swagger/OpenAPI generation.** FastAPI auto-generates interactive API docs from the route definitions. REST's predictable structure makes these docs immediately usable. GraphQL would need a separate playground; gRPC would need proto files.

5. **Portfolio accessibility.** Hiring managers and interviewers can test the API from a browser tab at `/docs`. No tooling installation required.

### Why Versioned (/api/v1/)

```python
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
```

The `/api/v1/` prefix exists for one reason: **non-breaking evolution**.

When the document store moves from in-memory to PostgreSQL, the response shape for `GET /documents/{id}` might change (new fields, different nesting). With versioning:
- Existing clients keep calling `/api/v1/` and get the old shape.
- New clients call `/api/v2/` and get the new shape.
- The old version gets deprecated on a timeline, not yanked immediately.

Without versioning, every schema change is a breaking change for every client simultaneously. In a production system with external consumers, this is how you cause outages.

The version is in the URL path (not a header or query param) because:
- It is visible in logs, making debugging easier.
- It is enforceable at the load balancer / API gateway level.
- It is the most common convention (GitHub, Stripe, Twilio all use path versioning).

### Why These Specific Endpoints

Each endpoint exists because a concrete user story requires it:

| Endpoint | User Story |
|----------|-----------|
| `POST /documents/upload` | "I have a research paper PDF. I want to upload it for analysis." |
| `GET /documents/{id}` | "I want to see everything we know about a paper — metadata, sections, tables." |
| `GET /documents/{id}/status` | "My upload is processing. Is it done yet?" |
| `GET /documents/` | "Show me all my papers with pagination." |
| `DELETE /documents/{id}` | "I uploaded the wrong file. Remove it completely." |
| `GET /documents/{id}/sections` | "Show me just the sections (for a table of contents view)." |
| `GET /documents/{id}/tables` | "Show me just the extracted tables (for data analysis)." |
| `POST /query` | "What does this paper say about attention mechanisms?" |
| `POST /search` | "Find all passages about transformer architectures." |
| `POST /compare` | "How do these two papers differ in their methodology?" |
| `GET /documents/{id}/summary/{level}` | "Give me a one-line / abstract / detailed summary." |
| `POST /feedback` | "The extracted author list is wrong. Here is the correction." |
| `GET /documents/{id}/extraction` | "Show me the structured extraction results." |
| `POST /documents/{id}/re-extract` | "Re-run extraction after I gave feedback." |
| `GET /feedback/stats` | "How many corrections have been submitted? Which papers need attention?" |
| `GET /admin/health` | "Is the service up? Are Redis and ChromaDB reachable?" |
| `GET /admin/metrics` | "How many documents have been processed? How many queries served?" |
| `GET /admin/costs` | "How much has LLM usage cost?" |
| `GET /admin/eval/latest` | "What are the latest evaluation scores?" |
| `WS /ws/processing/{task_id}` | "I want to watch my upload process in real time." |

Endpoints that were deliberately **not** built:
- `PATCH /documents/{id}` — Documents are processed outputs, not user-editable resources.
- `POST /documents/batch-upload` — Premature. Single upload with async processing is sufficient.
- `GET /query/history` — Would require session state. Not needed for the MVP.

### Complete Endpoint Map

```
Method  Path                                      Auth    Rate Limit   Status
──────  ────────────────────────────────────────  ──────  ───────────  ──────
POST    /api/v1/documents/upload                  Yes     10/hour      201
GET     /api/v1/documents/{id}                    Yes     —            200
GET     /api/v1/documents/{id}/status             Yes     —            200
GET     /api/v1/documents/                        Yes     —            200
DELETE  /api/v1/documents/{id}                    Yes     —            204
GET     /api/v1/documents/{id}/sections           Yes     —            200
GET     /api/v1/documents/{id}/tables             Yes     —            200

POST    /api/v1/query                             Yes     100/hour     200
POST    /api/v1/search                            Yes     100/hour     200
POST    /api/v1/compare                           Yes     100/hour     200
GET     /api/v1/documents/{id}/summary/{level}    Yes     —            200

POST    /api/v1/feedback                          Yes     —            200
GET     /api/v1/documents/{id}/extraction         Yes     —            200
POST    /api/v1/documents/{id}/re-extract         Yes     —            200
GET     /api/v1/feedback/stats                    Yes     —            200

GET     /api/v1/admin/health                      No*     —            200
GET     /api/v1/admin/metrics                     Yes     —            200
GET     /api/v1/admin/costs                       Yes     —            200
GET     /api/v1/admin/eval/latest                 Yes     —            200

GET     /health                                   No      —            200
WS      /ws/processing/{task_id}                  No      —            —

* Health check uses optional_api_key — works with or without a key.
```

---

## 4. File-by-File Deep Dive

### `src/main.py` — Application Entry Point

This file is the single place where the entire application is assembled. Every component registers itself here.

```python
"""FastAPI entry point for the Intelligent Document Processing platform.

Registers all API routers, configures middleware, rate limiting, and
provides a WebSocket endpoint for real-time processing updates.
"""
```

**Lines 7-27 — Imports and the lifespan context manager:**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
```

`asynccontextmanager` replaces the deprecated `@app.on_event("startup")` / `@app.on_event("shutdown")` pattern. FastAPI's recommended approach since version 0.109.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    settings.ensure_dirs()
    logger.info("Upload dir: %s", settings.upload_dir)
    yield
```

**Why a lifespan manager instead of startup/shutdown events:**
- `@app.on_event` is deprecated and will be removed.
- The lifespan pattern is a single function, not two disconnected handlers. Setup code runs before `yield`, teardown after. If you needed to close a database pool on shutdown, the `finally` block after `yield` is the natural place.
- It makes the startup/shutdown relationship explicit — they share a scope.

`settings.ensure_dirs()` creates the upload directory if it does not exist. This runs once at boot, not per-request.

**Lines 37-55 — Application construction and middleware:**

```python
app = FastAPI(
    title="Intelligent Document Processing",
    description="AI-powered research paper analysis platform",
    version="0.5.0",
    lifespan=lifespan,
)
```

The `title`, `description`, and `version` fields populate the auto-generated Swagger UI at `/docs`. They are not cosmetic — they are how an interviewer or hiring manager understands what this system does within 5 seconds of opening the page.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

CORS is set to permissive (`*`) for development. In production, `allow_origins` would be locked to the frontend domain. This middleware runs on every request before routing — it adds `Access-Control-Allow-Origin` headers that let browsers make cross-origin requests.

```python
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

Two things happen here:
1. The limiter is attached to `app.state` so slowapi can find it during request processing.
2. The exception handler converts `RateLimitExceeded` exceptions (thrown when a limit is hit) into proper 429 HTTP responses. Without this handler, rate limit violations would become unhandled 500 errors.

**Lines 57-61 — Router registration:**

```python
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(extract_router)
app.include_router(admin_router)
```

Each `include_router` call mounts all of a router's endpoints into the app. The routers themselves define their prefixes (`/api/v1/documents`, `/api/v1`, `/api/v1/admin`), so `main.py` does not need to know about URL structure. This is separation of concerns — `main.py` assembles, routers define.

**Lines 64-75 — WebSocket and root health check:**

```python
@app.websocket("/ws/processing/{task_id}")
async def ws_processing(websocket, task_id: str):
    await processing_websocket(websocket, task_id)
```

WebSocket routes cannot be put in an `APIRouter` (FastAPI limitation for WebSocket routes with path parameters in some versions). The thin wrapper delegates to `websocket.py` to keep `main.py` minimal.

```python
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()
```

The root `/health` endpoint is a minimal liveness probe for load balancers and Kubernetes. It returns `{"status": "ok"}` with no authentication. It is separate from `/api/v1/admin/health`, which checks Redis and ChromaDB status.

---

### `src/api/stores.py` — In-Memory Data Layer

This is the persistence layer for the MVP. It stores documents, API keys, and metrics in Python dictionaries. The design makes it trivially swappable with PostgreSQL later.

**Lines 30-49 — DocumentRecord dataclass:**

```python
@dataclass
class DocumentRecord:
    id: uuid.UUID
    filename: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    num_pages: int | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    task_id: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: DocumentMetadataSchema | None = None
    sections: list[DetectedSection] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    extraction: PaperExtraction | None = None
```

**Why a `@dataclass` instead of a Pydantic model:**
- `DocumentRecord` is an internal data container, not an API schema. It does not need JSON serialization or validation.
- Dataclasses are lighter and faster than Pydantic models for internal use.
- The fields deliberately mirror the `Document` ORM model in `src/models/database.py`. When the store is swapped to SQLAlchemy, the shape stays the same.

**Why `file_hash` is stored:**
- Enables deduplication. If a user uploads the same PDF twice, the hash matches and (in a future iteration) we can skip reprocessing.

**Why `task_id` lives on the record:**
- After `process_document.delay()` returns a Celery task ID, we need somewhere to store it. The client polls `/status` and gets this ID back, which they can use to connect to the WebSocket.

**Lines 52-60 — APIKeyRecord:**

```python
@dataclass
class APIKeyRecord:
    key: str
    name: str
    rate_limit_uploads: int = 10
    rate_limit_queries: int = 100
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

Per-key rate limits are stored but not yet enforced (slowapi uses IP-based limiting). The structure is ready for per-key limiting in production.

**Lines 67-121 — InMemoryDocumentStore:**

```python
class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, DocumentRecord] = {}
        self._lock = threading.Lock()
```

**Why `threading.Lock`:**
- FastAPI runs in an async event loop, but Celery workers run in threads. When the upload endpoint falls back to synchronous processing, `_process_document_impl` runs in the same process and writes to the store. Without a lock, concurrent saves could corrupt the dict.
- The lock is on writes only (`save`, `delete`, `update_status`). Reads (`get`) do not acquire the lock because Python's GIL makes dict reads atomic.

```python
def list_all(
    self, offset: int = 0, limit: int = 20,
    status_filter: DocumentStatus | None = None,
) -> tuple[list[DocumentRecord], int]:
    docs = sorted(self._store.values(), key=lambda d: d.created_at, reverse=True)
    if status_filter is not None:
        docs = [d for d in docs if d.status == status_filter]
    total = len(docs)
    return docs[offset : offset + limit], total
```

**Why return `(docs, total)` as a tuple:**
- The client needs both the page of results and the total count (for pagination UI). Returning them together avoids a second round-trip.
- `total` is computed **after** filtering but **before** slicing. This is correct — if there are 50 completed documents and you ask for page 2 of 10, you need to know the total is 50, not 10.

**Lines 124-147 — InMemoryAPIKeyStore:**

```python
class InMemoryAPIKeyStore:
    def __init__(self) -> None:
        self._keys: dict[str, APIKeyRecord] = {
            "test-api-key-12345": APIKeyRecord(
                key="test-api-key-12345",
                name="Test Key",
            ),
        }
```

A pre-populated test key. This means `curl -H "X-API-Key: test-api-key-12345" localhost:8000/api/v1/documents/` works immediately after starting the server, with zero configuration. This is critical for demos and interviews.

**Lines 154-189 — Singletons and FastAPI dependencies:**

```python
_doc_store = InMemoryDocumentStore()
_key_store = InMemoryAPIKeyStore()
_metrics: dict[str, Any] = {
    "total_documents": 0,
    "total_queries": 0,
    "total_llm_tokens": 0,
    "total_cost_usd": 0.0,
    "total_processing_time": 0.0,
}
```

Module-level singletons. Created once at import time, shared across all requests.

```python
def get_document_store() -> InMemoryDocumentStore:
    return _doc_store

def get_api_key_store() -> InMemoryAPIKeyStore:
    return _key_store
```

**Why wrap singletons in functions:**
- `Depends(get_document_store)` is overridable. In tests: `app.dependency_overrides[get_document_store] = lambda: mock_store`. If endpoints used `_doc_store` directly, there would be no injection point and tests would need `@patch` on every test.
- This is FastAPI's recommended pattern for swappable dependencies.

```python
def update_metrics(key: str, value: float) -> None:
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0.0) + value
```

Thread-safe metric increment. Used by both the API endpoints (query count) and the processing pipeline (processing time).

---

### `src/api/auth.py` — API Key Authentication

```python
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
```

`APIKeyHeader` is a FastAPI security scheme that:
1. Reads the `X-API-Key` header from every request.
2. Appears in Swagger UI as an "Authorize" button — users can enter their key and all subsequent requests include it.
3. `auto_error=False` means **we** control the error response (not FastAPI's generic 403). This lets us return 401 with a custom message.

```python
async def verify_api_key(
    api_key: str | None = Security(api_key_header),
    key_store: InMemoryAPIKeyStore = Depends(get_api_key_store),
) -> str:
```

**Why `Security()` instead of `Depends()`:**
- `Security()` is a specialized `Depends()` that integrates with OpenAPI security schemes. It tells Swagger that this endpoint requires authentication, which adds the padlock icon in the docs.

**Two-stage validation:**

```python
if not api_key:
    raise HTTPException(status_code=401, detail="Missing X-API-Key header",
                        headers={"WWW-Authenticate": "ApiKey"})

if not key_store.validate(api_key):
    raise HTTPException(status_code=401, detail="Invalid API key",
                        headers={"WWW-Authenticate": "ApiKey"})
```

- First check: Is the header present at all? Empty string and `None` are both "missing."
- Second check: Is the key in our store?
- Both return 401, not 403. The RFC distinction: 401 means "you haven't authenticated." 403 means "you authenticated but don't have permission." Since we are validating identity (the key itself), 401 is correct.
- The `WWW-Authenticate: ApiKey` header tells the client which authentication scheme to use. This is an HTTP standard requirement for 401 responses.

```python
async def optional_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    return api_key
```

Used by the health check endpoint. It extracts the key if present but never rejects the request. This lets authenticated users see their key in the response context while keeping health checks public.

---

### `src/api/rate_limiter.py` — Rate Limiting

```python
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/hour"],
    storage_uri="memory://",
)
```

- `key_func=get_remote_address` — Rate limits are per IP address. Every unique IP gets its own bucket.
- `default_limits=["200/hour"]` — Catch-all for endpoints without explicit limits.
- `storage_uri="memory://"` — Counters stored in process memory. In production, this would be `redis://` for shared state across multiple workers.

**Why in-memory storage for MVP:**
- No Redis required to run the app locally.
- Counters reset on restart, which is fine for development.
- The switch to Redis is a one-line change: `storage_uri=settings.redis_url`.

```python
UPLOAD_LIMIT = "10/hour"
QUERY_LIMIT = "100/hour"
ADMIN_LIMIT = "20/minute"
```

**Why these specific limits:**
- **Uploads (10/hour):** Each upload triggers heavy processing (PDF parsing, embedding generation, LLM extraction). 10/hour prevents a single user from overwhelming the processing queue.
- **Queries (100/hour):** Each query involves an LLM call. At ~$0.01 per query, 100/hour limits cost exposure to ~$1/hour/user.
- **Admin (20/minute):** Admin endpoints are lightweight reads but expose internal metrics. Rate limiting prevents scraping.

```python
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    retry_after = str(getattr(exc, "retry_after", 60) or 60)
    return Response(
        content=f'{{"detail":"Rate limit exceeded: {exc.detail}"}}',
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": retry_after},
    )
```

**Why a custom handler:**
- slowapi's default handler returns plain text. Our API consistently returns JSON. A custom handler ensures the error format matches every other error response.
- `Retry-After` is an HTTP standard header that tells the client how many seconds to wait before retrying. Well-behaved clients (and browser fetch APIs) use this automatically.

---

### `src/api/routes_documents.py` — Document CRUD

**POST /upload — The most complex endpoint:**

```python
@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
@limiter.limit(UPLOAD_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> DocumentUploadResponse:
```

**Why `status_code=201`:**
- HTTP 201 means "Created." The upload creates a new resource. Returning 200 would be technically incorrect — 200 means "your request succeeded but nothing new was created."

**Why `Request` is a parameter:**
- slowapi's `@limiter.limit()` decorator needs the raw `Request` object to extract the client IP. Without it, rate limiting silently fails.

**Validation chain (lines 60-70):**

```python
if not file.filename or not file.filename.lower().endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Only PDF files are supported")

content = await file.read()
file_size = len(content)
max_bytes = settings.max_upload_size_mb * 1024 * 1024
if file_size > max_bytes:
    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")
```

- **400 for wrong file type** — Client error. They sent something that is not a PDF.
- **413 for oversized file** — HTTP 413 "Payload Too Large" is the correct status for this case. It is distinct from 400 because the file *type* is correct, it is just too big.
- File size is checked **after** `await file.read()`. This is intentional — `UploadFile` does not expose size before reading. In production, you would add a content-length check middleware to reject oversized requests before reading the body.

**File storage and hashing (lines 72-78):**

```python
file_hash = hashlib.sha256(content).hexdigest()
doc_id = uuid.uuid4()
save_path = upload_dir / f"{doc_id}.pdf"
save_path.write_bytes(content)
```

- UUID4 for the document ID, not an auto-incrementing integer. UUIDs are unguessable (no IDOR vulnerability) and globally unique (no collision risk across distributed systems).
- SHA-256 hash enables future deduplication without exposing the file content.
- The file is saved as `{uuid}.pdf`, not the original filename. This prevents path traversal attacks (a filename like `../../etc/passwd.pdf` would be harmless).

**Celery with synchronous fallback (lines 88-108):**

```python
task_id: str | None = None
try:
    from src.workers.tasks import process_document
    if process_document is not None:
        result = process_document.delay(str(doc_id))
        task_id = result.id
    else:
        raise RuntimeError("Celery task not available")
except Exception:
    logger.info("Celery unavailable, processing synchronously: %s", doc_id)
    from src.workers.tasks import _process_document_impl
    doc.status = DocumentStatus.PROCESSING
    store.save(doc)
    try:
        _process_document_impl(str(doc_id), save_path)
    except Exception as exc:
        logger.warning("Sync processing failed for %s: %s", doc_id, exc)
```

This is the most important design pattern in Phase 5. The try/except creates a **graceful degradation path**:

1. **Happy path:** Celery + Redis are running. `process_document.delay()` enqueues the task. The upload returns immediately with a task ID. Processing happens asynchronously.
2. **Fallback path:** Redis is not running, or Celery is not installed. The exception is caught, and processing runs synchronously in the request handler. The upload still succeeds — it is just slower.

**Why this matters:**
- Developers can run `uvicorn src.main:app` and upload files immediately. No Redis, no Celery worker, no Docker Compose.
- Tests never need Celery infrastructure.
- In production, the Celery path gives you horizontal scaling — add more workers for more throughput.

**DELETE /{doc_id} — Three-layer cleanup:**

```python
file_path = Path(doc.file_path)
if file_path.exists():
    file_path.unlink()

try:
    from src.retrieval.vector_store import VectorStore
    VectorStore().delete_paper(str(doc_id))
except Exception as exc:
    logger.warning("Vector store cleanup failed for %s: %s", doc_id, exc)

store.delete(doc_id)
```

Delete touches three storage layers: disk (the PDF), ChromaDB (the vectors), and the document store (the record). Vector store deletion is best-effort — if ChromaDB is down, the document and file are still deleted. A warning is logged, and a periodic cleanup job could handle orphaned vectors later.

---

### `src/api/routes_query.py` — Query and Search

**Lazy service initialization — The pattern that makes everything testable:**

```python
_qa_engine = None
_retriever = None
_summarizer = None
_client = None

def _get_client():
    global _client
    if _client is None:
        from src.llm.gemini_client import GeminiClient
        _client = GeminiClient()
    return _client
```

**Why lazy initialization instead of creating services at import time:**

1. **Import-time safety.** `GeminiClient()` reads `settings.gemini_api_key`. If the key is not set, the constructor might fail. Lazy init defers this to first use, when the user has had a chance to configure the key.
2. **Test isolation.** Tests patch `_get_qa_engine` to return a mock. If the real engine was created at import time, the patch would need to undo real initialization.
3. **Startup speed.** Loading embedding models, connecting to ChromaDB, and building BM25 indexes all take time. Lazy init means the server starts instantly; the first query pays the initialization cost.

**Why module-level globals instead of FastAPI dependencies:**
- These services are stateful singletons (they hold loaded models and connections). A `Depends()` function would create a new instance per request, which would be wasteful and slow.
- The `_get_*` functions are easily patchable with `unittest.mock.patch`, which is how the tests work.

**POST /query — Citation-tracked QA:**

```python
response = qa_engine.answer(
    query=body.query,
    paper_ids=body.paper_ids,
    top_k=body.top_k,
)
```

One line in the endpoint, but behind it is the full Phase 3+4 pipeline: hybrid retrieval (BM25 + dense vectors + RRF fusion), passage ranking, LLM answer generation with citations, and faithfulness verification. The endpoint is just the HTTP shell around `QAEngine.answer()`.

**POST /search — Hybrid retrieval exposed directly:**

```python
results = retriever.retrieve(
    query=body.query, top_k=body.top_k, alpha=body.alpha,
    paper_id=body.paper_id, section_type=body.section_type,
)
```

The `alpha` parameter lets the client tune the dense/sparse balance:
- `alpha=1.0` — Pure dense (semantic) search. Good for conceptual queries.
- `alpha=0.0` — Pure BM25 (keyword) search. Good for exact term matching.
- `alpha=0.5` — Default hybrid. Best for most queries.

This is exposed as a user-tunable parameter because different queries benefit from different strategies. A search for "BERT" wants keyword matching; a search for "how do transformers handle long documents" wants semantic matching.

**POST /compare — Multi-paper comparison:**

```python
paper_summaries: dict[str, str] = {}
for paper_id in body.paper_ids:
    try:
        doc_uuid = uuid.UUID(paper_id)
        doc = store.get(doc_uuid)
        if doc and doc.sections:
            paper_summaries[paper_id] = "\n".join(s.text[:500] for s in doc.sections[:3])
```

Each paper is represented by its first 3 sections, truncated to 500 chars each. This keeps the LLM prompt under token limits while giving enough context for meaningful comparison. The prompt is constructed dynamically and sent to `GeminiClient.generate()`.

Key differences are extracted by parsing bullet points from the LLM response:

```python
key_differences = [
    line.strip("- ").strip()
    for line in comparison_text.split("\n")
    if line.strip().startswith("-") and len(line.strip()) > 3
][:5]
```

This is a heuristic — it assumes the LLM outputs differences as markdown bullet points. The `[:5]` caps the list at 5 items. Not all LLM outputs will follow this format, but it works well enough for the MVP.

---

### `src/api/routes_extract.py` — Extraction and Feedback

This router evolved from Phase 4's simple feedback endpoint into a full extraction management module.

**POST /feedback — Now with authentication:**

The original Phase 4 feedback endpoint had no auth. Phase 5 adds `api_key: str = Depends(verify_api_key)` as a parameter. This single change means:
- Anonymous users cannot submit corrections.
- Feedback is attributable (the key identifies who submitted it).
- The endpoint still appears in Swagger with the padlock icon.

**GET /feedback/stats — Aggregation with Counter:**

```python
by_field: dict[str, int] = dict(Counter(f["field_name"] for f in _FEEDBACK_STORE))

paper_field_counts: dict[tuple[str, str], int] = Counter(
    (f["paper_id"], f["field_name"]) for f in _FEEDBACK_STORE
)
flagged = list(
    {pf[0] for pf, count in paper_field_counts.items() if count >= _FEEDBACK_THRESHOLD}
)
```

`Counter` from `collections` does the aggregation. `flagged_papers` uses a set comprehension to deduplicate — a paper with 5+ corrections on *any* field is flagged once, not once per field.

---

### `src/api/routes_admin.py` — Admin and Monitoring

**GET /health — Dependency health checks:**

```python
async def health_check(
    api_key: str | None = Depends(optional_api_key),
) -> HealthCheckResponse:
```

Uses `optional_api_key` — no authentication required. Health checks must be public because load balancers and Kubernetes probes cannot provide API keys.

```python
redis_ok = False
try:
    import redis
    r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
    redis_ok = r.ping()
except Exception:
    pass
```

Each dependency is checked in a try/except with a timeout. If Redis is not running, the endpoint still returns 200 — but with `"redis": "unavailable"`. The service is degraded, not down. The 2-second timeout prevents the health check from hanging if Redis is unreachable.

**Why the health check returns "unavailable" instead of failing:**
- The API works without Redis (sync fallback). Reporting the API as unhealthy when Redis is down would cause load balancers to pull healthy instances out of rotation unnecessarily.

---

### `src/api/websocket.py` — Real-Time Processing Updates

```python
async def processing_websocket(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
```

**The WebSocket lifecycle:**

1. **Accept the connection.** The client connects to `ws://host/ws/processing/{task_id}`. `await websocket.accept()` completes the handshake.

2. **Initialize Celery result tracking:**

```python
try:
    from celery.result import AsyncResult
    from src.workers.celery_app import celery_app
    result = AsyncResult(task_id, app=celery_app)
except Exception as exc:
    update = ProcessingUpdate(task_id=task_id, status="error",
                              error=f"Celery unavailable: {exc}")
    await websocket.send_text(update.model_dump_json())
    await websocket.close()
    return
```

If Celery is not available, the client gets a single error message and the connection closes. This is better than hanging — the client knows immediately that real-time updates are not available.

3. **Polling loop:**

```python
while True:
    state = result.state
    meta = result.info if isinstance(result.info, dict) else {}

    if state == "PROCESSING":
        update = ProcessingUpdate(
            task_id=task_id, status="processing",
            stage=meta.get("stage"), progress=meta.get("progress", 0.0),
            step=meta.get("step"),
        )
    elif state == "SUCCESS":
        # ... send final update and break
    elif state in ("FAILURE", "REVOKED"):
        # ... send error update and break

    await websocket.send_text(update.model_dump_json())
    await asyncio.sleep(1)
```

**Why polling instead of push:**
- Celery does not natively support push notifications for state changes. `AsyncResult.state` is a pull-based API — you query it, it tells you the current state.
- 1-second polling is a practical compromise. Faster polling wastes CPU; slower polling makes the UI feel laggy.
- The `"PROCESSING"` custom state comes from `self.update_state()` in the Celery task. The `meta` dict contains `stage`, `progress`, and `step` — exactly what the frontend needs to render a progress bar.

4. **Graceful shutdown:**

```python
except WebSocketDisconnect:
    logger.info("WebSocket disconnected for task %s", task_id)
finally:
    try:
        await websocket.close()
    except Exception:
        pass
```

`WebSocketDisconnect` is not an error — it is the normal case when a user navigates away. The `finally` block ensures the connection is closed even if an unexpected exception occurs. The bare `except` in the close call prevents "already closed" errors from propagating.

---

### `src/workers/celery_app.py` — Celery Configuration

```python
celery_app = Celery(
    "intelligent_doc",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.workers.tasks"],
)
```

- **`broker`** — Where tasks are queued. Redis acts as a message queue.
- **`backend`** — Where results are stored. Same Redis instance (for simplicity; production might use a separate instance).
- **`include`** — Tells Celery where to find task definitions. Without this, `process_document.delay()` would fail with "task not registered."

```python
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
```

Key configuration decisions:

- **`task_serializer="json"`** — Tasks and results are serialized as JSON, not pickle. Pickle is a security risk (arbitrary code execution via crafted payloads). JSON is safe and human-readable.
- **`task_track_started=True`** — Workers report when they start a task, not just when they finish. This enables the `"PROCESSING"` state that the WebSocket uses.
- **`task_acks_late=True`** — The task is acknowledged **after** completion, not when dequeued. If a worker crashes mid-processing, the task goes back to the queue and gets retried by another worker. Without this, crashed tasks are silently lost.
- **`worker_prefetch_multiplier=1`** — Each worker fetches one task at a time. Document processing is CPU-heavy; prefetching multiple tasks would not help and could cause memory issues.

---

### `src/workers/tasks.py` — Processing Pipeline Task

This file contains the most important function in the entire system: `_process_document_impl`. It orchestrates every Phase 1-4 component into a single pipeline.

**The testable core (lines 35-197):**

```python
def _process_document_impl(
    doc_id: str,
    pdf_path: Path,
    update_fn: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
```

**Why this is a plain function, not a Celery task:**
- Celery tasks require Celery to be installed and configured. This function requires nothing — you can call it from a test, from a script, from the sync fallback in the upload endpoint, or from a Celery task.
- The `update_fn` callback is the only integration point with Celery. When called from Celery, it is `self.update_state()`. When called from tests, it is a mock that records calls. When called synchronously, it is `None`.

**Lazy imports inside the function (lines 56-61):**

```python
from src.api.stores import get_document_store, update_metrics
from src.chunking.chunker import SectionAwareChunker
from src.ingestion.layout_analyzer import LayoutAnalyzer
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import TableExtractor
```

Imports are inside the function, not at the top of the file. This is intentional:
- The function is defined in the `workers` package, which is imported when creating the Celery app. Moving these imports to the top would create circular dependencies (stores → schemas → ... → tasks).
- Lazy imports also mean the Celery worker only loads heavy libraries (PyMuPDF, pdfplumber, ChromaDB) when it actually processes a task, not at startup.

**The 9-stage pipeline (detailed in Section 8 below).**

**The Celery wrapper (lines 200-225):**

```python
def _create_celery_task():
    try:
        from celery import shared_task

        @shared_task(bind=True, name="process_document")
        def process_document(self, doc_id: str) -> dict:
            pdf_path = Path(settings.upload_dir) / f"{doc_id}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            def _celery_update(stage: str, progress: float, step: str) -> None:
                self.update_state(
                    state="PROCESSING",
                    meta={"stage": stage, "progress": progress, "step": step},
                )

            return _process_document_impl(doc_id, pdf_path, update_fn=_celery_update)

        return process_document
    except Exception:
        return None

process_document = _create_celery_task()
```

**Why `_create_celery_task()` instead of a top-level `@shared_task`:**
- If Celery is not installed, `from celery import shared_task` raises `ImportError`. The `try/except` catches this and returns `None`.
- The upload endpoint checks `if process_document is not None` before calling `.delay()`. If it is `None`, the sync fallback runs.
- This means the application **works without Celery installed**.

**Why `bind=True`:**
- `bind=True` gives the task access to `self`, which provides `self.update_state()`. This is how progress is reported to the WebSocket.

**Why `name="process_document"`:**
- Explicit task names prevent issues when the module path changes. Without a name, Celery auto-generates one from the module path, which breaks if you refactor.

---

### `src/models/schemas.py` — Phase 5 Additions

Phase 5 adds 16 Pydantic models. Key design decisions:

**QueryRequest validation:**

```python
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)
```

- `min_length=1` rejects empty queries at the validation layer (422), not in business logic.
- `max_length=2000` prevents LLM context overflow and abuse.
- `top_k` is bounded: `ge=1` (at least one passage), `le=20` (don't overwhelm the LLM context window).

**CompareRequest validation:**

```python
class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
```

`min_length=2` because comparing one paper is not a comparison. `max_length=5` because more than 5 papers would exceed LLM context limits.

**ProcessingUpdate — The WebSocket message schema:**

```python
class ProcessingUpdate(BaseModel):
    task_id: str
    status: str = Field(description="queued | processing | completed | failed")
    stage: str | None = None
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    step: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    result: dict | None = None
```

This is a union type disguised as a flat model. The `status` field determines which other fields are meaningful:
- `status="processing"` → `stage`, `progress`, `step` are populated.
- `status="completed"` → `progress=1.0`, `result` contains the pipeline summary.
- `status="failed"` → `error` contains the failure message.

**Why a flat model instead of discriminated unions:**
- WebSocket messages must be JSON-serializable. A flat model with optional fields is the simplest contract for frontend consumption. Discriminated unions add complexity without benefit for real-time status messages.

---

## 5. How the Async Celery Pipeline Works

The full flow from upload to completion:

```
Client                    FastAPI                 Redis              Celery Worker
  │                         │                       │                     │
  │  POST /upload           │                       │                     │
  │  (PDF file)             │                       │                     │
  │────────────────────────>│                       │                     │
  │                         │                       │                     │
  │                         │  1. Validate file     │                     │
  │                         │  2. Save to disk      │                     │
  │                         │  3. Create record     │                     │
  │                         │  4. process_document  │                     │
  │                         │     .delay(doc_id) ──>│  Task queued        │
  │                         │                       │                     │
  │  201 Created            │                       │                     │
  │  {id, task_id, status}  │                       │                     │
  │<────────────────────────│                       │                     │
  │                         │                       │                     │
  │                         │                       │  Task dequeued      │
  │                         │                       │────────────────────>│
  │                         │                       │                     │
  │                         │                       │                     │  Stage 1: PDF extraction
  │                         │                       │  update_state()  <──│
  │                         │                       │  {stage, progress}  │
  │                         │                       │                     │  Stage 2: Table extraction
  │                         │                       │  update_state()  <──│
  │                         │                       │                     │  ...
  │  WS /ws/processing/     │                       │                     │  Stage 9: Finalize
  │     {task_id}           │                       │                     │
  │────────────────────────>│                       │                     │
  │                         │  Poll AsyncResult ───>│                     │
  │  {"status":"processing",│<─────────────────────│                     │
  │   "progress": 0.55}     │                       │                     │
  │<────────────────────────│                       │                     │
  │                         │                       │                     │  Processing complete
  │                         │  Poll AsyncResult ───>│  state=SUCCESS      │
  │  {"status":"completed", │<─────────────────────│                     │
  │   "progress": 1.0}      │                       │                     │
  │<────────────────────────│                       │                     │
  │                         │                       │                     │
  │  WS closes              │                       │                     │
```

**When Redis is not available (sync fallback):**

```
Client                    FastAPI
  │                         │
  │  POST /upload           │
  │  (PDF file)             │
  │────────────────────────>│
  │                         │
  │                         │  1. Validate file
  │                         │  2. Save to disk
  │                         │  3. Create record
  │                         │  4. process_document.delay() → FAILS
  │                         │  5. Catch exception
  │                         │  6. Call _process_document_impl() directly
  │                         │     (runs synchronously in request handler)
  │                         │  7. Pipeline completes
  │                         │
  │  201 Created            │
  │  {id, status="completed"│
  │   task_id=null}         │
  │<────────────────────────│
```

The client does not need to know whether processing was async or sync. The response schema is identical. The only difference: `task_id` is `null` in sync mode (no WebSocket tracking possible).

---

## 6. How Authentication and Rate Limiting Work Together

The request processing order matters:

```
Request arrives
     │
     ▼
┌──────────────┐
│ CORS         │  Runs first (middleware). Adds headers. Always passes.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Route Match  │  FastAPI finds the handler function.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Rate Limiter │  @limiter.limit() checks IP before handler runs.
│              │  If limit exceeded: 429 returned. Handler never executes.
│              │  Auth has NOT been checked yet.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Dependencies │  Depends(verify_api_key) runs.
│              │  If key missing/invalid: 401 returned. Handler never executes.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Handler      │  Business logic runs.
└──────────────┘
```

**Why rate limiting runs before authentication:**

This is an intentional security decision. Rate limiting acts as a shield against brute-force attacks:

1. An attacker trying to guess API keys sends thousands of requests with different keys.
2. Rate limiting kicks in after 10/100/200 requests (depending on endpoint).
3. The attacker is blocked **before** the auth check runs, so they cannot probe which keys are valid.

If auth ran first, the attacker could distinguish "invalid key" (401) from "valid key, no permission" (403) at unlimited speed.

**Per-IP vs per-key limiting:**

The current implementation uses IP-based limiting (`key_func=get_remote_address`). This means:
- All users behind the same NAT share a rate limit bucket.
- A single user with multiple IPs gets multiple buckets.

In production, you would switch to per-key limiting:
```python
def get_api_key_or_ip(request: Request) -> str:
    key = request.headers.get("X-API-Key")
    return key if key else get_remote_address(request)
```

---

## 7. How the WebSocket Streams Real-Time Status

**The complete message sequence for a successful upload:**

```json
// Connection: ws://localhost:8000/ws/processing/abc-123

// Message 1 (task queued, not yet picked up)
{"task_id": "abc-123", "status": "pending", "progress": 0.0}

// Message 2 (worker picked up the task)
{"task_id": "abc-123", "status": "processing", "stage": "pdf_extraction",
 "progress": 0.10, "step": "Extracting text from PDF"}

// Message 3
{"task_id": "abc-123", "status": "processing", "stage": "table_extraction",
 "progress": 0.20, "step": "Extracting tables"}

// Message 4
{"task_id": "abc-123", "status": "processing", "stage": "layout_analysis",
 "progress": 0.30, "step": "Analyzing document layout"}

// ... messages for each stage ...

// Message 9
{"task_id": "abc-123", "status": "processing", "stage": "finalizing",
 "progress": 0.95, "step": "Saving results"}

// Message 10 (terminal — connection closes after this)
{"task_id": "abc-123", "status": "completed", "progress": 1.0,
 "result": {"document_id": "...", "total_pages": 12, "sections_found": 8,
            "chunks_created": 24, "processing_time_sec": 3.42}}
```

**For a failed upload:**

```json
{"task_id": "abc-123", "status": "failed",
 "error": "PDF extraction failed: not a valid PDF"}
```

**Frontend integration example:**

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/processing/${taskId}`);

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    progressBar.style.width = `${update.progress * 100}%`;
    statusText.textContent = update.step || update.status;

    if (update.status === 'completed' || update.status === 'failed') {
        ws.close();
    }
};
```

---

## 8. How the Celery Task Orchestrates All Phases

`_process_document_impl` is a 9-stage pipeline that calls every Phase 1-4 component in sequence:

```
Stage   Progress   Component                    Phase   What It Does
─────   ────────   ─────────────────────────   ─────   ──────────────────────────────────────
  1      10%       PDFParser.extract()           1      Extracts text from each PDF page
  2      20%       TableExtractor.extract()      1      Finds and parses tables from the PDF
  3      30%       LayoutAnalyzer.analyze()       1      Detects sections (abstract, intro, etc.)
  4      40%       MetadataExtractor.extract()    1      Extracts title, authors, DOI, keywords
  5      55%       SectionAwareChunker.chunk()    2      Splits sections into semantic chunks
  6      65%       EmbeddingService.embed()       2      Generates vector embeddings for chunks
  7      75%       VectorStore.store()            3      Indexes chunks in ChromaDB
  8      85%       PaperExtractor.extract()       4      LLM-based structured extraction
  9      95%       store.save(doc)                5      Saves all results to document store
```

**Error handling strategy — Fail-fast for required stages, warn-and-continue for optional:**

```python
# Stage 1-5: Required. If any fail, the entire pipeline fails.
parser = PDFParser()
extraction = parser.extract(pdf_path)  # Failure here → FAILED status

# Stage 6: Optional. No API key → skip with warning.
if settings.gemini_api_key and chunks:
    try:
        embeddings = embedding_service.embed_chunks(chunks)
    except Exception as exc:
        warnings.append(f"Embedding generation failed: {exc}")
        # Pipeline continues without embeddings

# Stage 8: Optional. LLM extraction failure is non-fatal.
if settings.gemini_api_key:
    try:
        paper_extraction = extractor.extract(paper_text, metadata)
    except Exception as exc:
        warnings.append(f"Structured extraction failed: {exc}")
        # Pipeline continues without structured extraction
```

**Why stages 6-8 are optional:**
- They depend on external services (Gemini API, ChromaDB). If those services are down, the document should still be searchable by keyword (BM25) and its sections/tables should still be viewable.
- The `warnings` list is returned in the result dict and stored — the user can see exactly what was skipped and why.

**The outermost try/except:**

```python
except Exception:
    logger.exception("Processing failed for %s", doc_id)
    try:
        store.update_status(doc_uuid, DocumentStatus.FAILED, error=str(doc_id))
    except Exception:
        pass
    raise
```

If any unhandled exception occurs:
1. The document is marked as FAILED in the store.
2. The exception is re-raised (Celery needs it to mark the task as FAILURE).
3. The nested try/except around `update_status` prevents a secondary exception from masking the original one.

---

## 9. Why Swagger Docs Matter for Portfolio Presentation

Run `uvicorn src.main:app` and open `http://localhost:8000/docs`. You get:

```
┌──────────────────────────────────────────────────────────────────┐
│  Intelligent Document Processing                                  │
│  AI-powered research paper analysis platform                      │
│  Version: 0.5.0                                                   │
│                                                                    │
│  [Authorize 🔒]                                                   │
│                                                                    │
│  ▼ documents (7 endpoints)                                        │
│    POST   /api/v1/documents/upload        Upload a PDF document   │
│    GET    /api/v1/documents/{doc_id}      Get document details    │
│    GET    /api/v1/documents/{doc_id}/status  Get processing status│
│    GET    /api/v1/documents/              List documents          │
│    DELETE /api/v1/documents/{doc_id}      Delete a document       │
│    GET    /api/v1/documents/{doc_id}/sections  Get sections       │
│    GET    /api/v1/documents/{doc_id}/tables    Get tables         │
│                                                                    │
│  ▼ query (4 endpoints)                                            │
│    POST   /api/v1/query                  Ask a question           │
│    POST   /api/v1/search                 Semantic search          │
│    POST   /api/v1/compare                Compare papers           │
│    GET    /api/v1/documents/{id}/summary/{level}  Get summary     │
│                                                                    │
│  ▼ extraction (4 endpoints)                                       │
│    POST   /api/v1/feedback               Submit correction        │
│    GET    /api/v1/documents/{id}/extraction  Get extraction       │
│    POST   /api/v1/documents/{id}/re-extract  Re-run extraction   │
│    GET    /api/v1/feedback/stats          Feedback statistics     │
│                                                                    │
│  ▼ admin (4 endpoints)                                            │
│    GET    /api/v1/admin/health            Service health check    │
│    GET    /api/v1/admin/metrics           System metrics          │
│    GET    /api/v1/admin/costs             Cost breakdown          │
│    GET    /api/v1/admin/eval/latest       Latest evaluation       │
└──────────────────────────────────────────────────────────────────┘
```

**Why this matters for interviews:**

1. **Immediate credibility.** An interviewer opens the URL and sees a professional API with grouped endpoints, descriptions, authentication, and schema documentation. This communicates "production-quality" without you saying a word.

2. **Interactive demo.** Click "Authorize," enter the test API key, and try any endpoint live. Upload a PDF, watch the status change, query it. No Postman setup, no curl commands, no README instructions needed.

3. **Schema documentation for free.** Click any endpoint, and Swagger shows the request body schema, response schema, and all possible status codes. Every `Field(description=...)` annotation appears here. This is documentation that stays in sync with the code because it IS the code.

4. **The "Try it out" button.** Swagger generates curl commands for every request. An interviewer can copy-paste them to test outside the browser. This is a better portfolio demo than a video or screenshot.

**What makes Swagger docs good:**

- `summary` and `description` on every route (the short text and long text in the UI).
- `tags` on every router (the grouping in the sidebar).
- `Field(description=...)` on every schema field (the parameter documentation).
- `response_model` on every endpoint (the response schema).
- `status_code` on mutating endpoints (201 for create, 204 for delete).

---

## 10. How Phase 5 Ties Together All Previous Phases

Before Phase 5, each phase was a library — callable from Python, testable in isolation, but not usable by anyone who is not a Python developer. Phase 5 turns libraries into a product.

```
Phase 1 (Ingestion)         → POST /upload triggers the pipeline
  PDFParser                    Stage 1: Extract text from uploaded PDF
  TableExtractor               Stage 2: Find and parse tables
  LayoutAnalyzer               Stage 3: Detect sections
  MetadataExtractor            Stage 4: Extract metadata

Phase 2 (Chunking)          → Stage 5 of the pipeline
  SectionAwareChunker          Creates semantic chunks for retrieval

Phase 3 (Retrieval)         → POST /search and POST /query
  EmbeddingService             Stage 6: Generate embeddings
  VectorStore                  Stage 7: Index in ChromaDB
  BM25Index                    Sparse retrieval for keyword search
  HybridRetriever              Combines dense + sparse results

Phase 4 (LLM Processing)   → POST /query, /compare, /summary, /extraction
  GeminiClient                 Shared LLM client with caching
  QAEngine                     Citation-tracked question answering
  PaperSummarizer              Multi-level summarization
  PaperExtractor               Stage 8: Structured extraction
  FaithfulnessVerifier         Hallucination detection in QA

Phase 5 (API Layer)         → Everything above, exposed via HTTP
  FastAPI + routers            RESTful interface for all operations
  Celery + Redis               Async processing pipeline
  WebSocket                    Real-time status updates
  Auth + Rate Limiting         Security and abuse prevention
  In-Memory Stores             MVP data layer (DB-ready)
```

**The integration test proves this works end-to-end:**

```python
def test_pipeline_completes(self, tmp_pdf: Path) -> None:
    result = _process_document_impl(str(doc_id), tmp_pdf)
    assert result["status"] == "completed"
    assert result["total_pages"] >= 1
    assert result["sections_found"] >= 1
    assert result["chunks_created"] >= 1
```

One function call runs every Phase 1-4 component on a real PDF and verifies the output. This is the integration test that proves the system works as a whole, not just in parts.

---

## 11. Testing Deep Dive

### Test Architecture

```
tests/
├── unit/
│   ├── test_auth.py                  8 tests  — Auth dependency in isolation
│   ├── test_rate_limiter.py          6 tests  — Rate limiter config and handler
│   ├── test_routes_documents.py     15 tests  — All 7 document endpoints
│   ├── test_routes_query.py         11 tests  — All 4 query/search endpoints
│   └── (300 tests from Phases 1-4)
└── integration/
    └── test_upload_pipeline.py       4 tests  — Full pipeline on real PDFs
```

**Total: 347 tests passing (300 from Phases 1-4 + 47 new in Phase 5).**

### Testing Pattern: Dependency Overrides

Every endpoint test follows the same pattern:

```python
# 1. Set up test data
doc = _make_doc(status=DocumentStatus.COMPLETED)
store = _make_store([doc])

# 2. Override dependencies to use test data
app.dependency_overrides[verify_api_key] = lambda: API_KEY
app.dependency_overrides[get_document_store] = lambda: store

try:
    # 3. Make the HTTP request
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/v1/documents/{doc.id}", headers=HEADERS)

    # 4. Assert on the response
    assert response.status_code == 200
    assert response.json()["filename"] == "test.pdf"
finally:
    # 5. Always clean up overrides
    app.dependency_overrides.clear()
```

**Why dependency overrides instead of `@patch`:**
- FastAPI's `dependency_overrides` is the framework-recommended pattern. It replaces the entire dependency resolution chain, not just one function call.
- `@patch` requires knowing the exact import path of the function being called. If an internal import path changes, all patches break. Dependency overrides are path-independent.
- The override replaces `verify_api_key` **at the FastAPI level**, meaning CORS, rate limiting, and middleware all still run. This tests the real request flow, not a mock.

**Why `try/finally` with `app.dependency_overrides.clear()`:**
- Overrides are global state on the app object. If a test fails and does not clean up, subsequent tests inherit the stale overrides. The `finally` block guarantees cleanup.

### Testing Auth in Isolation

```python
async def test_invalid_key(self, key_store: InMemoryAPIKeyStore) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key("bad-key", key_store)
    assert exc_info.value.status_code == 401
```

Auth tests call `verify_api_key` directly, passing in a fresh `InMemoryAPIKeyStore`. No HTTP request, no FastAPI app, no overrides. This is the fastest possible test — pure function call.

### Integration Tests: Real PDFs, Real Pipeline

```python
def test_pipeline_completes(self, tmp_pdf: Path) -> None:
    doc_id = uuid.uuid4()
    store = _setup_doc_in_store(doc_id, tmp_pdf)
    result = _process_document_impl(str(doc_id), tmp_pdf)
    assert result["status"] == "completed"
```

These tests use the `tmp_pdf` fixture from `conftest.py`, which creates a real 2-page PDF with PyMuPDF. They call `_process_document_impl` directly — no HTTP, no Celery, no Redis. This tests the full Phase 1-4 pipeline in one function call.

---

## 12. Interview Questions and Ideal Answers

### "Walk me through what happens when a user uploads a PDF."

> When a PDF hits `POST /api/v1/documents/upload`, six things happen before the response goes out:
>
> First, **validation**. The endpoint checks the file extension is `.pdf` and the file size is under the configured limit. These are cheap checks that reject bad requests immediately — 400 for wrong type, 413 for too large.
>
> Second, **integrity**. The file contents are SHA-256 hashed. This gives us a content fingerprint for deduplication later.
>
> Third, **storage**. The file is saved to disk as `{uuid}.pdf`. Using a UUID instead of the original filename prevents path traversal attacks and collision between files with the same name.
>
> Fourth, **record creation**. A `DocumentRecord` is created in the in-memory store with status PENDING.
>
> Fifth, **task dispatch**. The endpoint tries to call `process_document.delay(doc_id)` to queue the processing task in Celery via Redis. If Redis is not available — common in development — it catches the exception and falls back to calling `_process_document_impl()` synchronously, right in the request handler.
>
> Sixth, **response**. The client gets back a 201 Created with the document ID and task ID. If processing was async, they can connect to the WebSocket at `/ws/processing/{task_id}` to watch progress in real time.
>
> The processing pipeline itself runs 9 stages: PDF text extraction, table extraction, layout analysis, metadata extraction, semantic chunking, embedding generation, vector store indexing, LLM-based structured extraction, and finalization. Each stage reports progress through a callback that, in the Celery path, calls `self.update_state()` to update the task metadata in Redis. The WebSocket polls this metadata every second and streams it to the client.
>
> The key design decision is the Celery fallback. By wrapping the task dispatch in a try/except, the system works in three modes: full async with Celery and Redis, synchronous without any infrastructure, and degraded (some stages skipped) without a Gemini API key. This means a developer can `pip install` the package and start uploading PDFs immediately, while production gets horizontal scaling through Celery workers.

### "How do you handle long-running tasks in a web API?"

> The core problem is that HTTP requests should respond quickly — typically under a few seconds — but document processing takes 5-30 seconds depending on the PDF. Holding the connection open for that long is a bad user experience and risks timeouts.
>
> I solve this with the **task queue pattern**. The upload endpoint does the fast work (validate, save file, create record) and then delegates the slow work to a Celery task via `process_document.delay()`. The response returns immediately with a task ID.
>
> The client has three ways to track progress:
>
> 1. **Polling.** `GET /documents/{id}/status` returns the current status. Simple but inefficient — the client has to keep asking.
>
> 2. **WebSocket.** Connect to `/ws/processing/{task_id}` and receive progress updates every second. The server-side WebSocket handler polls the Celery `AsyncResult` and pushes `ProcessingUpdate` messages with `stage`, `progress` (0.0 to 1.0), and `step` (human-readable description). Terminal states (SUCCESS, FAILURE) close the connection.
>
> 3. **Fire and forget.** Upload, go do something else, come back later and check. The document's status will be COMPLETED or FAILED.
>
> The implementation separates the processing logic from Celery itself. `_process_document_impl` is a plain Python function — no Celery dependency. The Celery `@shared_task` wrapper is a thin layer that calls this function with a progress callback. This means:
> - Tests call the function directly. No Redis, no Celery, no infrastructure.
> - The sync fallback calls the function directly. It works without any task queue.
> - The Celery task adds only one capability: progress reporting via `self.update_state()`.
>
> For reliability, `task_acks_late=True` in the Celery config means tasks are acknowledged after completion. If a worker crashes mid-processing, the task goes back to the queue and another worker picks it up. Combined with `worker_prefetch_multiplier=1`, this prevents task loss and memory issues.

### "How does your rate limiting work?"

> Rate limiting uses **slowapi**, which is a FastAPI wrapper around the `limits` library. The limiter is configured with three tiers:
>
> - **Uploads: 10/hour.** Each upload triggers heavy processing — PDF parsing, embedding generation, LLM extraction. This prevents a single user from overwhelming the processing queue or running up LLM costs.
> - **Queries: 100/hour.** Each query involves an LLM call. At roughly $0.01 per query, this caps cost exposure at about $1/hour per user.
> - **Admin: 20/minute.** These are lightweight reads but expose internal metrics. Rate limiting prevents scraping.
>
> The key function is `get_remote_address`, which extracts the client IP. All limits are per-IP. The state is stored in memory (`storage_uri="memory://"`), which means counters reset on server restart. In production, this would point to Redis for persistence across restarts and sharing across multiple server instances.
>
> Rate limiting runs **before** authentication. This is a deliberate security choice. If auth ran first, an attacker could brute-force API keys at unlimited speed — they would only be rate limited after presenting a valid key. By checking rate limits first, we throttle all requests from a given IP, making brute-force attacks impractical.
>
> When a limit is exceeded, the custom handler returns a 429 response with a `Retry-After` header. This follows the HTTP standard — well-behaved clients (including browser fetch APIs) automatically respect this header and wait before retrying. The response body is JSON `{"detail": "Rate limit exceeded: 100 per 1 hour"}` to match the format of all other API errors.
>
> The rate limit is applied with a decorator: `@limiter.limit(QUERY_LIMIT)`. The handler function must accept a `Request` parameter (even if it does not use it) because slowapi needs the request object to extract the client IP. Without this parameter, rate limiting silently fails — which is a common gotcha with slowapi.

### "Why WebSocket instead of polling?"

> Both approaches work, and in fact our API supports both — `GET /documents/{id}/status` for polling and `ws://host/ws/processing/{task_id}` for real-time updates. But WebSocket is better for this use case for three reasons.
>
> **Latency.** Polling requires the client to choose a poll interval. Poll every 5 seconds and you might miss a 3-second processing window where the status jumped from 10% to 95%. Poll every 200ms and you are making 5 requests per second for a task that updates at most once per second. WebSocket pushes updates when they happen, not when the client asks.
>
> **Efficiency.** Each polling request is a full HTTP round trip — TCP handshake, TLS negotiation, HTTP headers, response. A WebSocket is a single persistent connection. For a task with 9 progress updates, polling sends 9+ HTTP requests; the WebSocket sends 9 tiny JSON messages over one connection.
>
> **User experience.** A progress bar that updates smoothly every second feels responsive and professional. A progress bar that jumps in 5-second increments feels broken. For a portfolio demo, the difference is significant.
>
> The implementation is pragmatic, not purist. The WebSocket handler does not use Celery's event system (which would require a dedicated Celery event monitor). Instead, it polls `AsyncResult.state` every second in an async loop. This is a server-side poll, but it is local (Redis query, not HTTP) and the client only sees smooth push updates. The 1-second interval matches the natural cadence of processing stages — most stages take 1-5 seconds.
>
> For graceful degradation, if Celery is not available, the WebSocket sends a single error message and closes. The client falls back to polling the REST endpoint. The frontend code handles both cases:
>
> ```javascript
> const ws = new WebSocket(wsUrl);
> ws.onerror = () => { startPolling(); }; // fallback
> ```
>
> This is the kind of detail that separates production-quality code from demo code — the happy path works, and so do all the failure modes.
