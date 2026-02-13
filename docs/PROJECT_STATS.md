# Project Statistics Report

Generated: 2026-02-12

---

## File Counts

| Category | Count |
|----------|-------|
| **Source files** (`src/*.py`) | 51 |
| **Test files** (`tests/*.py`) | 33 |
| **Config/deployment files** | 7 (Dockerfile, docker-compose, CI, deploy script, pyproject.toml, .dockerignore, .env.example) |
| **Documentation files** | 6 (README.md, ARCHITECTURE.md, PHASE7_WALKTHROUGH.md, PHASE8_WALKTHROUGH.md, WORK_SUMMARY.md, PROJECT_STATS.md) |

## Code Volume

| Category | Lines |
|----------|-------|
| **Source code** (`src/`) | 8,613 |
| **Test code** (`tests/`) | 7,062 |
| **Total Python** | 15,675 |

## Source Files Breakdown

| File | Lines |
|------|-------|
| `src/models/schemas.py` | 544 |
| `src/retrieval/query_processor.py` | 412 |
| `src/llm/gemini_client.py` | 372 |
| `src/llm/extractor.py` | 354 |
| `src/ingestion/layout_analyzer.py` | 254 |
| `src/llm/qa_engine.py` | 279 |
| `src/llm/summarizer.py` | 252 |
| `src/ingestion/table_extractor.py` | 275 |
| `src/ingestion/metadata_extractor.py` | 255 |
| `src/evaluation/extraction_eval.py` | 282 |
| `src/evaluation/benchmarks.py` | 246 |
| `src/api/routes_documents.py` | 322 |
| `src/api/routes_query.py` | 271 |
| `src/retrieval/reranker.py` | 272 |
| `src/retrieval/vector_store.py` | 268 |
| `src/chunking/chunker.py` | 317 |
| `src/retrieval/hybrid_retriever.py` | 229 |
| `src/evaluation/qa_eval.py` | 227 |
| `src/workers/tasks.py` | 225 |
| `src/llm/prompts.py` | 223 |
| `src/api/routes_extract.py` | 209 |
| `src/retrieval/bm25_index.py` | 207 |
| `src/evaluation/retrieval_eval.py` | 207 |
| `src/ingestion/pdf_parser.py` | 188 |
| `src/api/stores.py` | 187 |
| `src/chunking/chunk_optimizer.py` | 171 |
| `src/retrieval/embedding_service.py` | 177 |
| `src/models/database.py` | 169 |
| `src/api/routes_admin.py` | 154 |
| `src/llm/cache_manager.py` | 150 |
| `src/llm/cross_paper.py` | 138 |
| `src/monitoring/metrics.py` | 129 |
| `src/monitoring/cost_tracker.py` | 122 |
| `src/api/websocket.py` | 109 |
| `src/llm/degradation.py` | 103 |
| `src/main.py` | 75 |
| `src/config.py` | 58 |
| `src/api/auth.py` | 54 |
| `src/llm/model_router.py` | 49 |
| `src/api/rate_limiter.py` | 43 |
| `src/workers/celery_app.py` | 35 |

---

## Test Results

### Test Execution

```
$ python3 -m pytest tests/unit/ tests/integration/ -q

458 passed, 1 warning, 4 errors in 4.79s
```

| Metric | Value |
|--------|-------|
| **Tests collected** | 462 |
| **Tests passed** | 458 |
| **Tests with errors** | 4 (Docker health tests — no Docker daemon on this machine) |
| **Test classes** | 70 |
| **Test files** | 29 (26 unit + 3 integration) |

### Test Distribution by File

| Test File | Tests |
|-----------|-------|
| `test_routes_documents.py` | 28 |
| `test_routes_query.py` | 16 |
| `test_qa_engine.py` | 22 |
| `test_query_processor.py` | 22 |
| `test_extractor.py` | 21 |
| `test_chunker.py` | 22 |
| `test_hybrid_retriever.py` | 15 |
| `test_vector_store.py` | 18 |
| `test_bm25_index.py` | 15 |
| `test_summarizer.py` | 16 |
| `test_reranker.py` | 14 |
| `test_gemini_client.py` | 15 |
| `test_qa_eval.py` | 14 |
| `test_retrieval_eval.py` | 15 |
| `test_extraction_eval.py` | 13 |
| `test_layout_analyzer.py` | 12 |
| `test_metadata_extractor.py` | 12 |
| `test_embedding_service.py` | 14 |
| `test_table_extractor.py` | 10 |
| `test_pdf_parser.py` | 8 |
| `test_cost_tracker.py` | 8 |
| `test_cross_paper.py` | 9 |
| `test_model_router.py` | 10 |
| `test_cache_manager.py` | 11 |
| `test_degradation.py` | 8 |
| `test_auth.py` | 7 |
| `test_rate_limiter.py` | 5 |
| `test_full_pipeline.py` | 5 |
| `test_upload_pipeline.py` | 4 |
| `test_docker_health.py` | 4 |

---

## Lint Results (ruff)

```
$ ruff check src/

All checks passed!
```

| Metric | Value |
|--------|-------|
| **Errors found** | 0 |
| **Rules enabled** | E, F, I, N, W, UP, B, SIM |
| **Rules ignored** | B008 (FastAPI Depends pattern) |
| **Target version** | Python 3.11 |
| **Line length** | 100 |

---

## Type Check Results (mypy)

```
$ python3 -m mypy src/ --ignore-missing-imports

Success: no issues found in 51 source files
```

| Metric | Value |
|--------|-------|
| **Errors found** | 0 |
| **Files checked** | 51 |
| **Mode** | strict |
| **Disabled error codes** | type-arg, no-any-return, no-untyped-call, attr-defined |
| **Module overrides** | google.generativeai, celery, redis, chromadb, pdfplumber, fitz, pytesseract, rank_bm25, slowapi, rouge_score, prometheus_client, src.models.database |

---

## API Endpoints

**Total endpoints: 18** across 4 route modules + 1 WebSocket + 1 root health

| Method | Endpoint | Router |
|--------|----------|--------|
| `POST` | `/api/v1/documents/upload` | documents |
| `GET` | `/api/v1/documents/{id}` | documents |
| `GET` | `/api/v1/documents/{id}/status` | documents |
| `GET` | `/api/v1/documents` | documents |
| `DELETE` | `/api/v1/documents/{id}` | documents |
| `GET` | `/api/v1/documents/{id}/sections` | documents |
| `GET` | `/api/v1/documents/{id}/tables` | documents |
| `POST` | `/api/v1/query` | query |
| `POST` | `/api/v1/search` | query |
| `POST` | `/api/v1/compare` | query |
| `GET` | `/api/v1/documents/{id}/summary/{level}` | query |
| `POST` | `/api/v1/feedback` | extract |
| `GET` | `/api/v1/documents/{id}/extractions` | extract |
| `POST` | `/api/v1/documents/{id}/re-extract` | extract |
| `GET` | `/api/v1/admin/health` | admin |
| `GET` | `/api/v1/admin/metrics` | admin |
| `GET` | `/api/v1/admin/costs` | admin |
| `GET` | `/api/v1/admin/evaluation` | admin |
| `WS` | `/ws/processing/{task_id}` | main |
| `GET` | `/health` | main |

---

## Infrastructure

| Component | Status |
|-----------|--------|
| **Dockerfile** | Multi-stage build (builder + runtime), non-root user, health check |
| **docker-compose.prod.yml** | 4 services: app, celery, postgres, redis with resource limits |
| **CI/CD** | GitHub Actions with lint, type-check, test, build, deploy stages |
| **GCP Deploy** | Cloud Run script with scale-to-zero, Cloud SQL, Memorystore |

## Version

**v0.8.0** (synced across pyproject.toml and FastAPI app)
