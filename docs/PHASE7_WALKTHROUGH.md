# Phase 7: Docker & Deployment to GCP — Deep Walkthrough

> **Audience:** You, in an interview. Every section is written so you can explain
> each decision out loud, with the *why* front and centre.

---

## Table of Contents

1. [File-by-File Breakdown](#1-file-by-file-breakdown)
   - [.dockerignore](#11-dockerignore)
   - [Dockerfile (multi-stage)](#12-dockerfile-multi-stage)
   - [docker-compose.prod.yml](#13-docker-composeprodyml)
   - [.github/workflows/ci.yml](#14-githubworkflowsciyml)
   - [.github/workflows/pr.yml](#15-githubworkflowspryml)
   - [scripts/deploy_gcp.sh](#16-scriptsdeploy_gcpsh)
   - [tests/integration/test_docker_health.py](#17-testsintegrationtest_docker_healthpy)
   - [pyproject.toml changes](#18-pyprojecttoml-changes)
   - [cost_tracker.py timezone fix](#19-cost_trackerpy-timezone-fix)
2. [Why Multi-Stage Docker Build](#2-why-multi-stage-docker-build)
3. [Image Size Comparison: Single-Stage vs Multi-Stage](#3-image-size-comparison-single-stage-vs-multi-stage)
4. [Why Non-Root User in Docker](#4-why-non-root-user-in-docker)
5. [docker-compose.prod.yml vs docker-compose.yml](#5-docker-composeprodyml-vs-docker-composeyml)
6. [Why Health Checks Matter](#6-why-health-checks-matter)
7. [CI/CD Pipeline Step by Step](#7-cicd-pipeline-step-by-step)
8. [Why Evaluation Runs in CI](#8-why-evaluation-runs-in-ci)
9. [GCP Deployment Walkthrough](#9-gcp-deployment-walkthrough)
10. [Cost Breakdown](#10-cost-breakdown)
11. [Why Cloud Run Over GKE, App Engine, or Compute Engine](#11-why-cloud-run-over-gke-app-engine-or-compute-engine)
12. [Why min-instances=0 Saves Money](#12-why-min-instances0-saves-money)
13. [Interview Questions and Ideal Answers](#13-interview-questions-and-ideal-answers)

---

## 1. File-by-File Breakdown

### 1.1 `.dockerignore`

```
.git                 # Line 1  — Git history, often hundreds of MB. Never needed at runtime.
.venv                # Line 2  — Local virtualenv. The Dockerfile builds its own.
__pycache__          # Line 3  — Bytecode cache. Python regenerates it on import.
*.pyc                # Line 4  — Individual compiled bytecode files.
*.pyo                # Line 5  — Optimised bytecode. Same reasoning.
.env                 # Line 6  — CRITICAL: contains secrets (GEMINI_API_KEY, DB passwords).
                     #            Baking secrets into an image means anyone with
                     #            `docker pull` gets your API keys.
tests/               # Line 7  — Test code is not executed in production.
notebooks/           # Line 8  — Jupyter notebooks are dev-only artefacts.
docs/                # Line 9  — Documentation has no runtime purpose.
.mypy_cache          # Line 10 — Type-checker cache. Dev tool only.
.ruff_cache          # Line 11 — Linter cache. Dev tool only.
.pytest_cache        # Line 12 — Test runner cache.
data/                # Line 13 — Local data files. Production uses named Docker volumes.
*.egg-info/          # Line 14 — Package metadata created during `pip install -e .`.
.claude/             # Line 15 — Claude Code project context. Dev-only.
reports/             # Line 16 — Evaluation report outputs.
.github/             # Line 17 — CI/CD workflow definitions. Only GitHub Actions reads these.
```

**Design decision:** `.dockerignore` works like `.gitignore` but for `docker build`.
Without it, `COPY . .` sends *everything* to the Docker daemon as "build context."
This slows builds and risks leaking secrets.

**Why this matters:**
- **Security:** `.env` contains `GEMINI_API_KEY` and database passwords. If we don't
  exclude it, anyone who `docker inspect` or `docker history` the image can extract
  those secrets from the layer cache.
- **Speed:** `.git` alone can be hundreds of MB. Excluding it cuts build context
  transfer from seconds to milliseconds.
- **Image size:** Less context = fewer files to accidentally `COPY` into the image.

---

### 1.2 `Dockerfile` (multi-stage)

```dockerfile
# Line 1-3: Header comment identifying this as the production build.

# -----------------------------------------------------------------------
# STAGE 1: BUILDER
# -----------------------------------------------------------------------

# Line 8: FROM python:3.11-slim AS builder
```
**Why `python:3.11-slim`?** The `slim` variant is based on Debian with only the
minimal packages needed to run Python. The full `python:3.11` includes gcc, make,
and hundreds of other tools (~900 MB vs ~150 MB). We use `slim` in both stages
to minimise the base.

**Why `AS builder`?** This names the stage so Stage 2 can `COPY --from=builder`.
Docker discards this entire stage after the build — nothing from it ends up in the
final image unless explicitly copied.

```dockerfile
# Lines 10-13: Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \          # C compiler — needed to compile C extensions (e.g., asyncpg, uvloop)
    libpq-dev \    # PostgreSQL client headers — asyncpg compiles against these
    && rm -rf /var/lib/apt/lists/*   # Clean up apt cache to shrink this layer
```

**Why `--no-install-recommends`?** Apt by default installs "recommended" packages
that aren't strictly required. This flag skips them, saving ~50-100 MB.

**Why `rm -rf /var/lib/apt/lists/*`?** The apt package index is ~30 MB. We've
already installed what we need, so we delete it. This keeps the builder layer small,
and since this entire stage is discarded anyway, it's mostly good hygiene.

```dockerfile
# Lines 16-17: Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
```

**Why a venv inside Docker?** This seems redundant (the container IS isolated), but
there's a critical reason: we need to `COPY --from=builder /opt/venv /opt/venv`
into Stage 2. If we installed packages globally (`pip install` without a venv), we'd
have to copy scattered files from `/usr/local/lib/python3.11/site-packages/`,
`/usr/local/bin/`, and other locations. A venv is a single self-contained directory —
one `COPY` command grabs everything.

```dockerfile
# Line 19: WORKDIR /app

# Lines 22-23: Install production dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
```

**Why copy only `pyproject.toml` first?** Docker layer caching. If we copied the
entire source tree and *then* ran `pip install`, any code change (even a comment)
would bust the cache and re-download all dependencies. By copying only the dependency
manifest first, Docker reuses the `pip install` layer as long as dependencies haven't
changed. This turns a 2-minute rebuild into a 5-second one.

**Why `--no-cache-dir`?** Pip stores downloaded wheels in `~/.cache/pip/`. In a Docker
build, this cache is never reused (each build starts fresh), so it just wastes space.

**Why `pip install .` not `pip install -e ".[dev]"`?** Two reasons:
1. Production images should not have dev dependencies (pytest, ruff, mypy). They add
   ~100 MB and increase the attack surface.
2. `pip install .` (not `-e`) creates a proper installed package rather than a symlink,
   which is more reliable in production.

```dockerfile
# -----------------------------------------------------------------------
# STAGE 2: RUNTIME
# -----------------------------------------------------------------------

# Line 28: FROM python:3.11-slim AS runtime
```
**This starts a brand new image.** Nothing from Stage 1 exists here unless we
explicitly copy it. The final image is based on this line, not line 8. This is the
core mechanism that makes multi-stage builds produce small, secure images.

```dockerfile
# Lines 30-36: Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \       # OCR engine — the platform extracts text from scanned PDFs
    tesseract-ocr-eng \   # English language data for Tesseract
    poppler-utils \       # PDF rendering utilities (pdftotext, pdfinfo) used by pdfplumber
    libpq5 \              # PostgreSQL client library (runtime, NOT dev headers)
    curl \                # Used by HEALTHCHECK to probe the app
    && rm -rf /var/lib/apt/lists/*
```

**Key distinction: `libpq5` vs `libpq-dev`.**
- `libpq-dev` (Stage 1) = headers + static libs needed to *compile* asyncpg
- `libpq5` (Stage 2) = shared library needed to *run* asyncpg

This is the heart of multi-stage: compile in Stage 1, run in Stage 2, never ship the
compiler.

```dockerfile
# Lines 39-40: Copy the virtual environment from Stage 1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
```

This single `COPY` brings all installed Python packages (FastAPI, uvicorn, asyncpg,
pydantic, chromadb, etc.) without gcc, build headers, or pip cache.

```dockerfile
# Line 43: Copy application source code
COPY src/ /app/src/
```

**Why `src/` not `. `?** We only need the application code. Copying the entire repo
would pull in tests, docs, notebooks — exactly what `.dockerignore` prevents, but
being explicit here is defence in depth.

```dockerfile
# Lines 46-48: Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/uploads /app/data/chroma \
    && chown -R appuser:appuser /app
```

This creates the directories the app writes to (PDF uploads, ChromaDB persistence)
and gives `appuser` ownership. Everything in `/app` is now owned by `appuser`.
See [Section 4](#4-why-non-root-user-in-docker) for the security rationale.

```dockerfile
# Lines 50-51: Switch to non-root user
WORKDIR /app
USER appuser
```

**Every command after `USER appuser` runs as that user.** If the app is compromised,
the attacker has `appuser` privileges (can write to `/app/data/`) but cannot
`apt-get install` malware, modify system files, or read `/etc/shadow`.

```dockerfile
# Lines 53-54: Environment variables
ENV PYTHONPATH=/app          # So `from src.main import app` resolves correctly
ENV PYTHONUNBUFFERED=1       # Print logs immediately — don't buffer stdout/stderr
```

**Why `PYTHONUNBUFFERED=1`?** Without this, Python buffers stdout. If the container
crashes, the last few log lines are lost because they're still in the buffer. In
Docker, you want every `print()` and `logger.info()` to appear immediately in
`docker logs`.

```dockerfile
# Line 56: EXPOSE 8000
```

Documentation-only. `EXPOSE` doesn't actually open the port — `docker run -p 8000:8000`
does that. But it tells anyone reading the Dockerfile (and tools like Docker Desktop)
which port the app listens on.

```dockerfile
# Lines 58-59: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/admin/health || exit 1
```

Docker runs this command inside the container on a schedule:
- `--interval=30s` — Check every 30 seconds
- `--timeout=10s` — If curl takes >10s, treat it as a failure
- `--start-period=40s` — Give the app 40s to start before counting failures
  (uvicorn import + database connections take time)
- `--retries=3` — Mark "unhealthy" after 3 consecutive failures

`curl -f` means "fail silently on HTTP errors" — if the health endpoint returns 500,
curl exits with a non-zero code, and Docker marks the container unhealthy. See
[Section 6](#6-why-health-checks-matter) for why this matters.

```dockerfile
# Line 61: Default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why exec form `["uvicorn", ...]` not shell form `uvicorn ...`?** Exec form runs
uvicorn as PID 1 directly. Shell form runs `/bin/sh -c "uvicorn ..."`, which means
signals (SIGTERM from `docker stop`) go to `/bin/sh`, not uvicorn. Uvicorn needs
SIGTERM to gracefully drain connections.

**Why `--host 0.0.0.0`?** By default, uvicorn binds to `127.0.0.1` (localhost only).
Inside Docker, traffic comes from the Docker bridge network, which appears as an
external IP. Binding to `0.0.0.0` means "accept connections from any interface."

---

### 1.3 `docker-compose.prod.yml`

```yaml
# Lines 1-7: Header comment with usage instructions.
# "docker compose -f docker-compose.prod.yml up -d --build"
# The -f flag is required because the default filename is docker-compose.yml.
# The --build flag ensures images are rebuilt with latest code changes.

# Lines 9-42: APP SERVICE
services:
  app:
    build:
      context: .                    # Build context is the repo root
      dockerfile: Dockerfile        # Use the multi-stage Dockerfile
    ports:
      - "8000:8000"                 # Map host port 8000 to container port 8000
    env_file: .env                  # Load base env vars from .env file
    environment:                    # Override/add specific vars
      - DATABASE_URL=postgresql+asyncpg://docuser:docpass@postgres:5432/intelligent_doc
      #              ^^^^^^^^^^^^^^^^^^^^^^         ^^^^^^^^
      #              asyncpg driver                  "postgres" is the Docker service name.
      #                                              Docker DNS resolves this to the
      #                                              postgres container's IP.
      - REDIS_URL=redis://redis:6379/0
      - CHROMA_PERSIST_DIR=/app/data/chroma
      - UPLOAD_DIR=/app/data/uploads
    volumes:
      - upload_data:/app/data/uploads     # Named volume — persists across restarts
      - chroma_data:/app/data/chroma      # Named volume for vector DB persistence
    depends_on:
      postgres:
        condition: service_healthy         # Don't start app until postgres is HEALTHY
      redis:
        condition: service_healthy         # Don't start app until redis is HEALTHY
    restart: always                        # Restart on crash, OOM, or daemon restart
    deploy:
      resources:
        limits:
          memory: 2G                       # OOM kill if app exceeds 2GB
    healthcheck:                           # Same check as Dockerfile HEALTHCHECK
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/admin/health"]
      interval: 30s
      timeout: 10s
      start_period: 40s
      retries: 3
    networks:
      - app-network                        # Explicit bridge network
```

**Why `depends_on: condition: service_healthy`?** Without this, Docker starts all
containers simultaneously. The FastAPI app would try to connect to PostgreSQL before
PostgreSQL finishes initializing, causing a crash. `service_healthy` means "wait
until the postgres container's healthcheck passes."

**Why `restart: always`?** If the app crashes (uncaught exception, OOM), Docker
automatically restarts it. In production, you want self-healing. The alternative is
`on-failure` (restart only on non-zero exit), but `always` also handles Docker daemon
restarts (e.g., after a server reboot).

**Why `memory: 2G` limit?** Without a limit, a memory leak could consume all host
memory and crash other services. 2GB is enough for FastAPI + Tesseract OCR processing
but prevents runaway consumption.

```yaml
# Lines 44-63: CELERY WORKER SERVICE
  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A src.workers.celery_app worker --loglevel=info --concurrency=2
    #                 ^^^^^^^^^^^^^^^^^^^^^^^^^
    #  IMPORTANT: The dev docker-compose.yml had "src.worker" (singular, wrong module).
    #  The correct path is "src.workers.celery_app" — this was a bug fix.
```

**Why `--concurrency=2`?** This limits the Celery worker to 2 parallel tasks. PDF
processing is CPU-intensive (OCR, text extraction), so 2 tasks per worker prevents
CPU thrashing. In production, you scale by adding more worker containers, not more
concurrency per worker.

**Why the same Dockerfile for app and worker?** Both need the same Python packages
and system dependencies (tesseract, poppler). Using the same image means:
1. Only one image to build, test, and push.
2. No version drift between the web server and background workers.
3. The `command:` override replaces the `CMD` from the Dockerfile.

```yaml
# Lines 65-87: POSTGRESQL SERVICE
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: docuser
      POSTGRES_PASSWORD: docpass
      POSTGRES_DB: intelligent_doc
    ports:
      - "5432:5432"
    command: >
      postgres
        -c max_connections=100         # Default is 100, explicit for clarity
        -c shared_buffers=256MB        # Default is 128MB; 256MB is better for queries
        -c effective_cache_size=512MB  # Tells the query planner how much OS cache exists
    volumes:
      - postgres_data:/var/lib/postgresql/data    # Data survives container restarts
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U docuser -d intelligent_doc"]
      interval: 10s        # Check every 10s (faster than app because app depends on it)
      timeout: 5s
      retries: 5            # 5 retries x 10s = 50s max before giving up
    restart: always
```

**Why `postgres:16` not `postgres:16-alpine`?** The dev compose uses `alpine`. In
production, the Debian-based image is more battle-tested, has better glibc
compatibility, and Postgres official docs recommend it. Alpine uses musl libc which
can cause subtle issues with locale handling and DNS resolution under load.

**Why `shared_buffers=256MB`?** This is PostgreSQL's dedicated memory pool for
caching table and index data. The default 128MB is conservative. 256MB means more
query results are served from memory instead of disk, which improves read performance
for our document metadata queries.

**Why `effective_cache_size=512MB`?** This doesn't allocate memory — it tells the
PostgreSQL query planner how much total memory (shared_buffers + OS file cache) is
available. A higher value makes the planner prefer index scans over sequential scans,
which is correct when sufficient RAM exists.

```yaml
# Lines 89-103: REDIS SERVICE
  redis:
    image: redis:7-alpine             # Alpine is fine for Redis (pure C, no glibc issues)
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    #                     ^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Cap memory at 256MB.               When full, evict Least Recently Used keys.
    # Without this, Redis grows           LRU is correct for a cache: old data is
    # unbounded and OOM kills             evicted to make room for new data.
    # the container.
    volumes:
      - redis_data:/data               # Persist Redis data (Celery task results, cache)
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]    # Redis PING returns PONG if alive
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
```

**Why `allkeys-lru` not `volatile-lru`?** `volatile-lru` only evicts keys with a TTL
set. If any key is stored without a TTL (e.g., a Celery result), it becomes
non-evictable. Eventually, all 256MB is consumed by non-evictable keys, and Redis
starts rejecting writes. `allkeys-lru` can evict anything, which is safer.

```yaml
# Lines 105-113: VOLUMES AND NETWORKS
volumes:
  postgres_data:     # PostgreSQL database files
  redis_data:        # Redis persistence (RDB snapshots)
  upload_data:       # Uploaded PDF files
  chroma_data:       # ChromaDB vector embeddings

networks:
  app-network:
    driver: bridge   # Default Docker network driver; creates an isolated network
```

**Why named volumes?** Bind mounts (`./data:/app/data`) depend on host filesystem
paths, which vary between machines. Named volumes are managed by Docker, portable,
and survive `docker compose down` (only `down -v` removes them).

**Why an explicit network?** All services on `app-network` can resolve each other by
service name (e.g., `postgres`, `redis`). Without an explicit network, Docker Compose
creates one automatically, but being explicit makes the architecture visible and
allows future multi-compose setups to share a network.

---

### 1.4 `.github/workflows/ci.yml`

```yaml
# Lines 1-9: Header documenting the 5-stage pipeline.

# Lines 12-13: Trigger — only on push to main.
name: CI/CD Pipeline
on:
  push:
    branches: [main]
```

**Why only `main`?** Feature branches get PR checks (pr.yml). The full CI/CD pipeline
(including deploy) should only run on code that has been reviewed and merged.

```yaml
# Lines 18-19: Minimal permissions.
permissions:
  contents: read     # Can read repo files but cannot push, create branches, etc.
```

**Why explicit permissions?** GitHub Actions tokens default to read-write. If the
workflow is compromised (e.g., a malicious dependency in `pip install`), limiting to
`contents: read` prevents the attacker from modifying the repository.

```yaml
# Lines 21-25: Environment variables used across all jobs.
env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  SERVICE_NAME: research-processor
  IMAGE: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/research-processor/research-processor
```

**Why `us-central1`?** It has the best pricing for Cloud Run and Cloud SQL, and it's
co-located with most GCP services. Keeping everything in one region avoids cross-region
latency and egress charges.

**Why Artifact Registry path?** The format is
`REGION-docker.pkg.dev/PROJECT/REPOSITORY/IMAGE`. Google deprecated Container Registry
(gcr.io) in 2024. Artifact Registry is the current standard.

```yaml
# Lines 32-41: LINT JOB
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4      # Clone the repo
      - uses: actions/setup-python@v5  # Install Python
        with:
          python-version: "3.11"
      - run: pip install ruff          # Only install the linter, not the full project
      - run: ruff check src/           # Check for style/correctness violations
```

**Why install only `ruff`?** Ruff is a static analyser that reads source files
without importing them. It doesn't need FastAPI, asyncpg, or any project dependency.
Installing just ruff takes ~3 seconds vs ~45 seconds for the full project. This makes
the lint job finish in under 30 seconds.

```yaml
# Lines 43-52: TYPE CHECK JOB
  type-check:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"   # mypy needs all deps to resolve types
      - run: mypy src/
```

**Why `pip install -e ".[dev]"`?** Unlike ruff, mypy *does* need the full dependency
tree. It follows imports (`from fastapi import APIRouter`) and checks the types of
imported objects. Without FastAPI installed, mypy would report "module not found" for
every import.

```yaml
# Lines 54-67: UNIT TESTS JOB
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=term
      #       ^^^^^^^^^^^^^^^^^      ^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
      #       Only unit tests        Track    XML for artifact  Terminal for
      #       (no Docker needed)     coverage upload            CI log readability
      - uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage.xml          # Preserved for 90 days as a GitHub artifact
```

**Design decision: lint, type-check, and unit-tests run in parallel.** They have no
`needs:` clause, so GitHub runs all three simultaneously on separate VMs. This cuts
the wall-clock time from ~3 minutes (sequential) to ~1 minute (parallel). If any one
fails, the pipeline stops — there's no point running integration tests on code that
doesn't pass lint.

```yaml
# Lines 73-94: INTEGRATION TESTS JOB
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [lint, type-check, unit-tests]   # Only run if ALL three passed
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - name: Start backing services
        run: docker compose -f docker-compose.yml up -d postgres redis
        # Start ONLY postgres and redis — no need for the app container.
        # The tests use httpx to call the app directly.
      - name: Wait for services to be healthy
        run: |
          for i in $(seq 1 30); do
            docker compose -f docker-compose.yml exec -T postgres pg_isready -U docuser && break
            sleep 2
          done
          # Poll every 2s for up to 60s. pg_isready returns 0 when postgres is ready.
      - run: pytest tests/integration/ -v
      - name: Tear down services
        if: always()    # Run even if tests fail, so we don't leak containers
        run: docker compose -f docker-compose.yml down
```

**Why `needs: [lint, type-check, unit-tests]`?** This creates a dependency gate. If
lint fails (you have a style violation), there's no point spinning up Docker containers
for integration tests. This saves CI minutes (which cost money on GitHub Actions) and
gives faster feedback.

**Why `if: always()` on teardown?** Without this, if `pytest` fails, the teardown step
is skipped (GitHub Actions skips steps after a failure by default). This would leave
Docker containers running on the CI machine. `if: always()` means "run this step
regardless of whether previous steps passed or failed."

```yaml
# Lines 100-122: EVALUATION BENCHMARK JOB
  evaluation:
    name: Evaluation Benchmark
    runs-on: ubuntu-latest
    needs: [integration-tests]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e .
      - name: Run evaluation benchmark
        run: |
          mkdir -p reports
          python scripts/run_eval.py --output reports/eval.json
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        continue-on-error: true    # Don't block deploy if eval fails
      - uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: reports/eval.json
          if-no-files-found: ignore   # Don't fail if eval didn't produce a report
```

**Why `continue-on-error: true`?** The evaluation benchmark calls the Gemini API and
requires indexed documents. In a fresh CI environment, there may be no documents to
evaluate against. Blocking deployment because an evaluation couldn't run (not because
quality degraded) would be counterproductive. The report is uploaded as an artifact
for humans to review.

See [Section 8](#8-why-evaluation-runs-in-ci) for deeper reasoning.

```yaml
# Lines 128-149: BUILD AND PUSH JOB
  build-and-push:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    needs: [integration-tests]       # Don't build a broken image
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Authenticate to GCP
        id: auth
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
          # GCP_SA_KEY is a service account JSON key stored as a GitHub secret.
          # The service account needs: Artifact Registry Writer, Cloud Run Admin.
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
        # This adds Artifact Registry as a Docker credential helper,
        # so `docker push` authenticates automatically.
      - name: Build Docker image
        run: docker build -t ${{ env.IMAGE }}:${{ github.sha }} -t ${{ env.IMAGE }}:latest .
        # Two tags:
        #   :sha    — immutable, uniquely identifies this exact build
        #   :latest — mutable, always points to the most recent build
      - name: Push to Artifact Registry
        run: |
          docker push ${{ env.IMAGE }}:${{ github.sha }}
          docker push ${{ env.IMAGE }}:latest
```

**Why tag with both `${{ github.sha }}` and `latest`?** The SHA tag is immutable —
you can always roll back to a specific commit's image. The `latest` tag is a
convenience for manual `docker pull`. In the deploy job, we use the SHA tag to
guarantee we deploy exactly what was built.

```yaml
# Lines 155-187: DEPLOY JOB
  deploy:
    name: Deploy to Cloud Run
    runs-on: ubuntu-latest
    needs: [build-and-push]     # Only deploy after image is pushed
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image ${{ env.IMAGE }}:${{ github.sha }} \
            #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Deploy the EXACT image we just built, not :latest.
            # This prevents race conditions if two pushes happen close together.
            --region ${{ env.REGION }} \
            --platform managed \         # Cloud Run fully managed (not Anthos)
            --memory 2Gi \               # 2 GB RAM per instance
            --cpu 2 \                    # 2 vCPUs per instance
            --min-instances 0 \          # Scale to zero when idle (see Section 12)
            --max-instances 4 \          # Cap at 4 instances to limit cost
            --allow-unauthenticated \    # Public API (auth handled by app layer)
            --set-env-vars "PYTHONPATH=/app" \
            --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,DATABASE_URL=database-url:latest" \
            #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Secrets are mounted from GCP Secret Manager, never stored in env vars
            # or committed to the repo. Cloud Run reads them at container startup.
            --port 8000
      - name: Print service URL
        run: |
          gcloud run services describe ${{ env.SERVICE_NAME }} \
            --region ${{ env.REGION }} \
            --format "value(status.url)"
```

**Why `--image :${{ github.sha }}` not `:latest`?** If two developers push to main
within minutes, there's a race condition: push A builds image A, push B builds image
B and tags it `:latest`, then push A's deploy job runs and deploys `:latest` — which
is actually image B. Using the commit SHA eliminates this entirely.

---

### 1.5 `.github/workflows/pr.yml`

```yaml
# Lines 1-6: Header.

# Lines 8-11: Trigger on pull requests targeting main.
name: PR Checks
on:
  pull_request:
    branches: [main]

# Lines 14-16: Permissions — needs write access to post PR comments.
permissions:
  contents: read
  pull-requests: write   # Required for orgoro/coverage to post a comment
```

```yaml
# Lines 19-28: LINT JOB (identical to ci.yml)
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check src/
```

```yaml
# Lines 30-39: TYPE CHECK JOB (identical to ci.yml)
  type-check:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: mypy src/

# Lines 41-56: UNIT TESTS + COVERAGE JOB
  unit-tests:
    name: Unit Tests + Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=term
      - name: Post coverage comment
        uses: orgoro/coverage@v3.2
        with:
          coverageFile: coverage.xml
          token: ${{ secrets.GITHUB_TOKEN }}
        if: github.event_name == 'pull_request'
        # Posts a comment on the PR showing which files are covered and which aren't.
        # The GITHUB_TOKEN is automatically provided — no secret setup needed.
```

**Why separate pr.yml and ci.yml?** PRs should get *fast* feedback (lint + tests in
~1 minute). They should NOT deploy — deploying unreviewed code to production would be
dangerous. By splitting the workflows, we get:

| Workflow | Trigger | Deploys? | Speed |
|----------|---------|----------|-------|
| pr.yml | Pull request | No | ~1 min |
| ci.yml | Push to main | Yes | ~5 min |

---

### 1.6 `scripts/deploy_gcp.sh`

This is a manual deployment script for first-time GCP setup or environments
without CI/CD.

```bash
# Line 27: set -euo pipefail
```
- `set -e` — Exit immediately if any command fails (non-zero exit code).
- `set -u` — Treat unset variables as errors (catches typos like `$PROEJCT_ID`).
- `set -o pipefail` — A pipeline fails if *any* command in it fails, not just the last
  one. Without this, `curl | grep` succeeds even if `curl` fails, because `grep`'s
  exit code overwrites it.

**Why all three?** This is the "strict mode" for bash. In a deployment script, silently
continuing after a failed command could lead to deploying a broken application or
leaving infrastructure in an inconsistent state.

```bash
# Lines 32-40: Configuration variables
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
# ${VAR:-default} syntax: use $GCP_PROJECT_ID if set, otherwise use the literal string.
# The literal string will cause gcloud to fail with a clear error, which is better
# than silently using an empty string.

REGION="us-central1"
SERVICE_NAME="research-processor"
DB_INSTANCE="research-db"
DB_NAME="intelligent_doc"
DB_USER="docuser"
BUCKET_NAME="${PROJECT_ID}-research-uploads"
# Bucket names must be globally unique. Prefixing with project ID ensures uniqueness.

REPO_NAME="research-processor"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"
```

```bash
# Lines 48-50: Step 1 — Set GCP project
gcloud config set project "${PROJECT_ID}"
# All subsequent gcloud commands operate on this project.
```

```bash
# Lines 57-63: Step 2 — Enable APIs
gcloud services enable \
    run.googleapis.com \              # Cloud Run (hosting)
    sqladmin.googleapis.com \         # Cloud SQL Admin (database management)
    storage.googleapis.com \          # Cloud Storage (file uploads)
    artifactregistry.googleapis.com \ # Artifact Registry (Docker images)
    secretmanager.googleapis.com      # Secret Manager (API keys, DB passwords)
# This is idempotent — enabling an already-enabled API is a no-op.
# Cost: Free. You only pay when you use the services.
```

```bash
# Lines 74-82: Step 3 — Create Cloud SQL instance
gcloud sql instances create "${DB_INSTANCE}" \
    --database-version=POSTGRES_16 \
    --tier=db-f1-micro \              # Cheapest tier: shared CPU, 0.6GB RAM, ~$7.67/month
    --region="${REGION}" \
    --storage-type=HDD \              # HDD is cheaper than SSD ($0.09/GB vs $0.17/GB)
    --storage-size=10GB \             # Minimum size. Auto-grows if needed.
    --no-assign-ip \                  # No public IP — only accessible via Cloud SQL Proxy
    --network=default \               # Use VPC private networking
    || echo "    Instance may already exist, continuing..."
    # The || pattern makes this idempotent. If the instance already exists,
    # gcloud exits non-zero, but we catch it and continue.
```

**Why `--no-assign-ip`?** A public IP on a database is a security risk. With
`--no-assign-ip`, the database is only accessible from within the VPC (which includes
Cloud Run via `--add-cloudsql-instances`). An attacker who compromises a different
system cannot directly connect to your database.

```bash
# Lines 90-105: Step 4 — Create database, user, and store password
gcloud sql databases create "${DB_NAME}" --instance="${DB_INSTANCE}" \
    || echo "    Database may already exist, continuing..."

DB_PASSWORD=$(openssl rand -base64 24)
# Generate a random 24-byte password encoded as base64 (32 characters).
# NEVER hardcode passwords. NEVER use "password123".

gcloud sql users create "${DB_USER}" --instance="${DB_INSTANCE}" \
    --password="${DB_PASSWORD}" \
    || echo "    User may already exist, continuing..."

# Store password in Secret Manager
echo -n "${DB_PASSWORD}" | gcloud secrets create db-password --data-file=- 2>/dev/null \
    || echo -n "${DB_PASSWORD}" | gcloud secrets versions add db-password --data-file=-
# Try to create the secret. If it already exists (stderr suppressed), add a new version.
# Secret Manager keeps all versions — you can roll back if needed.
```

```bash
# Lines 115-135: Step 5 — Cloud Storage bucket with lifecycle
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \    # Disable per-object ACLs (simpler security model)
    || echo "    Bucket may already exist, continuing..."

# 30-day lifecycle rule
cat > /tmp/lifecycle.json <<'LIFECYCLE'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}       # Delete objects older than 30 days
      }
    ]
  }
}
LIFECYCLE
gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file=/tmp/lifecycle.json
rm -f /tmp/lifecycle.json
```

**Why a 30-day lifecycle?** Uploaded PDFs are processed (text extracted, embeddings
generated) and the results are stored in PostgreSQL and ChromaDB. The original PDFs are
only needed for re-processing. A 30-day window gives ample time for that while preventing
unbounded storage growth. At $0.020/GB/month, even a few hundred PDFs barely register, but
the lifecycle policy is good hygiene.

```bash
# Lines 146-157: Step 6 — Artifact Registry + Docker build
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    || echo "    Repository may already exist, continuing..."

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
# This modifies ~/.docker/config.json to use gcloud as a credential helper.

docker build -t "${IMAGE}:latest" .
docker push "${IMAGE}:latest"
```

```bash
# Lines 167-182: Step 7 — Store secrets
# Gemini API key
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "    WARNING: GEMINI_API_KEY not set in environment."
else
    echo -n "${GEMINI_API_KEY}" | gcloud secrets create gemini-api-key --data-file=- 2>/dev/null \
        || echo -n "${GEMINI_API_KEY}" | gcloud secrets versions add gemini-api-key --data-file=-
fi

# Database URL (constructed from Cloud SQL connection name)
CLOUD_SQL_CONNECTION="${PROJECT_ID}:${REGION}:${DB_INSTANCE}"
DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@/${DB_NAME}?host=/cloudsql/${CLOUD_SQL_CONNECTION}"
# The ?host=/cloudsql/... format tells asyncpg to connect via the Cloud SQL Proxy unix socket.
# Cloud Run automatically mounts this socket when you use --add-cloudsql-instances.
```

```bash
# Lines 193-205: Step 8 — Deploy to Cloud Run
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}:latest" \
    --region "${REGION}" \
    --platform managed \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \          # Scale to zero — $0 when idle
    --max-instances 4 \          # Cap at 4 to prevent runaway costs
    --allow-unauthenticated \    # Public API
    --add-cloudsql-instances "${CLOUD_SQL_CONNECTION}" \
    # This mounts the Cloud SQL Proxy sidecar, creating a unix socket at
    # /cloudsql/PROJECT:REGION:INSTANCE that asyncpg connects to.
    --set-env-vars "PYTHONPATH=/app,UPLOAD_DIR=/app/data/uploads,CHROMA_PERSIST_DIR=/app/data/chroma" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,DATABASE_URL=database-url:latest" \
    --port 8000
```

```bash
# Lines 210-229: Step 9 — Print results and cost tips
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo "  Service URL:   ${SERVICE_URL}"
echo "  Health check:  ${SERVICE_URL}/api/v1/admin/health"
echo "  API docs:      ${SERVICE_URL}/docs"
```

---

### 1.7 `tests/integration/test_docker_health.py`

```python
# Lines 1-8: Module docstring explaining this test requires Docker.
# Not in default CI because it needs running Docker containers.

# Lines 10-16: Imports
import subprocess    # To run docker compose commands
import time          # For polling loop
import httpx         # HTTP client (same one used in app tests)
import pytest        # Test framework

# Lines 18-21: Constants
COMPOSE_FILE = "docker-compose.prod.yml"
HEALTH_URL = "http://localhost:8000/api/v1/admin/health"
STARTUP_TIMEOUT = 120   # 2 minutes — enough for Postgres + app initialization
POLL_INTERVAL = 3       # Check every 3 seconds
```

```python
# Lines 24-71: Module-scoped fixture
@pytest.fixture(scope="module")
def docker_services():
    # scope="module" means: start containers ONCE for all tests in this file,
    # not once per test. Starting Docker takes ~30s; running it 4 times = 2 minutes wasted.

    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", "--build"],
        check=True,          # Raise exception if docker compose fails
        capture_output=True,  # Don't pollute test output
        text=True,
    )

    # Poll the health endpoint until the app is ready
    start = time.time()
    healthy = False
    while time.time() - start < STARTUP_TIMEOUT:
        try:
            resp = httpx.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                healthy = True
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass   # Container not ready yet — keep polling
        time.sleep(POLL_INTERVAL)

    if not healthy:
        # Capture logs BEFORE tearing down — essential for debugging
        logs = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "logs", "--tail=50"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
            capture_output=True, text=True,
        )
        pytest.fail(
            f"Services did not become healthy within {STARTUP_TIMEOUT}s.\n"
            f"{logs.stdout}\n{logs.stderr}"
        )

    yield   # Tests run here

    # Teardown: stop and remove containers + volumes
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
        check=True, capture_output=True, text=True,
    )
    # -v removes named volumes so each test run starts fresh.
```

```python
# Lines 74-103: Test class
class TestDockerHealth:

    def test_health_endpoint_returns_200(self, docker_services):
        resp = httpx.get(HEALTH_URL, timeout=10)
        assert resp.status_code == 200
        # Basic smoke test: the app is running and responding.

    def test_health_all_services_ok(self, docker_services):
        resp = httpx.get(HEALTH_URL, timeout=10)
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database"] == "ok"
        assert data["redis"] == "ok"
        assert "chromadb" in data
        # Verifies ALL backing services are connected, not just the app.

    def test_health_includes_version(self, docker_services):
        resp = httpx.get(HEALTH_URL, timeout=10)
        data = resp.json()
        assert "version" in data
        assert len(data["version"]) > 0
        # Version string helps debug "which version is deployed?" issues.

    def test_health_includes_timestamp(self, docker_services):
        resp = httpx.get(HEALTH_URL, timeout=10)
        data = resp.json()
        assert "timestamp" in data
        # Timestamp confirms the response is freshly generated, not cached.
```

---

### 1.8 `pyproject.toml` Changes

```toml
# Version bump: 0.1.0 → 0.7.0
# Why 0.7.0? This is Phase 7. The version tracks project milestones.
version = "0.7.0"

# Added metadata for PyPI compatibility and professional presentation:
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Divya"}]
classifiers = [
    "Programming Language :: Python :: 3.11",
    "Framework :: FastAPI",
]

[project.urls]
Repository = "https://github.com/divya/intelligent-doc"

# Added strict markers to pytest config:
addopts = "--strict-markers"
# This causes pytest to ERROR if a test uses an unregistered marker (e.g.,
# @pytest.mark.slooow — a typo). Without --strict-markers, pytest silently
# treats it as a new marker, and the test runs unconditionally.
```

---

### 1.9 `cost_tracker.py` Timezone Fix

```python
# BEFORE (bug):
def get_daily_cost(self, day: date | None = None) -> float:
    target = day or date.today()
    # date.today() uses the LOCAL timezone.
    # CostRecord.timestamp uses datetime.now(timezone.utc).
    # In UTC+5:30 (India), between 00:00 and 05:30 IST:
    #   date.today() = Feb 13 (IST)
    #   record.timestamp.date() = Feb 12 (UTC)
    # These don't match → daily cost returns 0 → test fails.

# AFTER (fix):
def get_daily_cost(self, day: date | None = None) -> float:
    target = day or datetime.now(timezone.utc).date()
    # Now both sides use UTC. Feb 12 UTC == Feb 12 UTC. Always matches.
```

**Lesson:** Never mix timezone-aware and timezone-naive datetimes. Always use
`datetime.now(timezone.utc)` instead of `datetime.now()` or `date.today()`.

---

## 2. Why Multi-Stage Docker Build

A multi-stage build uses multiple `FROM` statements. Each `FROM` creates an
independent stage. Only the final stage becomes the output image.

**The problem with single-stage:**
```dockerfile
# Single-stage: everything in one image
FROM python:3.11-slim
RUN apt-get install -y gcc libpq-dev tesseract-ocr poppler-utils curl
COPY . .
RUN pip install -e ".[dev]"    # Installs pytest, ruff, mypy too
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This image ships with:
- gcc (C compiler) — an attacker who gains shell access can compile exploits
- libpq-dev (headers) — useful only for compiling, not running
- pytest, ruff, mypy — dev tools that have no production purpose
- `tests/`, `docs/`, `.git/` — unnecessary files

**The multi-stage solution:**

```
Stage 1 (builder):     Stage 2 (runtime):
┌─────────────────┐    ┌─────────────────┐
│ python:3.11-slim│    │ python:3.11-slim│
│ + gcc           │    │ + tesseract     │
│ + libpq-dev     │    │ + poppler-utils │
│ + pip install   │───>│ + /opt/venv     │  (COPY --from=builder)
│                 │    │ + src/          │
│ DISCARDED       │    │ = FINAL IMAGE   │
└─────────────────┘    └─────────────────┘
```

Stage 1 compiles everything. Stage 2 copies only the compiled output (`/opt/venv`).
The compiler, headers, pip cache, and build artifacts stay in Stage 1 and are
discarded.

**Security benefit:** If an attacker exploits a vulnerability in the running
application, they find themselves in a minimal environment with no compiler, no
package manager cache, no dev tools. The attack surface is drastically reduced.

**Size benefit:** See the next section.

---

## 3. Image Size Comparison: Single-Stage vs Multi-Stage

| Component | Single-Stage | Multi-Stage Runtime |
|-----------|-------------|-------------------|
| Base image (`python:3.11-slim`) | 150 MB | 150 MB |
| gcc + build-essential | ~150 MB | 0 MB (discarded) |
| libpq-dev (headers) | ~20 MB | 0 MB (discarded) |
| libpq5 (runtime lib) | ~5 MB | ~5 MB |
| tesseract-ocr + eng | ~30 MB | ~30 MB |
| poppler-utils | ~10 MB | ~10 MB |
| curl | ~5 MB | ~5 MB |
| Python packages (prod) | ~200 MB | ~200 MB |
| Python packages (dev: pytest, ruff, mypy) | ~100 MB | 0 MB (not installed) |
| pip cache | ~100 MB | 0 MB (`--no-cache-dir`) |
| apt cache | ~50 MB | 0 MB (`rm -rf /var/lib/apt/lists/*`) |
| Source code (`src/`) | ~2 MB | ~2 MB |
| Tests, docs, notebooks | ~5 MB | 0 MB (not copied) |
| `.git` directory | ~50 MB | 0 MB (`.dockerignore`) |
| **Total estimated** | **~877 MB** | **~402 MB** |

**The multi-stage image is roughly half the size** — ~400 MB vs ~900 MB.

Why image size matters:
1. **Faster deploys:** Cloud Run pulls the image from Artifact Registry on every cold
   start. 400 MB pulls in ~4 seconds; 900 MB takes ~9 seconds. This directly affects
   cold start latency.
2. **Lower storage cost:** Artifact Registry charges $0.10/GB/month. A 400 MB image
   costs $0.04/month; 900 MB costs $0.09/month. Small per-image, but multiplied by
   every tagged version.
3. **Faster CI:** `docker push` in CI takes half the time, saving CI minutes.
4. **Smaller attack surface:** Fewer installed packages = fewer potential CVEs.

---

## 4. Why Non-Root User in Docker

By default, Docker containers run as `root`. This is dangerous because:

1. **Container escape:** If an attacker exploits a kernel vulnerability to escape the
   container, they're root on the host machine. As `appuser`, they'd be an
   unprivileged user — significantly less damage potential.

2. **File system access:** Root inside a container can read `/etc/shadow`, install
   packages via `apt-get`, and modify system configuration. `appuser` can only write
   to `/app/data/` (owned by appuser) and `/home/appuser/`.

3. **Principle of least privilege:** The application needs to:
   - Read Python files in `/app/src/` (any user can read these)
   - Write uploaded PDFs to `/app/data/uploads/` (owned by appuser)
   - Write ChromaDB data to `/app/data/chroma/` (owned by appuser)
   - Listen on port 8000 (ports > 1024 don't require root)

   It does NOT need to install packages, modify system files, or access other users'
   data. Running as root grants all of these unnecessarily.

4. **Cloud Run requirement:** Google Cloud Run expects containers to run as non-root
   by default. While not strictly enforced, it's a security best practice that
   Google recommends, and some organisations enforce it via policy.

The implementation:
```dockerfile
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/uploads /app/data/chroma \
    && chown -R appuser:appuser /app
# Create user, create data directories, make appuser own everything under /app.

USER appuser
# All subsequent RUN, CMD, and ENTRYPOINT commands run as appuser.
```

---

## 5. docker-compose.prod.yml vs docker-compose.yml

| Feature | Dev (`docker-compose.yml`) | Prod (`docker-compose.prod.yml`) |
|---------|--------------------------|-------------------------------|
| **Restart policy** | None (container stops, stays stopped) | `restart: always` (auto-recover) |
| **Memory limits** | None (can OOM the host) | `memory: 2G` (bounded) |
| **Health checks on app** | None | curl-based health check |
| **Postgres image** | `postgres:16-alpine` | `postgres:16` (Debian, more stable) |
| **Postgres tuning** | Defaults | `shared_buffers=256MB`, `max_connections=100`, `effective_cache_size=512MB` |
| **Redis memory policy** | None (grows unbounded) | `maxmemory 256mb`, `allkeys-lru` |
| **Celery command** | `celery -A src.worker worker` (BUG: wrong module path) | `celery -A src.workers.celery_app worker` (correct) |
| **Explicit network** | None (default) | `app-network` bridge |
| **depends_on conditions** | `service_healthy` | `service_healthy` |
| **Environment vars** | `DATABASE_URL`, `REDIS_URL` | + `CHROMA_PERSIST_DIR`, `UPLOAD_DIR` |

**Key differences explained:**

**`restart: always`** — In development, if a container crashes, you want to see the
error and fix it. In production, you want the container to restart automatically while
you're paged and investigating. `always` also handles Docker daemon restarts after
host reboots.

**Memory limits** — In development, your machine is yours and you'll notice if
something eats too much memory. In production, a memory leak in one container can
starve PostgreSQL of memory, causing cascading failures. The 2GB limit acts as a
circuit breaker.

**PostgreSQL tuning** — The defaults are designed for a machine with 128 MB total RAM
(compatibility with very old hardware). With 2+ GB available, increasing
`shared_buffers` to 256 MB means more query data is cached in PostgreSQL's own memory,
reducing disk I/O.

**Celery module path fix** — The dev compose had `celery -A src.worker` (singular),
but the actual module is `src.workers.celery_app` (plural `workers`, with the specific
module name). This was a bug that would cause `ModuleNotFoundError` when starting the
Celery worker.

**Redis `maxmemory`** — In dev, Redis might store a few hundred keys and never hit a
limit. In production, without a memory cap, Redis grows until the container is OOM
killed. With `maxmemory 256mb` and `allkeys-lru`, Redis gracefully evicts old cache
entries instead of crashing.

---

## 6. Why Health Checks Matter

Health checks let orchestrators distinguish between "the container process is running"
and "the application is actually working."

**Without health checks:**
```
Container starts → Process is running → Orchestrator says "healthy"
                   (but app crashed internally, listening on port but returning 500s)
```

**With health checks:**
```
Container starts → Process is running → Health check: curl /health → 200 OK → Healthy
                                      → Health check: curl /health → 500   → Unhealthy
                                      → Orchestrator restarts container
```

### How our health check works:

```
                    Every 30s
                       │
                       ▼
              ┌─────────────────┐
              │ curl -f         │
              │ /api/v1/admin/  │
              │ health          │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │                 │
          200 OK          Non-200 / timeout
              │                 │
          Reset counter    Increment failure
              │                 │
          Mark healthy     Failures >= 3?
                               │
                    ┌──────────┴──────────┐
                    │                     │
                   No                    Yes
                    │                     │
                Keep checking      Mark UNHEALTHY
                                         │
                              Docker restarts container
```

The health endpoint (`/api/v1/admin/health`) checks:
1. **Database connectivity** — Can we reach PostgreSQL?
2. **Redis connectivity** — Can we reach Redis?
3. **ChromaDB status** — Is the vector store operational?

If any backing service is down, the health check reflects it, and the orchestrator
can take action (restart, route traffic away, alert).

### The `start_period` parameter:

```
Container start ──────── 40s ──────── Health checks begin
                  start_period         │
                                      30s interval
                                       │
                               First real check
```

`--start-period=40s` gives the application 40 seconds to initialise before health
checks count. During this window, failures are ignored. This is critical because:
- uvicorn takes ~2-5 seconds to import the FastAPI app
- SQLAlchemy connection pool takes ~2-3 seconds to establish
- ChromaDB initialization takes ~5-10 seconds
- Tesseract model loading takes ~3-5 seconds

Without `start_period`, the container would be marked unhealthy and restarted before
it finishes starting, creating an infinite restart loop.

---

## 7. CI/CD Pipeline Step by Step

```
Push to main
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
  ┌──────┐       ┌───────────┐      ┌───────────┐
  │ Lint │       │ Type Check│      │Unit Tests │  ← Stage 1: Parallel quality gates
  │(ruff)│       │  (mypy)   │      │ (pytest)  │     ~1 minute total
  └──┬───┘       └─────┬─────┘      └─────┬─────┘
     │                 │                   │
     └────────┬────────┴───────────────────┘
              │
              ▼          ALL three must pass
     ┌─────────────────┐
     │Integration Tests│  ← Stage 2: Full-stack testing
     │(Docker + pytest)│     ~2 minutes
     └────────┬────────┘
              │
       ┌──────┴──────┐
       ▼              ▼
┌────────────┐  ┌───────────┐
│ Evaluation │  │Build+Push │  ← Stage 3: Parallel
│ Benchmark  │  │  Docker   │     ~3 minutes
│(optional)  │  │  Image    │
└────────────┘  └─────┬─────┘
                      │
                      ▼
               ┌──────────────┐
               │   Deploy to  │  ← Stage 4: Production deploy
               │  Cloud Run   │     ~2 minutes
               └──────────────┘

Total: ~5-6 minutes from push to production
```

### Why this order?

1. **Lint first (fast fail):** Ruff runs in <10 seconds. If you have a syntax error,
   there's no point running tests that will fail on import.

2. **Type check and tests parallel with lint:** They take ~45 seconds each. Running
   them simultaneously means Stage 1 completes in ~45 seconds, not ~2 minutes.

3. **Integration tests after all quality checks:** These require Docker (slower,
   more expensive). Only run them on code that has already passed basic quality gates.

4. **Evaluation and build in parallel:** Evaluation calls the Gemini API (slow).
   Building a Docker image is also slow. Neither depends on the other, so run them
   simultaneously.

5. **Deploy last:** Only deploy after we have a confirmed-working image. If any
   previous stage fails, we never deploy broken code.

---

## 8. Why Evaluation Runs in CI

The evaluation benchmark (`scripts/run_eval.py`) measures:
- **Retrieval precision/recall** — Are we finding the right document chunks?
- **Extraction accuracy** — Are we correctly extracting metadata (authors, dates)?
- **QA faithfulness** — Are LLM answers grounded in source documents?
- **Latency** — Is the system fast enough?

### This is a quality gate:

```
Version N (deployed):     Retrieval precision = 0.85
                          QA faithfulness = 0.92

Developer changes the chunking algorithm

Version N+1 (candidate):  Retrieval precision = 0.62   ← REGRESSION
                           QA faithfulness = 0.88       ← REGRESSION

CI evaluation catches the regression → developer is alerted
```

Without evaluation in CI, you'd only notice the regression when users complain that
answers are suddenly wrong. By running the benchmark automatically, you catch quality
regressions before they reach production.

### Why `continue-on-error: true`?

The evaluation requires:
1. A `GEMINI_API_KEY` secret configured in GitHub
2. Indexed documents to query against

In a fresh CI environment or a forked repository, these may not exist. We don't want
a missing API key to block deployment of a legitimate bug fix. The evaluation report
is uploaded as an artifact for manual review.

In a mature setup, you would:
1. Maintain a fixture dataset in the test suite
2. Set thresholds (e.g., precision must be > 0.80)
3. Remove `continue-on-error` so the pipeline actually blocks on regressions

---

## 9. GCP Deployment Walkthrough

The `deploy_gcp.sh` script provisions the entire infrastructure:

```
┌─────────────────────────────────────────────────────┐
│                    Google Cloud                      │
│                                                     │
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │   Secret     │    │     Artifact Registry     │   │
│  │   Manager    │    │  ┌────────────────────┐  │   │
│  │ ┌─────────┐  │    │  │ research-processor │  │   │
│  │ │API KEY  │──│────│──│    :latest          │  │   │
│  │ │DB PASS  │  │    │  │    :<sha>           │  │   │
│  │ └─────────┘  │    │  └─────────┬──────────┘  │   │
│  └─────────────┘    └────────────│──────────────┘   │
│                                  │                   │
│  ┌───────────────────────────────▼─────────────────┐│
│  │              Cloud Run                           ││
│  │  ┌──────────────────────────────────────────┐   ││
│  │  │ research-processor (0-4 instances)        │   ││
│  │  │   Port 8000                               │   ││
│  │  │   2 vCPU, 2GB RAM                         │   ││
│  │  │   Scale to zero when idle                 │   ││
│  │  └──────┬─────────────────────┬──────────────┘   ││
│  └─────────│─────────────────────│──────────────────┘│
│            │ Cloud SQL Proxy     │                    │
│            ▼ (unix socket)       ▼                    │
│  ┌─────────────────┐   ┌──────────────────┐         │
│  │   Cloud SQL      │   │  Cloud Storage   │         │
│  │   PostgreSQL 16  │   │  gs://...-uploads│         │
│  │   db-f1-micro    │   │  30-day lifecycle│         │
│  │   No public IP   │   │                  │         │
│  └─────────────────┘   └──────────────────┘         │
└─────────────────────────────────────────────────────┘
```

### What each `gcloud` command does:

| Step | Command | What It Does | Why |
|------|---------|-------------|-----|
| 1 | `gcloud config set project` | Sets the active project for all subsequent commands | Avoids passing `--project` to every command |
| 2 | `gcloud services enable` | Turns on Cloud Run, Cloud SQL, Storage, Artifact Registry, Secret Manager APIs | APIs are disabled by default; must be enabled before use |
| 3 | `gcloud sql instances create` | Provisions a PostgreSQL 16 server on `db-f1-micro` | The platform needs a relational database for document metadata |
| 4a | `gcloud sql databases create` | Creates the `intelligent_doc` database | Separates our data from the default `postgres` database |
| 4b | `gcloud sql users create` | Creates the `docuser` with a random password | Principle of least privilege — don't use the superuser `postgres` account |
| 4c | `gcloud secrets create db-password` | Stores the generated password in Secret Manager | Secrets should never be in code, env files, or CI logs |
| 5a | `gcloud storage buckets create` | Creates a GCS bucket for PDF uploads | Cloud Run has ephemeral storage — files disappear on restart |
| 5b | `gcloud storage buckets update --lifecycle-file` | Sets 30-day auto-delete policy | Prevents unbounded storage cost growth |
| 6a | `gcloud artifacts repositories create` | Creates a Docker image repository | Needed before you can push images |
| 6b | `gcloud auth configure-docker` | Configures Docker to authenticate with Artifact Registry | `docker push` needs credentials for private registries |
| 6c | `docker build` + `docker push` | Builds the multi-stage image and pushes it | Cloud Run pulls the image from Artifact Registry |
| 7 | `gcloud secrets create/versions add` | Stores GEMINI_API_KEY and DATABASE_URL | Cloud Run mounts secrets at startup via `--set-secrets` |
| 8 | `gcloud run deploy` | Deploys the container to Cloud Run with all config | This is the actual deployment — everything before was infrastructure setup |
| 9 | `gcloud run services describe` | Retrieves the auto-generated HTTPS URL | Cloud Run generates a unique `*.run.app` URL |

---

## 10. Cost Breakdown

### Monthly cost for a low-traffic deployment (~100 requests/day):

| Service | Configuration | Monthly Cost | Notes |
|---------|--------------|-------------|-------|
| **Cloud Run** | min-instances=0, max=4, 2 vCPU, 2GB | **~$0-2** | Scale-to-zero: $0 when idle. Free tier covers 2M requests/month. At 100 req/day (~3K/month), you stay well within the free tier. |
| **Cloud SQL** | db-f1-micro, 10GB HDD | **~$7.67** | Cheapest managed PostgreSQL. Free for first 12 months under GCP free tier. Shared CPU, 0.6 GB RAM. |
| **Cloud Storage** | Standard, 30-day lifecycle | **~$0.02** | At ~1 GB of PDFs: $0.02/GB. Lifecycle auto-deletes after 30 days. |
| **Artifact Registry** | Image storage | **~$0.04** | ~400 MB image at $0.10/GB. |
| **Secret Manager** | 3 secrets | **~$0.00** | First 10,000 access operations/month are free. |
| **Gemini Flash API** | ~3K requests/month | **~$0.50** | $0.075/1M input tokens. With caching, most repeated queries hit Redis instead of calling Gemini. |
| **Networking** | Egress within same region | **~$0.00** | Intra-region traffic is free. Cloud Run ↔ Cloud SQL ↔ Storage all in us-central1. |
| | | | |
| **Total (Year 1)** | | **~$1-3/month** | Cloud SQL is free for 12 months under GCP free tier. |
| **Total (After Year 1)** | | **~$8-12/month** | Cloud SQL becomes the main cost (~$7.67). |

### Cost comparison with alternatives:

| Approach | Monthly Cost | Notes |
|----------|-------------|-------|
| **This architecture (Cloud Run + Cloud SQL)** | $8-12 | Optimised for low traffic |
| **GKE Autopilot** | $70+ | Minimum cluster cost even when idle |
| **Compute Engine (single VM)** | $25-50 | Always-on, no scale-to-zero |
| **App Engine Standard** | $0-10 | Similar to Cloud Run but less flexible |
| **Heroku** | $5-25 | Simpler but less control |
| **AWS (ECS + RDS)** | $15-30 | RDS db.t3.micro ~$15/month |

---

## 11. Why Cloud Run Over GKE, App Engine, or Compute Engine

### Cloud Run vs GKE (Google Kubernetes Engine)

| Factor | Cloud Run | GKE |
|--------|-----------|-----|
| **Cost at low traffic** | ~$0 (scale to zero) | ~$70/month (minimum cluster) |
| **Operational overhead** | Zero — Google manages everything | High — node pools, networking, RBAC, upgrades |
| **Scaling** | Automatic, per-request | Automatic, but requires HPA configuration |
| **Cold starts** | 1-5 seconds | None (pods always running) |
| **When to use** | Stateless HTTP services, <=4 vCPU | Stateful workloads, complex networking, multi-service meshes |

**Our choice: Cloud Run.** This is a stateless HTTP API. We don't need persistent
connections, custom networking, or sidecar containers. Cloud Run gives us
auto-scaling and scale-to-zero without any cluster management.

### Cloud Run vs App Engine

| Factor | Cloud Run | App Engine Standard |
|--------|-----------|-------------------|
| **Container control** | Full Dockerfile control | Buildpack-based, limited runtime choices |
| **Port binding** | Any port | Port 8080 only |
| **Local testing** | `docker run` works identically | `dev_appserver.py` has compatibility issues |
| **Pricing** | Per-request, per-second | Per-instance-hour |
| **Scaling speed** | <1 second | 5-10 seconds |

**Our choice: Cloud Run.** We need Tesseract and poppler-utils installed in the
container, which requires a custom Dockerfile. App Engine Standard doesn't support
custom system dependencies well. App Engine Flexible does, but it doesn't scale to
zero.

### Cloud Run vs Compute Engine

| Factor | Cloud Run | Compute Engine |
|--------|-----------|---------------|
| **Cost when idle** | $0 | $5-50/month (VM always running) |
| **Scaling** | Automatic (0 to N instances) | Manual or with managed instance groups |
| **Maintenance** | Zero (no OS patches, no SSH) | You manage the OS, packages, firewalls |
| **Deploy** | `gcloud run deploy` (30 seconds) | SSH, pull code, restart services (5+ minutes) |

**Our choice: Cloud Run.** Compute Engine is the right choice for long-running
stateful services (databases, message queues). For a stateless HTTP API, Cloud Run
is simpler, cheaper, and more scalable.

---

## 12. Why min-instances=0 Saves Money

```
Cloud Run pricing:
  vCPU:    $0.00002400 / vCPU-second
  Memory:  $0.00000250 / GiB-second

With min-instances=1 (always on):
  1 instance × 2 vCPU × 86400 s/day × 30 days = 5,184,000 vCPU-seconds
  Cost: 5,184,000 × $0.00002400 = $124.42/month just for CPU

With min-instances=0 (scale to zero):
  At 100 requests/day, ~0.5s each = 50 seconds of compute/day
  50s × 2 vCPU × 30 days = 3,000 vCPU-seconds
  Cost: 3,000 × $0.00002400 = $0.07/month

  Savings: $124.35/month (99.9%)
```

**The tradeoff is cold starts.** When an instance scales from 0 to 1, Cloud Run must:
1. Pull the container image (if not cached): ~2-4 seconds
2. Start the container: ~1-2 seconds
3. Run the application startup (imports, DB connections): ~3-5 seconds

Total cold start: **~5-10 seconds** for the first request after a period of inactivity.

For our use case (research paper processing), users upload a document and wait for
analysis. A 5-second cold start on the first request is acceptable because:
1. Document processing itself takes 10-30 seconds (OCR, embedding, summarization).
2. Users don't expect instant responses from a document processing tool.
3. Subsequent requests hit a warm instance (no cold start) as long as traffic
   continues within ~15 minutes.

If you needed <100ms latency (e.g., a real-time chat API), you'd set
`min-instances=1` and accept the ~$125/month cost.

---

## 13. Interview Questions and Ideal Answers

### "How do you deploy your application?"

> We use a fully automated CI/CD pipeline through GitHub Actions. When code is merged
> to main, the pipeline runs lint, type checking, and unit tests in parallel as a
> first quality gate. If all three pass, integration tests run against real Docker
> containers. Then, the multi-stage Dockerfile builds a production image — about 400 MB
> with only runtime dependencies — and pushes it to Google Artifact Registry, tagged
> with the Git commit SHA for immutable traceability. Finally, `gcloud run deploy`
> deploys that exact image to Cloud Run with scale-to-zero configuration.
>
> For initial infrastructure setup, we have an idempotent bash script
> (`deploy_gcp.sh`) that provisions Cloud SQL, Cloud Storage with a 30-day lifecycle
> policy, Artifact Registry, and Secret Manager. It uses `set -euo pipefail` and
> `|| echo "continuing"` patterns so it's safe to re-run.
>
> Secrets — the Gemini API key and database URL — are stored in GCP Secret Manager
> and mounted into Cloud Run at startup. They never appear in code, environment files,
> or CI logs.

### "What happens when you push to main?"

> Five things happen in sequence, with the first three running in parallel:
>
> 1. **Ruff** checks for style violations and common bugs (~10 seconds).
> 2. **Mypy** verifies type correctness across the entire codebase (~45 seconds).
> 3. **Pytest** runs 411 unit tests with coverage reporting (~30 seconds).
>
> All three must pass. If any fails, the pipeline stops immediately.
>
> 4. **Integration tests** spin up PostgreSQL and Redis in Docker, run tests against
>    real services, and tear down. This catches issues that unit tests with mocks
>    can't — like incorrect SQL queries or Redis serialization bugs.
>
> 5. **Build and deploy** happen in parallel with an evaluation benchmark. The Docker
>    image is built using a multi-stage Dockerfile (discarding the compiler and dev
>    tools), pushed to Artifact Registry with both a SHA tag and `latest`, and deployed
>    to Cloud Run. The SHA tag ensures we deploy exactly what was tested — no race
>    conditions even if two pushes happen simultaneously.
>
> Total time from push to production: about 5-6 minutes.

### "How do you ensure quality doesn't degrade with new deployments?"

> We have multiple layers of quality assurance:
>
> **Static analysis:** Ruff catches style issues, unused imports, and common
> anti-patterns. Mypy catches type errors — if someone changes a function signature,
> mypy flags every call site that's now wrong.
>
> **411 unit tests** cover every component in isolation: PDF parsing, text chunking,
> metadata extraction, hybrid retrieval, QA generation, cost tracking, rate limiting,
> authentication, and all API endpoints. They run in ~4 seconds because they use
> mocks for external services.
>
> **Integration tests** verify the full stack works together with real PostgreSQL and
> Redis containers.
>
> **Evaluation benchmarks** measure retrieval precision, recall, MRR, and NDCG; metadata
> extraction accuracy; and LLM-as-judge faithfulness scoring. These metrics are
> uploaded as CI artifacts so we can track them over time and catch regressions in
> AI quality that unit tests can't detect — for example, a chunking algorithm change
> that passes all unit tests but degrades retrieval precision from 0.85 to 0.62.
>
> **Docker health checks** in production continuously verify the app and all backing
> services (PostgreSQL, Redis, ChromaDB) are healthy. If the health endpoint returns
> non-200 three times in a row, the container is automatically restarted.
>
> **Coverage reporting** on pull requests posts a comment showing exactly which lines
> are covered, making it visible when new code lacks tests.

### "How much does it cost to run this in production?"

> For a low-traffic deployment — say 100 requests per day — about **$1-3 per month**
> in the first year, rising to **$8-12 per month** after the GCP free tier expires.
>
> The main cost driver is Cloud SQL at $7.67/month for the smallest PostgreSQL
> instance. Everything else is nearly free at low traffic:
>
> - **Cloud Run** scales to zero, so we pay nothing when nobody is using it. At 100
>   requests/day, compute costs are about $0.07/month.
> - **Gemini Flash** at $0.075 per million input tokens is extremely cheap. With Redis
>   caching for repeated queries, most requests never hit the LLM.
> - **Cloud Storage** with a 30-day lifecycle auto-deletes old uploads, keeping
>   storage costs under $0.05/month.
>
> If costs were a concern, the biggest optimisation would be replacing Cloud SQL with
> SQLite for a single-user deployment, which would bring the total to under $1/month.
> But for a portfolio project demonstrating production architecture, Cloud SQL shows
> you know how to work with managed databases.

### "Why Docker? Why multi-stage?"

> **Why Docker at all?** The application depends on system-level tools — Tesseract
> for OCR, poppler-utils for PDF rendering, and specific PostgreSQL client libraries.
> "Works on my machine" is a real problem: macOS has different library paths, Ubuntu
> 22.04 ships Tesseract 5 but 24.04 ships Tesseract 5.3 with different defaults, and
> Alpine's musl libc breaks some Python packages. Docker gives us a reproducible
> environment: the same image runs identically on my laptop, in CI, and on Cloud Run.
>
> **Why multi-stage?** Three reasons, in order of importance:
>
> 1. **Security:** The production image has no compiler (gcc), no package manager cache,
>    and no dev tools. If an attacker exploits a vulnerability and gets shell access,
>    they can't compile exploit code or install tools. The attack surface is minimised.
>
> 2. **Image size:** Our multi-stage image is about 400 MB versus ~900 MB for a
>    single-stage build. Smaller images mean faster deploys (Cloud Run pulls the image
>    on every cold start), lower storage costs, and faster CI pipelines.
>
> 3. **Separation of concerns:** Build-time dependencies (gcc, libpq-dev, pip cache)
>    are conceptually different from runtime dependencies (tesseract, libpq5, curl).
>    Multi-stage makes this separation explicit in the Dockerfile itself.
>
> The key technique is the `/opt/venv` pattern: Stage 1 creates a virtual environment,
> installs all Python packages into it, and Stage 2 copies that single directory. This
> cleanly transfers all compiled packages without bringing along the compiler.

### "How would you scale this to handle 1000 users?"

> The architecture is already designed for horizontal scaling, so 1000 users doesn't
> require architectural changes — just configuration tuning:
>
> **Cloud Run auto-scaling** handles the web tier. Set `--max-instances` from 4 to 20.
> Cloud Run spins up new instances automatically based on request concurrency. Each
> instance handles ~80 concurrent requests (the default), so 20 instances handle
> 1,600 concurrent users.
>
> **Cloud SQL** would need an upgrade from `db-f1-micro` (0.6 GB RAM) to
> `db-custom-2-7680` (2 vCPU, 7.5 GB RAM, ~$50/month). Add read replicas if read
> traffic is the bottleneck.
>
> **Celery workers** would scale by increasing the number of worker containers. Each
> worker runs 2 concurrent tasks, so 5 workers handle 10 simultaneous PDF processing
> jobs. For Cloud Run, I'd move background processing to Cloud Tasks or Pub/Sub +
> Cloud Run Jobs instead of Celery, since Cloud Run is stateless.
>
> **Redis** would upgrade from the in-container instance to Cloud Memorystore
> (~$35/month for 1 GB), which provides managed failover and monitoring.
>
> **CDN** — Adding Cloud CDN in front of Cloud Run caches static responses (health
> checks, API docs) and reduces origin requests.
>
> **What I would NOT do:** Move to Kubernetes. GKE Autopilot costs $70+/month minimum
> and adds significant operational complexity. Cloud Run auto-scales to 1000 instances
> — far more than 1000 users need. Kubernetes becomes worthwhile at ~10,000+ users
> when you need fine-grained pod scheduling, service mesh, or persistent workloads.
>
> The bottleneck at 1000 users is most likely the Gemini API rate limit, not our
> infrastructure. I'd address this with aggressive Redis caching (cache embeddings
> and LLM responses for identical queries) and request deduplication.

---

*Phase 7 delivers a production-ready container, automated CI/CD, and a GCP deployment
that costs under $12/month. Every choice — from the multi-stage Dockerfile to
scale-to-zero Cloud Run — optimises for security, cost, and operational simplicity.*
