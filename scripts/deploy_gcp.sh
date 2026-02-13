#!/usr/bin/env bash
# =============================================================================
# deploy_gcp.sh — Deploy Intelligent Document Processing to Google Cloud
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker installed locally
#   - A GCP project with billing enabled
#
# Usage:
#   export GCP_PROJECT_ID=my-project-id
#   export GEMINI_API_KEY=your-api-key
#   bash scripts/deploy_gcp.sh
#
# Cost Estimates (as of 2026):
#   - Cloud Run:          Free tier covers 2M requests/month; scale-to-zero
#                         means $0 when idle. ~$0.00002400/vCPU-second.
#   - Cloud SQL:          db-f1-micro ~$7.67/month (free tier for 12 months).
#                         Shared core, 0.6 GB RAM — fine for dev/staging.
#   - Cloud Storage:      $0.020/GB/month (Standard). 30-day lifecycle
#                         auto-deletes old uploads to keep costs low.
#   - Artifact Registry:  $0.10/GB/month for image storage.
#   - Gemini Flash:       $0.075 per 1M input tokens — very cheap.
#                         Aggressive caching (Redis) reduces API calls further.
#   - Total estimated:    ~$10-20/month for low-traffic usage.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — change these for your project
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="us-central1"
SERVICE_NAME="research-processor"
DB_INSTANCE="research-db"
DB_NAME="intelligent_doc"
DB_USER="docuser"
BUCKET_NAME="${PROJECT_ID}-research-uploads"
REPO_NAME="research-processor"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

echo "=== Deploying to project: ${PROJECT_ID} ==="
echo "=== Region: ${REGION} ==="

# ---------------------------------------------------------------------------
# Step 1: Set the active GCP project
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Setting GCP project..."
gcloud config set project "${PROJECT_ID}"

# ---------------------------------------------------------------------------
# Step 2: Enable required APIs
# Cost: Free to enable — you only pay for actual resource usage.
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Enabling GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

# ---------------------------------------------------------------------------
# Step 3: Create Cloud SQL PostgreSQL instance
# Cost: db-f1-micro is the cheapest tier (~$7.67/month).
#        Free for first 12 months under GCP free tier.
#        Use db-f1-micro for dev/staging; db-custom-1-3840 for production.
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Creating Cloud SQL instance (db-f1-micro)..."
echo "    This takes 5-10 minutes on first creation."
gcloud sql instances create "${DB_INSTANCE}" \
    --database-version=POSTGRES_16 \
    --tier=db-f1-micro \
    --region="${REGION}" \
    --storage-type=HDD \
    --storage-size=10GB \
    --no-assign-ip \
    --network=default \
    || echo "    Instance may already exist, continuing..."

# ---------------------------------------------------------------------------
# Step 4: Create database and user
# The || pattern makes the script idempotent (safe to re-run).
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 4: Creating database and user..."
gcloud sql databases create "${DB_NAME}" \
    --instance="${DB_INSTANCE}" \
    || echo "    Database may already exist, continuing..."

# Generate a random password for production (never hardcode passwords)
DB_PASSWORD=$(openssl rand -base64 24)
gcloud sql users create "${DB_USER}" \
    --instance="${DB_INSTANCE}" \
    --password="${DB_PASSWORD}" \
    || echo "    User may already exist, continuing..."

echo "    Database password generated. Storing in Secret Manager..."

# Store password in Secret Manager for secure access
echo -n "${DB_PASSWORD}" | gcloud secrets create db-password --data-file=- 2>/dev/null \
    || echo -n "${DB_PASSWORD}" | gcloud secrets versions add db-password --data-file=-

# ---------------------------------------------------------------------------
# Step 5: Create Cloud Storage bucket with lifecycle policy
# Cost: Standard storage $0.020/GB/month.
#        30-day lifecycle auto-deletes old uploads to save costs.
#        This prevents unbounded storage growth from uploaded PDFs.
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 5: Creating Cloud Storage bucket..."
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    || echo "    Bucket may already exist, continuing..."

# Set 30-day lifecycle rule to auto-delete old files
cat > /tmp/lifecycle.json <<'LIFECYCLE'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
LIFECYCLE
gcloud storage buckets update "gs://${BUCKET_NAME}" \
    --lifecycle-file=/tmp/lifecycle.json
rm -f /tmp/lifecycle.json
echo "    30-day lifecycle policy applied (cost optimisation)."

# ---------------------------------------------------------------------------
# Step 6: Create Artifact Registry repo and build/push Docker image
# Cost: $0.10/GB/month for image storage.
#        Keep only a few image tags to minimise storage costs.
#        Uses Artifact Registry (not deprecated Container Registry).
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 6: Building and pushing Docker image..."
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Research processor Docker images" \
    || echo "    Repository may already exist, continuing..."

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE}:latest" .
docker push "${IMAGE}:latest"

echo "    Image pushed: ${IMAGE}:latest"

# ---------------------------------------------------------------------------
# Step 7: Store secrets in Secret Manager
# Secrets are mounted into Cloud Run at deploy time via --set-secrets.
# This avoids storing sensitive values in environment variables or code.
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 7: Storing secrets in Secret Manager..."

# Gemini API key
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "    WARNING: GEMINI_API_KEY not set in environment."
    echo "    Run: echo -n 'YOUR_KEY' | gcloud secrets create gemini-api-key --data-file=-"
else
    echo -n "${GEMINI_API_KEY}" | gcloud secrets create gemini-api-key --data-file=- 2>/dev/null \
        || echo -n "${GEMINI_API_KEY}" | gcloud secrets versions add gemini-api-key --data-file=-
    echo "    Stored GEMINI_API_KEY in Secret Manager."
fi

# Database URL (constructed from Cloud SQL connection name)
CLOUD_SQL_CONNECTION="${PROJECT_ID}:${REGION}:${DB_INSTANCE}"
DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@/${DB_NAME}?host=/cloudsql/${CLOUD_SQL_CONNECTION}"
echo -n "${DATABASE_URL}" | gcloud secrets create database-url --data-file=- 2>/dev/null \
    || echo -n "${DATABASE_URL}" | gcloud secrets versions add database-url --data-file=-
echo "    Stored DATABASE_URL in Secret Manager."

# ---------------------------------------------------------------------------
# Step 8: Deploy to Cloud Run
# Cost: --min-instances 0 means ZERO cost when idle (scale-to-zero).
#        2 vCPU + 2Gi RAM per instance when running.
#        --max-instances 4 caps costs during traffic spikes.
#        Cloud Run auto-scales based on request concurrency.
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 8: Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}:latest" \
    --region "${REGION}" \
    --platform managed \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 4 \
    --allow-unauthenticated \
    --add-cloudsql-instances "${CLOUD_SQL_CONNECTION}" \
    --set-env-vars "PYTHONPATH=/app,UPLOAD_DIR=/app/data/uploads,CHROMA_PERSIST_DIR=/app/data/chroma" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,DATABASE_URL=database-url:latest" \
    --port 8000

# ---------------------------------------------------------------------------
# Step 9: Print the deployed URL
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo ">>> Deployment complete!"
echo "=========================================="
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)")
echo ""
echo "  Service URL:   ${SERVICE_URL}"
echo "  Health check:  ${SERVICE_URL}/api/v1/admin/health"
echo "  API docs:      ${SERVICE_URL}/docs"
echo ""
echo "=== Cost optimisation tips ==="
echo "  - Cloud Run min-instances=0 means zero cost when idle"
echo "  - Cloud SQL db-f1-micro is free for 12 months (~\$7.67/month after)"
echo "  - Cloud Storage 30-day lifecycle auto-deletes old uploads"
echo "  - Gemini Flash is very cheap (\$0.075/1M input tokens)"
echo "  - Redis caching reduces LLM API calls significantly"
echo "  - Set budget alerts: gcloud billing budgets create ..."
echo ""
