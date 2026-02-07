#!/usr/bin/env bash

IMAGE="athithya5354/civic-indexing:amd64-arm64-v11"
NETWORK="civic-net"
PG_CONTAINER="pgvector"
PG_VOLUME="pgvector-data"

echo "==> Pulling indexing image (idempotent)"
docker pull "$IMAGE"

echo "==> Creating docker network (idempotent)"
docker network create "$NETWORK" >/dev/null 2>&1 || true

echo "==> Removing any existing pgvector container (idempotent)"
docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 || true

echo "==> Starting pgvector (background, exposed to host)"
docker run -d \
  --name "$PG_CONTAINER" \
  --network "$NETWORK" \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=postgres \
  -v "$PG_VOLUME":/var/lib/postgresql/data \
  ankane/pgvector:v0.5.1

echo "==> Waiting for Postgres to accept connections"
for i in {1..30}; do
  if docker exec "$PG_CONTAINER" pg_isready -U postgres >/dev/null 2>&1; then
    echo "    Postgres is ready"
    break
  fi
  sleep 1
done

echo "==> Running civic indexing pipeline (one-shot, foreground)"
docker run --rm \
  --network "$NETWORK" \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?missing}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?missing}" \
  -e AWS_REGION="ap-south-1" \
  -e S3_BUCKET="civic-data-raw-prod" \
  -e SEED_URLS="https://www.india.gov.in/my-government/schemes,https://www.myscheme.gov.in,https://csc.gov.in" \
  -e ALLOWED_DOMAINS="india.gov.in,myscheme.gov.in,csc.gov.in" \
  -e PG_HOST="pgvector" \
  -e PG_PORT="5432" \
  -e PG_DB="postgres" \
  -e PG_USER="postgres" \
  -e PG_PASSWORD="postgres" \
  "$IMAGE"

echo "==> Indexing completed successfully"
echo "==> pgvector is running and accessible at localhost:5432"


export PG_HOST=localhost
export PG_PORT=5432
export PG_DB=postgres
export PG_USER=postgres
export PG_PASSWORD=postgres
export AWS_REGION=ap-south-1

python3 -m inference_pipeline.core.query
