#!/usr/bin/env bash
# scripts/build_and_push.sh
# Build the indexing_pipeline Docker image and optionally push to Docker Hub or ECR.
# Configure behaviour using environment variables (no CLI args).
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-civic-indexing}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_CONTEXT="${BUILD_CONTEXT:-./indexing_pipeline}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${BUILD_CONTEXT}/Dockerfile}"
IMAGE_LOCAL="${IMAGE_NAME}:${IMAGE_TAG}"

OCR_LANGUAGES="${OCR_LANGUAGES:-eng,tam,hin}"
INDIC_OCR_SIZE="${INDIC_OCR_SIZE:-best}"

PUSH="${PUSH:-true}"                    # true|false
ECR_REGISTRY="${ECR_REGISTRY:-false}"   # true => push to ECR
ECR_REPO="${ECR_REPO:-${IMAGE_NAME}}"
AWS_REGION="${AWS_REGION:-}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"

SMOKE_TEST="${SMOKE_TEST:-false}"
HEALTH_PORT="${HEALTH_PORT:-8080}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-60}"

RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_DELAY="${RETRY_DELAY:-2}"

CONTAINER_NAME="${CONTAINER_NAME:-build-smoke-${IMAGE_NAME//[^a-zA-Z0-9]/}_$$}"

# -------------------------
# Helpers
# -------------------------
log(){ printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn(){ printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

normalize_bool(){ local v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"; case "$v" in 1|true|yes|y) echo true ;; *) echo false ;; esac; }
PUSH="$(normalize_bool "${PUSH}")"
ECR_REGISTRY="$(normalize_bool "${ECR_REGISTRY}")"
SMOKE_TEST="$(normalize_bool "${SMOKE_TEST}")"

retry_cmd(){
  local attempts=0 rc=0
  while :; do
    attempts=$((attempts+1))
    "$@" && { rc=0; break; } || rc=$?
    if [ "$attempts" -ge "$RETRY_ATTEMPTS" ]; then break; fi
    sleep $((RETRY_DELAY ** (attempts - 1)))
  done
  return $rc
}

cleanup_container(){
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}
trap 'cleanup_container' EXIT

# -------------------------
# Preconditions
# -------------------------
if ! command -v docker >/dev/null 2>&1; then err "docker CLI not found"; exit 2; fi
if [ ! -f "${DOCKERFILE_PATH}" ]; then err "Dockerfile not found at ${DOCKERFILE_PATH}"; exit 3; fi

log "Building ${IMAGE_LOCAL}"
log "OCR_LANGUAGES=${OCR_LANGUAGES} INDIC_OCR_SIZE=${INDIC_OCR_SIZE}"
log "PUSH=${PUSH} ECR_REGISTRY=${ECR_REGISTRY}"

# -------------------------
# Build (no buildx)
# -------------------------
docker build \
  --build-arg OCR_LANGUAGES="${OCR_LANGUAGES}" \
  --build-arg INDIC_OCR_SIZE="${INDIC_OCR_SIZE}" \
  -t "${IMAGE_LOCAL}" \
  "${BUILD_CONTEXT}" \
|| { err "docker build failed"; exit 4; }

log "Built ${IMAGE_LOCAL}"

# -------------------------
# Optional smoke test
# -------------------------
if [ "${SMOKE_TEST}" = "true" ]; then
  log "Starting smoke test container ${CONTAINER_NAME}"
  cleanup_container
  docker run --rm -d --name "${CONTAINER_NAME}" -p "${HEALTH_PORT}:${CONTAINER_PORT}" --shm-size=1g "${IMAGE_LOCAL}" >/dev/null
  start_ts=$(date +%s)
  ok=false
  while :; do
    if curl -fsS --max-time 2 "http://127.0.0.1:${HEALTH_PORT}${HEALTH_PATH}" >/dev/null 2>&1; then
      ok=true; break
    fi
    now=$(date +%s)
    if [ $((now-start_ts)) -ge "${SMOKE_TIMEOUT}" ]; then break; fi
    sleep 1
  done

  if [ "${ok}" != "true" ]; then
    docker logs --tail 200 "${CONTAINER_NAME}" || true
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    err "Smoke test failed (health endpoint did not respond within ${SMOKE_TIMEOUT}s)"
    exit 5
  fi

  log "Smoke test passed; stopping container"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# If not pushing, exit
if [ "${PUSH}" != "true" ]; then
  log "PUSH is false; image is available locally as ${IMAGE_LOCAL}"
  exit 0
fi

# -------------------------
# Push to ECR
# -------------------------
if [ "${ECR_REGISTRY}" = "true" ]; then
  if ! command -v aws >/dev/null 2>&1; then
    err "aws CLI not found; required for ECR push"; exit 6
  fi
  if [ -z "${AWS_REGION:-}" ]; then
    AWS_REGION="$(aws configure get region || true)"
  fi
  if [ -z "${AWS_REGION:-}" ]; then err "AWS_REGION required for ECR push"; exit 7; fi

  AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
  if [ -z "${AWS_ACCOUNT_ID}" ]; then err "Unable to resolve AWS account id (STS failed)"; exit 8; fi

  ECR_REGISTRY_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  TARGET_IMAGE="${ECR_REGISTRY_URL}/${ECR_REPO}:${IMAGE_TAG}"

  # create repository if missing (idempotent)
  if ! aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "ECR repo ${ECR_REPO} not found; creating..."
    aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null
  else
    log "ECR repo ${ECR_REPO} exists"
  fi

  log "Logging into ECR ${ECR_REGISTRY_URL}"
  aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY_URL}" || { err "docker login to ECR failed"; exit 9; }

  docker tag "${IMAGE_LOCAL}" "${TARGET_IMAGE}"
  log "Pushing ${TARGET_IMAGE}"
  if ! retry_cmd docker push "${TARGET_IMAGE}"; then err "docker push to ECR failed"; exit 10; fi

  log "Push to ECR succeeded: ${TARGET_IMAGE}"
  exit 0
fi

# -------------------------
# Push to Docker Hub (fallback)
# -------------------------
if [ -z "${DOCKER_USERNAME:-}" ]; then
  warn "DOCKER_USERNAME not set; skipping push. Local image: ${IMAGE_LOCAL}"
  exit 0
fi

if [ -n "${DOCKER_PASSWORD:-}" ]; then
  log "Logging into Docker registry as ${DOCKER_USERNAME}"
  printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "Docker login failed"; exit 11; }
else
  warn "DOCKER_PASSWORD not set; attempting push without login may fail"
fi

TARGET_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
docker tag "${IMAGE_LOCAL}" "${TARGET_IMAGE}"
log "Pushing ${TARGET_IMAGE}"
if ! retry_cmd docker push "${TARGET_IMAGE}"; then err "docker push to Docker Hub failed"; exit 12; fi

log "Push to Docker Hub succeeded: ${TARGET_IMAGE}"
exit 0
