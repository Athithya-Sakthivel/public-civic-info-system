#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
warn(){ printf '[%s] WARN: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
err(){ printf '[%s] ERROR: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }

MODE="${1:-}"
PURGE_BUCKET=false
if [ "${2:-}" = "--purge-bucket" ]; then
  PURGE_BUCKET=true
fi

: "${AWS_REGION:=ap-south-1}"
: "${AWS_DEFAULT_REGION:=${AWS_REGION}}"
: "${STACK:=prod}"
: "${PULUMI_S3_BUCKET:=pulumi-backend-516}"
: "${PULUMI_S3_PREFIX:=pulumi}"
: "${PULUMI_DDB_TABLE:=pulumi-state-locks}"
PULUMI_LOGIN_URL="s3://${PULUMI_S3_BUCKET}/${PULUMI_S3_PREFIX}"

# required CLIs
for bin in aws pulumi python3 jq; do
  command -v "$bin" >/dev/null 2>&1 || { err "required binary not found: $bin"; exit 10; }
done

aws sts get-caller-identity >/dev/null 2>&1 || { err "AWS credentials invalid for current environment (aws sts failed)"; exit 11; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}" >/dev/null 2>&1 || true' EXIT

retry_cmd(){
  local tries=${1:-6}; shift
  local delay=${1:-1}; shift
  local i=0 rc=0
  while [ $i -lt "$tries" ]; do
    set +e
    "$@"
    rc=$?
    set -e
    [ $rc -eq 0 ] && return 0
    i=$((i+1))
    sleep $delay
    delay=$((delay*2))
  done
  return $rc
}

ensure_bucket(){
  log "ensure: s3 bucket ${PULUMI_S3_BUCKET}"
  if aws s3api head-bucket --bucket "${PULUMI_S3_BUCKET}" >/dev/null 2>&1; then
    log "s3: bucket exists"
    return 0
  fi
  log "s3: creating bucket ${PULUMI_S3_BUCKET} region=${AWS_REGION}"
  if [ "${AWS_REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${PULUMI_S3_BUCKET}" >/dev/null 2>&1 || true
  else
    aws s3api create-bucket --bucket "${PULUMI_S3_BUCKET}" --create-bucket-configuration LocationConstraint="${AWS_REGION}" >/dev/null 2>&1 || true
  fi
  retry_cmd 8 1 aws s3api head-bucket --bucket "${PULUMI_S3_BUCKET}" >/dev/null 2>&1 || { err "s3: create/visibility failed"; exit 12; }
  aws s3api put-bucket-versioning --bucket "${PULUMI_S3_BUCKET}" --versioning-configuration Status=Enabled >/dev/null 2>&1 || warn "s3: versioning set failed"
  aws s3api put-bucket-encryption --bucket "${PULUMI_S3_BUCKET}" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null 2>&1 || warn "s3: encryption set failed"
  log "s3: created & configured ${PULUMI_S3_BUCKET}"
}

ensure_ddb_table(){
  log "ensure: dynamodb table ${PULUMI_DDB_TABLE}"
  if aws dynamodb describe-table --table-name "${PULUMI_DDB_TABLE}" >/dev/null 2>&1; then
    log "ddb: table exists"
    return 0
  fi

  local attempts=8
  local delay=2
  local last_err=""
  for i in $(seq 1 "${attempts}"); do
    local errf="${TMPDIR}/ddb-create-err-${i}.txt"
    set +e
    aws dynamodb create-table \
      --table-name "${PULUMI_DDB_TABLE}" \
      --attribute-definitions AttributeName=LockID,AttributeType=S \
      --key-schema AttributeName=LockID,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST \
      --region "${AWS_REGION}" 2> "${errf}"
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
      retry_cmd 12 2 aws dynamodb wait table-exists --table-name "${PULUMI_DDB_TABLE}" --region "${AWS_REGION}" >/dev/null 2>&1 || { err "ddb: wait for ACTIVE failed"; exit 13; }
      log "ddb: created and ACTIVE"
      rm -f "${errf}" || true
      return 0
    fi
    last_err="$(cat "${errf}" 2>/dev/null || true)"
    # race detected: another actor created the table
    if echo "${last_err}" | grep -qi "ResourceInUseException"; then
      log "ddb: ResourceInUseException observed; verifying existence"
      if aws dynamodb describe-table --table-name "${PULUMI_DDB_TABLE}" >/dev/null 2>&1; then
        log "ddb: table visible after race; continuing"
        rm -f "${errf}" || true
        return 0
      fi
    fi
    if echo "${last_err}" | grep -qi "AccessDenied\|UnauthorizedOperation"; then
      err "ddb: create failed due to permissions: ${last_err}"
      exit 14
    fi
    if echo "${last_err}" | grep -qi "ThrottlingException\|ProvisionedThroughputExceededException"; then
      warn "ddb: throttled (attempt ${i}/${attempts}), retrying"
      sleep "${delay}"
      delay=$((delay*2))
      continue
    fi
    warn "ddb: create-table attempt ${i} failed; retrying; last_err: ${last_err}"
    sleep "${delay}"
    delay=$((delay*2))
  done

  # final check before failing
  if aws dynamodb describe-table --table-name "${PULUMI_DDB_TABLE}" >/dev/null 2>&1; then
    log "ddb: table exists after retries; proceeding"
    return 0
  fi

  err "ddb: create-table failed after retries; last error: ${last_err}"
  exit 15
}

venv_install_requirements(){
  local venv_dir=".venv"
  if [ ! -d "${venv_dir}" ]; then
    log "venv: creating ${venv_dir}"
    python3 -m venv "${venv_dir}"
  fi
  # shellcheck disable=SC1090
  . "${venv_dir}/bin/activate"
  if [ -f "requirements.txt" ]; then
    log "pip: installing requirements.txt (quiet)"
    python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || warn "pip upgrade failed; continuing"
    python3 -m pip install -q -r requirements.txt >/dev/null 2>&1 || warn "pip install -r requirements.txt returned non-zero; inspect ${venv_dir}"
    log "pip: installed"
  else
    warn "requirements.txt not found; skipping pip install"
  fi
}

pulumi_login_and_select(){
  log "pulumi: login ${PULUMI_LOGIN_URL}"
  export AWS_REGION AWS_DEFAULT_REGION
  export AWS_SDK_LOAD_CONFIG=1
  retry_cmd 6 1 pulumi login "${PULUMI_LOGIN_URL}" >/dev/null 2>&1 || { err "pulumi login failed"; exit 16; }
  if pulumi stack select "${STACK}" >/dev/null 2>&1; then
    log "pulumi: selected ${STACK}"
  else
    pulumi stack init "${STACK}" >/dev/null 2>&1 || true
    pulumi stack select "${STACK}" >/dev/null 2>&1 || { err "pulumi: cannot create/select stack ${STACK}"; exit 17; }
    log "pulumi: created & selected ${STACK}"
  fi
  pulumi config set aws:region "${AWS_REGION}" --non-interactive >/dev/null 2>&1 || true
}

# S3 deletion helpers (handles versioned buckets)
delete_s3_versions_under_prefix(){
  local bucket="$1" prefix="$2"
  log "s3-clean: s3://${bucket}/${prefix:-/}"
  # list object versions, delete in batches (max 1000)
  while :; do
    local payload
    if [ -n "${prefix}" ]; then
      payload="$(aws s3api list-object-versions --bucket "${bucket}" --prefix "${prefix}" --output json 2>/dev/null || echo '{}')"
    else
      payload="$(aws s3api list-object-versions --bucket "${bucket}" --output json 2>/dev/null || echo '{}')"
    fi
    local objects
    objects="$(printf '%s' "${payload}" | jq -c '[.Versions[], .DeleteMarkers[]] | map({Key:.Key, VersionId:.VersionId})' 2>/dev/null || echo '[]')"
    if [ "${objects}" = "[]" ]; then
      break
    fi
    local tmpf
    tmpf="$(mktemp -p "${TMPDIR}")"
    printf '{"Objects":%s}' "${objects}" > "${tmpf}"
    aws s3api delete-objects --bucket "${bucket}" --delete "file://${tmpf}" >/dev/null 2>&1 || warn "s3: delete-objects partial failure"
    rm -f "${tmpf}"
  done
  log "s3-clean: done s3://${bucket}/${prefix:-/}"
}

delete_ddb_table(){
  local table="$1"
  log "ddb-delete: ${table}"
  if aws dynamodb describe-table --table-name "${table}" >/dev/null 2>&1; then
    aws dynamodb delete-table --table-name "${table}" --region "${AWS_REGION}" >/dev/null 2>&1 || warn "ddb: delete-table returned non-zero"
    aws dynamodb wait table-not-exists --table-name "${table}" --region "${AWS_REGION}" >/dev/null 2>&1 || warn "ddb: wait-not-exists returned non-zero"
    log "ddb-delete: done"
  else
    log "ddb-delete: not present; skipping"
  fi
}

if [ "${MODE}" = "create" ]; then
  log "MODE=create"
  ensure_bucket
  ensure_ddb_table
  venv_install_requirements
  pulumi_login_and_select
  log "pulumi: up --yes (non-interactive)"
  if ! pulumi up --yes >/dev/null 2>&1; then
    err "pulumi up failed; run 'pulumi up --diff' locally for diagnostics"
    exit 18
  fi
  log "create: complete"
  exit 0
fi

if [ "${MODE}" = "delete" ]; then
  log "MODE=delete"
  pulumi_login_and_select || true
  log "pulumi: destroy stack (if exists)"
  pulumi destroy --yes >/dev/null 2>&1 || warn "pulumi destroy returned non-zero"
  pulumi stack rm --yes >/dev/null 2>&1 || warn "pulumi stack rm returned non-zero"
  log "cleaning Pulumi state in s3://${PULUMI_S3_BUCKET}/${PULUMI_S3_PREFIX}/${STACK}"
  delete_s3_versions_under_prefix "${PULUMI_S3_BUCKET}" "${PULUMI_S3_PREFIX}/${STACK}"
  log "cleaning Pulumi state in s3://${PULUMI_S3_BUCKET}/${PULUMI_S3_PREFIX}"
  delete_s3_versions_under_prefix "${PULUMI_S3_BUCKET}" "${PULUMI_S3_PREFIX}"
  delete_ddb_table "${PULUMI_DDB_TABLE}"
  if [ "${PURGE_BUCKET}" = "true" ]; then
    log "purge requested: deleting all objects and the bucket ${PULUMI_S3_BUCKET}"
    delete_s3_versions_under_prefix "${PULUMI_S3_BUCKET}" ""
    aws s3api delete-bucket --bucket "${PULUMI_S3_BUCKET}" --region "${AWS_REGION}" >/dev/null 2>&1 || warn "bucket delete attempted (may need manual cleanup)"
  fi
  log "delete: complete"
  exit 0
fi

err "usage: $0 create|delete [--purge-bucket]"
exit 2
