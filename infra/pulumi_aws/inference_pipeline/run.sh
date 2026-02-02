#!/usr/bin/env bash
set -euo pipefail

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: do not source this file. Run it: bash $0" >&2
  return 1 2>/dev/null || exit 1
fi

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

require() { command -v "$1" >/dev/null 2>&1 || { echo "required: $1" >&2; exit 10; }; }

require aws
require pulumi
require python3

aws sts get-caller-identity >/dev/null 2>&1 || { echo "ERROR: AWS credentials not configured or invalid (aws sts failed)" >&2; exit 11; }

log() { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { echo "ERROR: $*" >&2; exit "${2:-1}"; }

retry_cmd() {
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
    delay=$((delay * 2))
  done
  return $rc
}

ensure_bucket() {
  log "ensure: S3 bucket ${PULUMI_S3_BUCKET}"
  if aws s3api head-bucket --bucket "$PULUMI_S3_BUCKET" >/dev/null 2>&1; then
    log "s3: bucket exists"
    return 0
  fi
  log "s3: creating bucket ${PULUMI_S3_BUCKET} in region ${AWS_REGION}"
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$PULUMI_S3_BUCKET" >/dev/null 2>&1 || true
  else
    aws s3api create-bucket --bucket "$PULUMI_S3_BUCKET" --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null 2>&1 || true
  fi
  retry_cmd 8 1 aws s3api head-bucket --bucket "$PULUMI_S3_BUCKET" >/dev/null 2>&1 || die "s3: bucket create/visibility failed"
  aws s3api put-bucket-versioning --bucket "$PULUMI_S3_BUCKET" --versioning-configuration Status=Enabled >/dev/null 2>&1 || true
  aws s3api put-bucket-encryption --bucket "$PULUMI_S3_BUCKET" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null 2>&1 || true
  log "s3: bucket created and configured"
}

ensure_ddb_table() {
  log "ensure: DynamoDB table ${PULUMI_DDB_TABLE}"
  if aws dynamodb describe-table --table-name "$PULUMI_DDB_TABLE" >/dev/null 2>&1; then
    log "ddb: table exists"
    return 0
  fi

  local attempts=6
  local delay=2
  local i=0
  while [ $i -lt $attempts ]; do
    tmp=$(mktemp) || die "mktemp failed"
    set +e
    aws dynamodb create-table \
      --table-name "$PULUMI_DDB_TABLE" \
      --attribute-definitions AttributeName=LockID,AttributeType=S \
      --key-schema AttributeName=LockID,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST \
      --region "$AWS_REGION" 2> "$tmp"
    rc=$?
    set -e

    if [ $rc -eq 0 ]; then
      rm -f "$tmp"
      retry_cmd 12 2 aws dynamodb wait table-exists --table-name "$PULUMI_DDB_TABLE" --region "$AWS_REGION" >/dev/null 2>&1 || die "ddb: wait for table active failed"
      log "ddb: created and ACTIVE"
      return 0
    fi

    stderr_content="$(cat "$tmp" 2>/dev/null || true)"
    rm -f "$tmp"

    if echo "$stderr_content" | grep -qi "ResourceInUseException"; then
      log "ddb: ResourceInUseException returned; verifying existence"
      if aws dynamodb describe-table --table-name "$PULUMI_DDB_TABLE" >/dev/null 2>&1; then
        log "ddb: table exists after race; continuing"
        return 0
      fi
    fi

    if echo "$stderr_content" | grep -qi "AccessDenied\|UnauthorizedOperation"; then
      die "ddb: create failed due to permissions; $stderr_content"
    fi

    if echo "$stderr_content" | grep -qi "ThrottlingException\|ProvisionedThroughputExceededException"; then
      log "ddb: transient throttling, retrying (attempt $((i+1))/$attempts)"
      i=$((i+1))
      sleep $delay
      delay=$((delay * 2))
      continue
    fi

    # last-ditch check: maybe table became visible shortly after error
    if aws dynamodb describe-table --table-name "$PULUMI_DDB_TABLE" >/dev/null 2>&1; then
      log "ddb: table observed after create attempt (eventual consistency) — continuing"
      return 0
    fi

    log "ddb: create-table returned non-zero (rc=$rc); stderr was: ${stderr_content}"
    i=$((i+1))
    sleep $delay
    delay=$((delay * 2))
  done

  die "ddb: create-table failed after retries; last error: ${stderr_content}"
}

delete_s3_versions_under_prefix() {
  local bucket="$1" prefix="$2"
  log "s3-delete-versions: bucket=${bucket} prefix=${prefix:-(root)}"
  while :; do
    rsp="$(aws s3api list-object-versions --bucket "$bucket" ${prefix:+--prefix "$prefix"} --output json 2>/dev/null || echo '{}')"
    objs="$(python3 - "$rsp" <<'PY'
import sys,json
try:
    r=json.load(sys.stdin)
except Exception:
    r={}
arr=[]
for k in ("Versions","DeleteMarkers"):
    for it in r.get(k,[]):
        arr.append({"Key": it.get("Key"), "VersionId": it.get("VersionId")})
print(json.dumps(arr))
PY
)"
    [ "$objs" = "[]" ] && break
    tmp="$(mktemp)" || exit 1
    printf '{"Objects":%s}' "$objs" >"$tmp"
    aws s3api delete-objects --bucket "$bucket" --delete "file://$tmp" >/dev/null 2>&1 || true
    rm -f "$tmp"
  done
  log "s3-delete-versions: done for s3://${bucket}/${prefix}"
}

delete_ddb_table() {
  local table="$1"
  log "ddb-delete: deleting table ${table} if exists"
  if aws dynamodb describe-table --table-name "$table" >/dev/null 2>&1; then
    aws dynamodb delete-table --table-name "$table" --region "$AWS_REGION" >/dev/null 2>&1 || true
    aws dynamodb wait table-not-exists --table-name "$table" --region "$AWS_REGION" >/dev/null 2>&1 || true
    log "ddb-delete: done"
  else
    log "ddb-delete: table not found; skipping"
  fi
}

pulumi_login_and_select() {
  log "pulumi: login ${PULUMI_LOGIN_URL}"
  export AWS_REGION AWS_DEFAULT_REGION
  export AWS_SDK_LOAD_CONFIG=1
  retry_cmd 6 1 pulumi login "$PULUMI_LOGIN_URL" >/dev/null 2>&1 || die "pulumi login failed"
  if pulumi stack select "$STACK" >/dev/null 2>&1; then
    log "pulumi: selected stack $STACK"
  else
    pulumi stack init "$STACK" >/dev/null 2>&1 || true
    pulumi stack select "$STACK" >/dev/null 2>&1 || die "pulumi: cannot create/select stack $STACK"
    log "pulumi: created & selected stack $STACK"
  fi
  pulumi config set aws:region "$AWS_REGION" --non-interactive >/dev/null 2>&1 || true
  export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-}"
  export PULUMI_DYNAMODB_TABLE="$PULUMI_DDB_TABLE"
}

if [ "$MODE" = "create" ]; then
  log "MODE=create"
  ensure_bucket
  ensure_ddb_table
  pulumi_login_and_select
  log "pulumi: running up --yes"
  pulumi up --yes || die "pulumi up failed"
  log "create: complete"
  exit 0
fi

if [ "$MODE" = "delete" ]; then
  log "MODE=delete"
  pulumi_login_and_select
  log "pulumi: destroy stack (if exists)"
  pulumi destroy --yes >/dev/null 2>&1 || log "pulumi destroy returned non-zero (continuing)"
  pulumi stack rm --yes >/dev/null 2>&1 || log "pulumi stack rm returned non-zero (continuing)"
  log "cleaning pulumi state in s3://${PULUMI_S3_BUCKET}/${PULUMI_S3_PREFIX}/${STACK}"
  delete_s3_versions_under_prefix "$PULUMI_S3_BUCKET" "${PULUMI_S3_PREFIX}/${STACK}"
  log "cleaning pulumi state in s3://${PULUMI_S3_BUCKET}/${PULUMI_S3_PREFIX}"
  delete_s3_versions_under_prefix "$PULUMI_S3_BUCKET" "${PULUMI_S3_PREFIX}"
  delete_ddb_table "$PULUMI_DDB_TABLE"
  if [ "$PURGE_BUCKET" = "true" ]; then
    log "purge requested: deleting all objects and bucket ${PULUMI_S3_BUCKET}"
    delete_s3_versions_under_prefix "$PULUMI_S3_BUCKET" ""
    aws s3api delete-bucket --bucket "$PULUMI_S3_BUCKET" --region "$AWS_REGION" >/dev/null 2>&1 || log "bucket delete attempted (may require manual cleanup)"
  fi
  log "delete: complete"
  exit 0
fi

echo "usage: $0 create|delete [--purge-bucket]" >&2
exit 2
