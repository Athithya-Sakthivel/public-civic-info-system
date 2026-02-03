#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: do not source this file. Run it directly." >&2
  exit 1
fi

MODE="${1:-}"
if [ "$MODE" != "create" ] && [ "$MODE" != "delete" ]; then
  echo "usage: $0 create|delete" >&2
  exit 2
fi

log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

# --- env / defaults ---
export AWS_REGION="${AWS_REGION:-ap-south-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/indexing_cronjob}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"

export PULUMI_STACK="${PULUMI_STACK:-prod}"
export STACK="$PULUMI_STACK"

export PULUMI_STATE_BUCKET="${PULUMI_STATE_BUCKET:-pulumi-backend-516}"
export PULUMI_STATE_PREFIX="${PULUMI_STATE_PREFIX:-pulumi}"
export DDB_TABLE="${DDB_TABLE:-pulumi-state-locks}"

export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-password}"
export PULUMI_LOGIN_URL="s3://${PULUMI_STATE_BUCKET}/${PULUMI_STATE_PREFIX}"

export PULUMI_PYTHON_CMD="${VENV_DIR}/bin/python"
export AWS_DYNAMODB_LOCK_TABLE="$DDB_TABLE"
export AWS_SDK_LOAD_CONFIG=1

# --- fast checks ---
for c in aws pulumi python3 jq; do
  command -v "$c" >/dev/null 2>&1 || die "missing dependency: $c"
done

aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials not working"

# --- helpers ---
ensure_bucket() {
  log "s3: ensure bucket ${PULUMI_STATE_BUCKET}"
  if aws s3api head-bucket --bucket "$PULUMI_STATE_BUCKET" >/dev/null 2>&1; then
    log "s3: bucket exists"
  else
    if [ "$AWS_REGION" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$PULUMI_STATE_BUCKET" >/dev/null
    else
      aws s3api create-bucket \
        --bucket "$PULUMI_STATE_BUCKET" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null
    fi
    aws s3api wait bucket-exists --bucket "$PULUMI_STATE_BUCKET"
    log "s3: bucket created"
  fi
  aws s3api put-bucket-versioning \
    --bucket "$PULUMI_STATE_BUCKET" \
    --versioning-configuration Status=Enabled >/dev/null 2>&1 || true
}

ensure_ddb_table() {
  log "ddb: ensure table ${DDB_TABLE}"
  if aws dynamodb describe-table --table-name "$DDB_TABLE" >/dev/null 2>&1; then
    log "ddb: table exists"
    return
  fi
  aws dynamodb create-table \
    --table-name "$DDB_TABLE" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$AWS_REGION" >/dev/null 2>&1
  aws dynamodb wait table-exists --table-name "$DDB_TABLE" --region "$AWS_REGION"
  log "ddb: table ready"
}

create_venv_and_install() {
  if [ ! -d "$VENV_DIR" ]; then
    log "venv: creating at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    # ensure pip compatible with pinned pulumi
    "$VENV_DIR/bin/python" -m pip install --upgrade "pip<26"
  fi

  if [ -f "$REQ_FILE" ]; then
    log "venv: installing $REQ_FILE"
    "$VENV_DIR/bin/python" -m pip install -q -r "$REQ_FILE"
  else
    log "venv: installing minimal pulumi deps"
    "$VENV_DIR/bin/python" -m pip install -q pulumi pulumi-aws boto3
  fi
}

pulumi_prepare() {
  cd "$PROJECT_DIR"
  if [ ! -f Pulumi.yaml ]; then
    cat >Pulumi.yaml <<YAML
name: indexing-cronjob
runtime: python
YAML
  fi

  pulumi login "$PULUMI_LOGIN_URL" >/dev/null
  if pulumi stack select "$STACK" >/dev/null 2>&1; then
    log "pulumi: selected existing stack $STACK"
  else
    log "pulumi: initializing stack $STACK"
    pulumi stack init "$STACK" >/dev/null
  fi
  pulumi config set aws:region "$AWS_REGION" --non-interactive >/dev/null
}

pulumi_preview_and_up() {
  # run preview; if preview exits non-zero, abort without changing infra
  log "pulumi: running preview"
  if pulumi preview --stack "$STACK" --non-interactive; then
    log "pulumi: preview succeeded, proceeding to up"
    pulumi up --yes --stack "$STACK"
  else
    die "pulumi preview failed; aborting. Fix issues and retry."
  fi
}

pulumi_destroy_only() {
  # Select stack; if not present, nothing to do
  cd "$PROJECT_DIR"
  if pulumi stack select "$STACK" >/dev/null 2>&1; then
    log "pulumi: destroying stack $STACK"
    pulumi destroy --yes --stack "$STACK" || log "pulumi: destroy returned non-zero, continuing cleanup"
    pulumi stack rm --yes "$STACK" || log "pulumi: stack rm returned non-zero"
  else
    log "pulumi: stack $STACK not found; skipping destroy"
  fi
}

delete_s3_prefix() {
  log "s3: deleting pulumi state prefix ${PULUMI_STATE_PREFIX}"
  while :; do
    objs="$(aws s3api list-object-versions \
      --bucket "$PULUMI_STATE_BUCKET" \
      --prefix "$PULUMI_STATE_PREFIX" \
      --output json | jq -c '[.Versions[],.DeleteMarkers[]] | map({Key:.Key,VersionId:.VersionId})')"
    [ "$objs" = "[]" ] && break
    aws s3api delete-objects \
      --bucket "$PULUMI_STATE_BUCKET" \
      --delete "{\"Objects\":$objs}" >/dev/null
  done
}

delete_ddb_table() {
  if aws dynamodb describe-table --table-name "$DDB_TABLE" >/dev/null 2>&1; then
    aws dynamodb delete-table --table-name "$DDB_TABLE" >/dev/null
    aws dynamodb wait table-not-exists --table-name "$DDB_TABLE" --region "$AWS_REGION" || true
    log "ddb: deleted $DDB_TABLE"
  else
    log "ddb: table $DDB_TABLE not present"
  fi
}

# --- main branching; strictly separate flows ---
if [ "$MODE" = "create" ]; then
  log "mode=create"
  ensure_bucket
  ensure_ddb_table
  create_venv_and_install
  pulumi_prepare
  pulumi_preview_and_up
  log "CREATE complete"
  exit 0
fi

if [ "$MODE" = "delete" ]; then
  log "mode=delete"
  # pulumi_prepare used only to select login and stack where necessary
  cd "$PROJECT_DIR"
  pulumi login "$PULUMI_LOGIN_URL" >/dev/null
  pulumi_destroy_only
  # cleanup backend artifacts
  delete_s3_prefix
  delete_ddb_table
  log "DELETE complete"
  exit 0
fi
