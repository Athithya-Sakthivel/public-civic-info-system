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

export AWS_REGION="${AWS_REGION:-ap-south-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
export FORCE_DELETE="${FORCE_DELETE:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/indexing_cronjob}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"
export IMAGE_URI="${IMAGE_URI:-athithya5354/civic-indexing:latest}"

export PULUMI_STACK="${PULUMI_STACK:-prod}"
STACK="$PULUMI_STACK"
export PULUMI_STATE_BUCKET="${PULUMI_STATE_BUCKET:-pulumi-backend-670}"
export PULUMI_STATE_PREFIX="${PULUMI_STATE_PREFIX:-pulumi}"
export DDB_TABLE="${DDB_TABLE:-pulumi-state-locks}"
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-password}"
export PULUMI_LOGIN_URL="s3://${PULUMI_STATE_BUCKET}/${PULUMI_STATE_PREFIX}"

LOG(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
DIE(){ echo "ERROR: $*" >&2; exit 1; }

for bin in aws pulumi python3 jq; do
  command -v "$bin" >/dev/null 2>&1 || DIE "missing required CLI: $bin"
done
aws sts get-caller-identity >/dev/null 2>&1 || DIE "AWS credentials not working"

ensure_bucket(){
  LOG "s3: verify state bucket ${PULUMI_STATE_BUCKET}"
  if aws s3api head-bucket --bucket "$PULUMI_STATE_BUCKET" >/dev/null 2>&1; then
    LOG "s3: bucket exists"
    aws s3api put-bucket-versioning --bucket "$PULUMI_STATE_BUCKET" --versioning-configuration Status=Enabled >/dev/null 2>&1 || true
  else
    DIE "s3: state bucket ${PULUMI_STATE_BUCKET} does not exist; refusing to create it. Set PULUMI_STATE_BUCKET to an existing bucket."
  fi
}

ensure_ddb(){
  LOG "ddb: ensure table ${DDB_TABLE}"
  if aws dynamodb describe-table --table-name "$DDB_TABLE" >/dev/null 2>&1; then
    LOG "ddb: table exists"
    return
  fi
  aws dynamodb create-table --table-name "$DDB_TABLE" --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --billing-mode PAY_PER_REQUEST --region "$AWS_REGION" >/dev/null
  aws dynamodb wait table-exists --table-name "$DDB_TABLE" --region "$AWS_REGION"
  LOG "ddb: table ready"
}

create_venv_if_missing(){
  if [ ! -d "$VENV_DIR" ]; then
    LOG "venv: creating at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade "pip<26" >/dev/null
  fi
  if [ -f "$REQ_FILE" ]; then
    LOG "venv: installing from $REQ_FILE"
    "$VENV_DIR/bin/python" -m pip install -q -r "$REQ_FILE"
  else
    LOG "venv: no requirements.txt found; installing minimal runtime"
    "$VENV_DIR/bin/python" -m pip install -q pulumi pulumi-aws boto3
  fi
  export PULUMI_PYTHON_CMD="${PULUMI_PYTHON_CMD:-$VENV_DIR/bin/python}"
  LOG "env: PULUMI_PYTHON_CMD set to $PULUMI_PYTHON_CMD"
}

pulumi_prepare(){
  LOG "pulumi: preparing project in $PROJECT_DIR"
  cd "$PROJECT_DIR"
  if [ ! -f Pulumi.yaml ]; then
    cat >Pulumi.yaml <<YAML
name: indexing-cronjob
runtime: python
YAML
  fi
  pulumi login "$PULUMI_LOGIN_URL" >/dev/null
  if pulumi stack select "$STACK" >/dev/null 2>&1; then
    LOG "pulumi: selected existing stack $STACK"
  else
    LOG "pulumi: initializing stack $STACK"
    pulumi stack init "$STACK" >/dev/null
  fi
  pulumi config set aws:region "$AWS_REGION" --non-interactive >/dev/null
  if pulumi config get indexing-cronjob:image_uri --stack "$STACK" >/dev/null 2>&1; then
    LOG "pulumi: image_uri present in stack config; leaving unchanged"
  else
    LOG "pulumi: setting indexing-cronjob:image_uri to $IMAGE_URI"
    pulumi config set indexing-cronjob:image_uri "$IMAGE_URI" --stack "$STACK" --non-interactive >/dev/null
  fi
}

safe_delete_s3_prefix(){
  LOG "s3: cleaning state prefix s3://${PULUMI_STATE_BUCKET}/${PULUMI_STATE_PREFIX}/"
  TMP_LIST="$(mktemp)"
  TMP_DELETE="$(mktemp)"
  trap 'rm -f "$TMP_LIST" "$TMP_DELETE"' EXIT
  while true; do
    aws s3api list-object-versions --bucket "$PULUMI_STATE_BUCKET" --prefix "$PULUMI_STATE_PREFIX" --output json > "$TMP_LIST" || echo '{}' > "$TMP_LIST"
    jq -r '([(.Versions // []), (.DeleteMarkers // [])] | add) | map({Key: .Key, VersionId: .VersionId})' "$TMP_LIST" > "$TMP_DELETE" || true
    if [ ! -s "$TMP_DELETE" ] || [ "$(jq 'length' "$TMP_DELETE")" -eq 0 ]; then
      LOG "s3: no objects left under prefix"
      break
    fi
    jq -n --slurpfile items "$TMP_DELETE" '{Objects: $items[0]}' > "${TMP_DELETE}.payload"
    LOG "s3: deleting $(jq 'length' "$TMP_DELETE") objects from $PULUMI_STATE_BUCKET"
    aws s3api delete-objects --bucket "$PULUMI_STATE_BUCKET" --delete "file://${TMP_DELETE}.payload" >/dev/null
    rm -f "${TMP_DELETE}.payload"
  done
  LOG "s3: cleanup complete"
  rm -f "$TMP_LIST" "$TMP_DELETE" || true
  trap - EXIT
}

delete_ddb_table(){
  if aws dynamodb describe-table --table-name "$DDB_TABLE" >/dev/null 2>&1; then
    LOG "ddb: deleting table $DDB_TABLE"
    aws dynamodb delete-table --table-name "$DDB_TABLE" >/dev/null
    aws dynamodb wait table-not-exists --table-name "$DDB_TABLE" --region "$AWS_REGION" || true
    LOG "ddb: deleted"
  else
    LOG "ddb: table not present, skipping"
  fi
}

write_outputs_json(){
  cd "$PROJECT_DIR"
  LOG "pulumi: writing outputs to $PROJECT_DIR/outputs.json"
  pulumi stack output --stack "$STACK" --json > outputs.json
  LOG "pulumi: outputs written"
}

remove_outputs_json(){
  cd "$PROJECT_DIR"
  if [ -f outputs.json ]; then
    LOG "removing $PROJECT_DIR/outputs.json"
    rm -f outputs.json
  else
    LOG "no outputs.json present"
  fi
}

if [ "$MODE" = "create" ]; then
  LOG "mode=create"
  ensure_bucket
  ensure_ddb
  create_venv_if_missing
  pulumi_prepare
  LOG "pulumi: running preview (will abort on failure)"
  if pulumi preview --stack "$STACK" --non-interactive; then
    LOG "pulumi: preview succeeded; running up"
    pulumi up --stack "$STACK" --yes
    write_outputs_json
    LOG "CREATE complete"
    exit 0
  else
    DIE "pulumi preview failed; aborting create"
  fi
fi

if [ "$MODE" = "delete" ]; then
  LOG "mode=delete"
  create_venv_if_missing
  cd "$PROJECT_DIR"
  pulumi login "$PULUMI_LOGIN_URL" >/dev/null
  if pulumi stack select "$STACK" >/dev/null 2>&1; then
    LOG "pulumi: destroying stack $STACK"
    pulumi destroy --stack "$STACK" --yes || LOG "pulumi destroy returned non-zero"
    LOG "pulumi: removing stack metadata $STACK"
    pulumi stack rm --stack "$STACK" --yes || LOG "pulumi stack rm returned non-zero"
  else
    LOG "pulumi: stack $STACK not found, skipping destroy"
  fi

  if [ "$FORCE_DELETE" = "1" ]; then
    safe_delete_s3_prefix
    delete_ddb_table
  else
    LOG "FORCE_DELETE != 1; skipping backend S3 and DynamoDB deletion to preserve state"
  fi

  remove_outputs_json
  LOG "DELETE complete"
  exit 0
fi

DIE "internal error: unreachable"
