from __future__ import annotations
import os
import sys
import json
import logging
import datetime
from typing import List, Dict, Any, Optional, Iterator

log = logging.getLogger("embed_and_index_s3")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(handler)
log.setLevel(logging.INFO)

def jlog(obj: Dict[str, Any]):
    log.info(json.dumps(obj, default=str))

try:
    import boto3
except Exception as e:
    jlog({"level":"CRITICAL","event":"boto3_missing","detail":str(e)})
    raise SystemExit(2)

try:
    import psycopg
    from psycopg.rows import class_row
except Exception as e:
    jlog({"level":"CRITICAL","event":"psycopg_missing","detail":str(e)})
    raise SystemExit(3)

try:
    from pgvector import Vector
except Exception:
    Vector = None

try:
    from opensearchpy import OpenSearch
except Exception:
    OpenSearch = None

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").lstrip("/").rstrip("/") + "/"
VECTOR_DB = os.getenv("VECTOR_DB", "pgvector").lower()
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
AWS_REGION = os.getenv("AWS_REGION")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DB = os.getenv("PG_DB", "postgres")
PG_TABLE_NAME = os.getenv("PG_INDEX_NAME", "documents")

OPENSEARCH_HOSTS = os.getenv("OPENSEARCH_HOSTS", "http://localhost:9200").split(",")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "documents")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

if not S3_BUCKET:
    jlog({"level":"CRITICAL","event":"s3_bucket_missing","hint":"Set S3_BUCKET env var"})
    raise SystemExit(10)

if not AWS_REGION:
    jlog({"level":"CRITICAL","event":"aws_region_missing","hint":"Set AWS_REGION env var (region must have Bedrock enabled)."})
    raise SystemExit(11)

def init_boto3_clients():
    try:
        session = boto3.session.Session(region_name=AWS_REGION)
        s3 = session.client("s3")
        bedrock = session.client("bedrock-runtime")
    except Exception as e:
        jlog({"level":"CRITICAL","event":"boto3_client_init_failed","detail":str(e)})
        raise SystemExit(12)
    return s3, bedrock

def list_chunk_objects(s3, bucket: str, prefix: str) -> Iterator[Dict[str, Any]]:
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    for page in paginator.paginate(**kwargs):
        contents = page.get("Contents") or []
        for obj in contents:
            key = obj.get("Key")
            if not key:
                continue
            if key.endswith(".chunks.jsonl"):
                yield {"Key": key, "Size": obj.get("Size", 0)}

def stream_jsonl_objects_from_s3(s3, bucket: str, key: str) -> Iterator[Dict[str, Any]]:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        jlog({"level":"ERROR","event":"s3_get_object_failed","key":key,"detail":str(e)})
        raise
    body = resp["Body"]
    for raw_line in body.iter_lines():
        if not raw_line:
            continue
        try:
            line = raw_line.decode("utf-8").strip()
        except Exception:
            try:
                line = raw_line.decode("latin-1").strip()
            except Exception:
                continue
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            jlog({"level":"ERROR","event":"jsonl_parse_failed","key":key,"line_sample":line[:200]})
            continue

def normalize_and_validate_chunk(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    required = ["document_id", "chunk_id", "text", "chunk_index", "token_count", "token_range", "document_total_tokens", "parser_version"]
    missing = [k for k in required if k not in obj]
    ingest_val = obj.get("ingest_time") or obj.get("timestamp")
    if not ingest_val:
        missing.append("ingest_time|timestamp")
    if missing:
        jlog({"level":"DEBUG","event":"schema_missing_fields","missing":missing,"chunk_id":obj.get("chunk_id")})
        return None
    out: Dict[str, Any] = {}
    out["document_id"] = str(obj["document_id"])
    out["chunk_id"] = str(obj["chunk_id"])
    out["chunk_index"] = int(obj["chunk_index"])
    out["chunk_type"] = str(obj.get("chunk_type", "token_window"))
    text = obj.get("text") or ""
    out["text"] = text.replace("\r\n", "\n").strip()
    out["token_count"] = int(obj.get("token_count", len(out["text"].split())))
    tr = obj.get("token_range")
    if isinstance(tr, list) and len(tr) == 2:
        out["token_range"] = [int(tr[0]), int(tr[1])]
    else:
        out["token_range"] = [0, out["token_count"]]
    out["document_total_tokens"] = int(obj.get("document_total_tokens", out["token_count"]))
    out["semantic_region"] = obj.get("semantic_region")
    out["source_url"] = obj.get("source_url")
    out["page_number"] = None if obj.get("page_number") is None else int(obj.get("page_number"))
    out["language"] = obj.get("language")
    out["ingest_time"] = str(ingest_val)
    out["parser_version"] = str(obj["parser_version"])
    out["meta"] = {}
    for k in ("headings","heading_path","figures","layout_tags","file_type","used_ocr","provenance","trust_level","region","topic_tags","source_domain"):
        if k in obj:
            out["meta"][k] = obj[k]
    return out

def init_bedrock_client(bedrock):
    return bedrock

def get_embedding_from_bedrock(client, model_id: str, text: str, dimensions: int = EMBED_DIM, normalize: bool = True) -> List[float]:
    native_request = {"inputText": text}
    if dimensions:
        native_request["dimensions"] = int(dimensions)
    if normalize:
        native_request["normalize"] = bool(normalize)
    request_body = json.dumps(native_request)
    try:
        response = client.invoke_model(modelId=model_id, body=request_body, contentType="application/json")
    except Exception as e:
        jlog({"level":"ERROR","event":"bedrock_invoke_failed","detail":str(e)})
        raise
    body_stream = response.get("body")
    if hasattr(body_stream, "read"):
        raw = body_stream.read()
    else:
        raw = body_stream
    try:
        model_response = json.loads(raw)
    except Exception as e:
        jlog({"level":"ERROR","event":"bedrock_response_decode_failed","detail":str(e)})
        raise
    embedding = model_response.get("embedding")
    if not isinstance(embedding, list):
        jlog({"level":"ERROR","event":"bedrock_no_embedding","response":model_response})
        raise RuntimeError("no_embedding_in_bedrock_response")
    if len(embedding) != EMBED_DIM:
        jlog({"level":"ERROR","event":"embedding_dim_mismatch","expected":EMBED_DIM,"received":len(embedding)})
        raise RuntimeError("embedding_dim_mismatch")
    return embedding

def init_pg_connection():
    conninfo = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"
    try:
        conn = psycopg.connect(conninfo)
    except Exception as e:
        jlog({"level":"CRITICAL","event":"pg_connect_failed","detail":str(e)})
        raise SystemExit(20)
    conn.autocommit = True
    return conn

def ensure_pgvector_table(conn):
    create_ext_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {PG_TABLE_NAME} (
      chunk_id TEXT PRIMARY KEY,
      document_id TEXT,
      content TEXT,
      embedding vector({EMBED_DIM}),
      meta JSONB,
      token_count INT,
      token_range INT[],
      document_total_tokens INT,
      semantic_region TEXT,
      source_url TEXT,
      page_number INT,
      language TEXT,
      ingest_time TIMESTAMP,
      parser_version TEXT
    );
    """
    try:
        with conn.cursor() as cur:
            cur.execute(create_ext_sql)
            cur.execute(create_table_sql)
    except Exception as e:
        jlog({"level":"CRITICAL","event":"pg_schema_setup_failed","detail":str(e)})
        raise SystemExit(21)

def pg_doc_exists(conn, chunk_id: str) -> bool:
    sql = f"SELECT 1 FROM {PG_TABLE_NAME} WHERE chunk_id = %s LIMIT 1;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (chunk_id,))
            return cur.fetchone() is not None
    except Exception as e:
        jlog({"level":"ERROR","event":"pg_exists_check_failed","detail":str(e),"chunk_id":chunk_id})
        return False

def pg_insert_batch(conn, docs: List[Dict[str, Any]]):
    insert_sql = f"""
    INSERT INTO {PG_TABLE_NAME}
      (chunk_id, document_id, content, embedding, meta, token_count, token_range, document_total_tokens, semantic_region, source_url, page_number, language, ingest_time, parser_version)
    VALUES (
      %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (chunk_id) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            for d in docs:
                embedding = d.get("embedding")
                if not isinstance(embedding, list):
                    raise RuntimeError("missing_embedding")
                if len(embedding) != EMBED_DIM:
                    raise RuntimeError("embedding_dim_mismatch")
                emb_str = "[" + ",".join(map(str, embedding)) + "]"
                meta = json.dumps(d.get("meta") or {})
                token_range = d.get("token_range") or [0, d.get("token_count",0)]
                ingest_ts = None
                try:
                    ingest_val = d.get("ingest_time")
                    if ingest_val and ingest_val.endswith("Z"):
                        ingest_val = ingest_val.replace("Z","+00:00")
                    ingest_ts = datetime.datetime.fromisoformat(ingest_val) if ingest_val else None
                except Exception:
                    ingest_ts = None
                cur.execute(insert_sql, (
                    d["chunk_id"],
                    d.get("document_id"),
                    d.get("text"),
                    emb_str,
                    meta,
                    d.get("token_count"),
                    token_range,
                    d.get("document_total_tokens"),
                    d.get("semantic_region"),
                    d.get("source_url"),
                    d.get("page_number"),
                    d.get("language"),
                    ingest_ts,
                    d.get("parser_version"),
                ))
    except Exception as e:
        jlog({"level":"ERROR","event":"pg_insert_failed","detail":str(e)})
        raise

def init_opensearch_client():
    if OpenSearch is None:
        jlog({"level":"CRITICAL","event":"opensearch_client_missing","detail":"opensearch-py not installed"})
        raise SystemExit(30)
    http_auth = (OPENSEARCH_USER, OPENSEARCH_PASSWORD) if OPENSEARCH_USER and OPENSEARCH_PASSWORD else None
    try:
        client = OpenSearch(hosts=OPENSEARCH_HOSTS, http_auth=http_auth, timeout=60)
    except Exception as e:
        jlog({"level":"CRITICAL","event":"opensearch_connect_failed","detail":str(e)})
        raise SystemExit(31)
    return client

def ensure_opensearch_index(client):
    mapping = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "meta": {"type": "object", "enabled": True},
                "embedding": {"type": "dense_vector", "dims": EMBED_DIM}
            }
        }
    }
    try:
        client.indices.create(index=OPENSEARCH_INDEX, body=mapping, ignore=400)
    except Exception as e:
        jlog({"level":"CRITICAL","event":"opensearch_index_create_failed","detail":str(e)})
        raise SystemExit(32)

def opensearch_doc_exists(client, chunk_id: str) -> bool:
    try:
        return client.exists(index=OPENSEARCH_INDEX, id=chunk_id)
    except Exception as e:
        jlog({"level":"ERROR","event":"opensearch_exists_failed","detail":str(e),"chunk_id":chunk_id})
        return False

def opensearch_index_batch(client, docs: List[Dict[str, Any]]):
    try:
        for d in docs:
            if not isinstance(d.get("embedding"), list):
                raise RuntimeError("missing_embedding")
            if len(d["embedding"]) != EMBED_DIM:
                raise RuntimeError("embedding_dim_mismatch")
            body = {
                "chunk_id": d["chunk_id"],
                "document_id": d.get("document_id"),
                "content": d.get("text"),
                "embedding": d.get("embedding"),
                "meta": d.get("meta") or {},
                "token_count": d.get("token_count"),
                "token_range": d.get("token_range"),
                "document_total_tokens": d.get("document_total_tokens"),
                "semantic_region": d.get("semantic_region"),
                "source_url": d.get("source_url"),
                "page_number": d.get("page_number"),
                "language": d.get("language"),
                "ingest_time": d.get("ingest_time"),
                "parser_version": d.get("parser_version"),
            }
            client.index(index=OPENSEARCH_INDEX, id=d["chunk_id"], body=body)
    except Exception as e:
        jlog({"level":"ERROR","event":"opensearch_index_failed","detail":str(e)})
        raise

def main():
    jlog({"event":"startup","vector_db":VECTOR_DB,"embed_model":EMBED_MODEL_ID,"s3_bucket":S3_BUCKET,"s3_prefix":S3_PREFIX})
    s3, bedrock_client = init_boto3_clients()
    total_indexed = 0
    total_skipped_schema = 0
    if VECTOR_DB == "pgvector":
        conn = init_pg_connection()
        ensure_pgvector_table(conn)
        for obj in list_chunk_objects(s3, S3_BUCKET, S3_PREFIX):
            key = obj["Key"]
            file_indexed = 0
            jlog({"event":"processing_s3_object","key":key,"size":obj.get("Size",0)})
            try:
                items_iter = stream_jsonl_objects_from_s3(s3, S3_BUCKET, key)
                to_index: List[Dict[str, Any]] = []
                for raw in items_iter:
                    c = normalize_and_validate_chunk(raw)
                    if not c:
                        total_skipped_schema += 1
                        continue
                    cid = c["chunk_id"]
                    if pg_doc_exists(conn, cid):
                        continue
                    emb = get_embedding_from_bedrock(bedrock_client, EMBED_MODEL_ID, c["text"])
                    c["embedding"] = emb
                    to_index.append(c)
                    if len(to_index) >= BATCH_SIZE:
                        pg_insert_batch(conn, to_index)
                        file_indexed += len(to_index)
                        total_indexed += len(to_index)
                        to_index = []
                if to_index:
                    pg_insert_batch(conn, to_index)
                    file_indexed += len(to_index)
                    total_indexed += len(to_index)
                jlog({"event":"file_done","key":key,"added_in_file":file_indexed,"cumulative_added":total_indexed,"skipped_schema":total_skipped_schema})
            except Exception as e:
                jlog({"level":"ERROR","event":"file_error","key":key,"detail":str(e)})
    elif VECTOR_DB == "opensearch":
        client = init_opensearch_client()
        ensure_opensearch_index(client)
        for obj in list_chunk_objects(s3, S3_BUCKET, S3_PREFIX):
            key = obj["Key"]
            file_indexed = 0
            jlog({"event":"processing_s3_object","key":key,"size":obj.get("Size",0)})
            try:
                items_iter = stream_jsonl_objects_from_s3(s3, S3_BUCKET, key)
                to_index: List[Dict[str, Any]] = []
                for raw in items_iter:
                    c = normalize_and_validate_chunk(raw)
                    if not c:
                        total_skipped_schema += 1
                        continue
                    cid = c["chunk_id"]
                    if opensearch_doc_exists(client, cid):
                        continue
                    emb = get_embedding_from_bedrock(bedrock_client, EMBED_MODEL_ID, c["text"])
                    c["embedding"] = emb
                    to_index.append(c)
                    if len(to_index) >= BATCH_SIZE:
                        opensearch_index_batch(client, to_index)
                        file_indexed += len(to_index)
                        total_indexed += len(to_index)
                        to_index = []
                if to_index:
                    opensearch_index_batch(client, to_index)
                    file_indexed += len(to_index)
                    total_indexed += len(to_index)
                jlog({"event":"file_done","key":key,"added_in_file":file_indexed,"cumulative_added":total_indexed,"skipped_schema":total_skipped_schema})
            except Exception as e:
                jlog({"level":"ERROR","event":"file_error","key":key,"detail":str(e)})
    else:
        jlog({"level":"CRITICAL","event":"unsupported_vector_db","value":VECTOR_DB})
        raise SystemExit(42)
    jlog({"event":"complete","total_indexed":total_indexed,"total_skipped_schema":total_skipped_schema})
    if total_skipped_schema > 0:
        raise SystemExit(50)
    return 0

if __name__ == "__main__":
    sys.exit(main())
