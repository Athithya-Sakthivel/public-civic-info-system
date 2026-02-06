
#!/usr/bin/env python3
"""
Production-ready retriever (final files for retriever/)

Assumptions & Invariants (fail-fast on import):
* Python 3.11 runtime.
* Exact pinned packages from requirements.txt installed in Lambda zip.
* Vector-store contract:
  - Table name (PG_TABLE_NAME) exists and contains columns:
    (document_id TEXT, chunk_id TEXT PRIMARY KEY, chunk_index INT,
     content TEXT, embedding vector(EMBED_DIM), source_url TEXT,
     page_number INT, parser_version TEXT, meta JSONB)
  - An HNSW (pgvector) index has been created on the embedding column by the indexing pipeline.
    Index creation example (indexing pipeline must run this):
      CREATE INDEX ON civic_chunks USING hnsw (embedding);
* This file exposes:
    - retrieve(event: dict) -> dict   (pure, deterministic)
    - handler(event, context)         (Lambda adapter)
* Embeddings produced by Bedrock (no mock).
* Filter keys are validated with META_KEY_RE whitelist.
* All env var initialization is centralized and validated at import.

Operational notes:
* Uses filter-first SQL then nearest-neighbor ORDER BY (embedding <-> %s).
* Checks existence of HNSW index at startup and logs a warning if missing.
* Returns full candidate metadata for Query Router to audit/hydrate.
* Read-only: no DDL or writes are performed here.
* Duplicate passages (near-identical text) are suppressed at retrieval time
  — we deduplicate by normalized text, keeping the nearest candidate first.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import datetime
import re
import hashlib
import unicodedata
from typing import Any, Dict, List, Optional

# ----- Structured logger -----
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("retriever")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))


# ----- Fail-fast imports -----
try:
    import boto3
except Exception as e:
    jlog({"level": "CRITICAL", "event": "boto3_missing", "detail": str(e)})
    raise SystemExit(2)

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception as e:
    jlog({"level": "CRITICAL", "event": "psycopg_missing", "detail": str(e)})
    raise SystemExit(3)

# ----- Environment knobs (validated) -----
def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if v is not None else default


AWS_REGION = _env("AWS_REGION")
if not AWS_REGION:
    jlog({"level": "CRITICAL", "event": "env_missing", "hint": "AWS_REGION must be set"})
    raise SystemExit(10)

EMBED_MODEL_ID = _env("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = int(_env("EMBED_DIM", "1024"))
VECTOR_DB = _env("VECTOR_DB", "pgvector").strip().lower()

# Postgres params (required for pgvector)
PG_HOST = _env("PG_HOST")
PG_PORT = int(_env("PG_PORT", "5432"))
PG_USER = _env("PG_USER", "postgres")
PG_PASSWORD = _env("PG_PASSWORD")
PG_DB = _env("PG_DB", "postgres")
PG_TABLE_NAME = _env("PG_INDEX_NAME", "civic_chunks")

# Retrieval knobs (tuneable by env)
RAW_K = int(_env("RAW_K", "50"))
FINAL_K = int(_env("FINAL_K", "5"))

# Filters key name regex (safe whitelist)
META_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Validate critical envs
if VECTOR_DB != "pgvector":
    jlog({"level": "CRITICAL", "event": "invalid_vector_db", "value": VECTOR_DB, "hint": "Only 'pgvector' is supported"})
    raise SystemExit(11)

if VECTOR_DB == "pgvector":
    if not PG_HOST or not PG_PASSWORD:
        jlog({"level": "CRITICAL", "event": "pg_env_missing", "hint": "PG_HOST and PG_PASSWORD required for pgvector"})
        raise SystemExit(12)

jlog({"event": "startup_ok", "vector_db": VECTOR_DB, "embed_model": EMBED_MODEL_ID, "embed_dim": EMBED_DIM, "pg_table": PG_TABLE_NAME})

# ----- Module-level clients (cached) -----
_bedrock = None
_pg_conn = None


def init_bedrock_client():
    global _bedrock
    if _bedrock is not None:
        return _bedrock
    try:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        jlog({"level": "CRITICAL", "event": "bedrock_client_failed", "detail": str(e)})
        _bedrock = None
        raise
    jlog({"event": "bedrock_client_init"})
    return _bedrock


def init_pg_connection():
    global _pg_conn
    if _pg_conn is not None:
        return _pg_conn
    conninfo = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"
    try:
        conn = psycopg.connect(conninfo, autocommit=True, row_factory=dict_row)
    except Exception as e:
        jlog({"level": "CRITICAL", "event": "pg_connect_failed", "detail": str(e)})
        _pg_conn = None
        raise
    _pg_conn = conn
    jlog({"event": "pg_connect_ok", "host": PG_HOST, "db": PG_DB})
    # perform a sanity check: table exists and HNSW index exists
    try:
        _check_table_and_hnsw_index(conn, PG_TABLE_NAME)
    except Exception as e:
        jlog({"level": "WARN", "event": "index_check_failed", "detail": str(e)})
    return _pg_conn


# ----- Index sanity check (HNSW) -----
def _check_table_and_hnsw_index(conn, table_name: str) -> None:
    """
    Check that the table exists and that an HNSW index (pgvector) exists.
    Log WARNING if index not found (indexing pipeline must create it).
    """
    try:
        with conn.cursor() as cur:
            # Check table exists
            cur.execute("SELECT to_regclass(%s) AS reg", (table_name,))
            tr = cur.fetchone()
            if not tr or not tr.get("reg"):
                raise RuntimeError(f"table_missing:{table_name}")

            # Check for HNSW index in pg_indexes.indexdef
            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = %s AND indexdef ILIKE '%%USING hnsw%%';
            """,
                (table_name,),
            )
            rows = cur.fetchall()
            if not rows:
                # Not fatal here; log a clear operational warning
                jlog(
                    {
                        "level": "WARN",
                        "event": "hnsw_index_missing",
                        "hint": f"Expect an HNSW index on {table_name}. Create with: CREATE INDEX ON {table_name} USING hnsw (embedding);",
                    }
                )
            else:
                jlog({"event": "hnsw_index_ok", "count": len(rows)})
    except Exception:
        # bubble up only as warning to avoid blocking transient DB issues; caller logs
        raise


# ----- Embedding (Bedrock) -----
def get_embedding_from_bedrock(text: str) -> List[float]:
    client = init_bedrock_client()
    body = json.dumps({"inputText": text})
    try:
        resp = client.invoke_model(modelId=EMBED_MODEL_ID, body=body, contentType="application/json")
    except Exception as e:
        jlog({"level": "ERROR", "event": "bedrock_invoke_failed", "detail": str(e)})
        raise
    body_stream = resp.get("body")
    raw = body_stream.read() if hasattr(body_stream, "read") else body_stream
    try:
        mr = json.loads(raw)
    except Exception as e:
        jlog({"level": "ERROR", "event": "bedrock_decode_failed", "detail": str(e), "sample": str(raw)[:300]})
        raise
    emb = mr.get("embedding") or mr.get("embeddings") or mr.get("vector")
    if not isinstance(emb, list):
        jlog({"level": "ERROR", "event": "bedrock_no_embedding", "response_sample": mr})
        raise RuntimeError("bedrock returned no embedding list")
    if len(emb) != EMBED_DIM:
        jlog({"level": "ERROR", "event": "bedrock_dim_mismatch", "expected": EMBED_DIM, "received": len(emb)})
        raise RuntimeError("embedding_dim_mismatch")
    return emb


# ----- Helpers: normalize text key for deduplication -----
def _normalize_text_key(s: Optional[str]) -> str:
    if not s:
        return ""
    # NFKC normalize, lowercase, collapse whitespace
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # To keep keys small and stable, hash the normalized text
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h


# ----- pgvector search (filter-first) -----
def pgvector_search(query_emb: List[float], filters: Dict[str, str], raw_k: int) -> List[Dict[str, Any]]:
    conn = init_pg_connection()
    emb_str = "[" + ",".join(map(str, query_emb)) + "]"
    where_clauses = []
    params: List[Any] = []

    if isinstance(filters, dict):
        for k, v in sorted(filters.items()):
            if not isinstance(k, str) or not META_KEY_RE.match(k):
                jlog({"level": "WARN", "event": "filter_key_skipped", "key": k})
                continue
            where_clauses.append(f"meta->>'{k}' = %s")
            params.append(v)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"""
    SELECT document_id, chunk_id, chunk_index, content, meta, source_url, page_number,
           (embedding <-> %s) AS distance
    FROM {PG_TABLE_NAME}
    {where_sql}
    ORDER BY embedding <-> %s
    LIMIT %s;
    """
    final_params = [emb_str] + params + [emb_str, raw_k] if where_clauses else [emb_str, emb_str, raw_k]
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(final_params))
            rows = cur.fetchall()
    except Exception as e:
        jlog({"level": "ERROR", "event": "pg_search_failed", "detail": str(e)})
        raise
    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "document_id": r["document_id"],
                "chunk_id": r["chunk_id"],
                "chunk_index": r.get("chunk_index"),
                "text": r.get("content"),
                "meta": r.get("meta") or {},
                "source_url": r.get("source_url"),
                "page_number": r.get("page_number"),
                "distance": float(r.get("distance")) if r.get("distance") is not None else None,
            }
        )
    return results


# ----- ranking helpers -----
def compute_similarity_from_distance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0


def trust_weight_for(meta: Dict[str, Any]) -> float:
    tl = (meta or {}).get("trust_level") or (meta or {}).get("trust") or ""
    if isinstance(tl, str):
        mapping = {"gov": 1.0, "government": 1.0, "implementing_agency": 0.9, "agency": 0.9, "ngo": 0.7, "news": 0.6}
        return float(mapping.get(tl.lower(), 1.0))
    return 1.0


def re_rank_and_select(candidates: List[Dict[str, Any]], final_k: int) -> List[Dict[str, Any]]:
    scored = []
    for c in candidates:
        dist = c.get("distance")
        sim = compute_similarity_from_distance(dist)
        tw = trust_weight_for(c.get("meta") or {})
        final_score = sim * tw
        c["similarity"] = sim
        c["trust_weight"] = tw
        c["final_score"] = final_score
        scored.append(c)
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:final_k]


# ----- deduplicate near-duplicate candidates by normalized text key -----
def dedupe_candidates_keep_nearest(candidates: List[Dict[str, Any]], max_keep: int) -> List[Dict[str, Any]]:
    seen_keys = set()
    deduped: List[Dict[str, Any]] = []
    for c in candidates:
        key = _normalize_text_key(c.get("text"))
        if key in seen_keys:
            # skip duplicate text (we keep the nearest because candidates are ordered by distance)
            continue
        seen_keys.add(key)
        deduped.append(c)
        if len(deduped) >= max_keep:
            break
    return deduped


# ----- Core retriever (pure) -----
def retrieve(event: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    request_id = event.get("request_id") or f"r-{int(start * 1000)}"
    query_text = (event.get("query") or event.get("question") or "").strip()
    if not query_text:
        jlog({"level": "ERROR", "event": "invalid_request", "request_id": request_id})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "invalid_request"}

    top_k = int(event.get("top_k") or FINAL_K)
    raw_k = int(event.get("raw_k") or RAW_K)
    filters = event.get("filters") or {}

    jlog({"event": "retrieve_start", "request_id": request_id, "query_len": len(query_text), "raw_k": raw_k, "top_k": top_k, "filters": list(filters.keys())})

    # 1) embed (Bedrock)
    try:
        emb = get_embedding_from_bedrock(query_text)
    except Exception as e:
        jlog({"level": "ERROR", "event": "embed_failed", "request_id": request_id, "detail": str(e)})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "embed_failed"}

    # 2) raw vector search (pgvector only)
    try:
        candidates = pgvector_search(emb, filters, raw_k)
    except Exception as e:
        jlog({"level": "ERROR", "event": "vector_search_error", "request_id": request_id, "detail": str(e)})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "vector_search_failed"}

    if not candidates:
        jlog({"event": "no_candidates", "request_id": request_id})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0}

    # 2.b) deduplicate by normalized text (keep nearest—input rows are ordered by distance)
    # We try to keep up to raw_k deduped candidates but cap at raw_k.
    deduped = dedupe_candidates_keep_nearest(candidates, raw_k)

    # 3) rerank and select final_k
    ranked = re_rank_and_select(deduped, top_k)
    passages = []
    chunk_ids = []
    for i, r in enumerate(ranked):
        passages.append(
            {
                "number": i + 1,
                "chunk_id": r["chunk_id"],
                "document_id": r.get("document_id"),
                "chunk_index": r.get("chunk_index"),
                "text": r.get("text") or "",
                "meta": r.get("meta") or {},
                "source_url": r.get("source_url"),
                "page_number": r.get("page_number"),
                "score": float(r.get("final_score", 0.0)),
                "distance": float(r.get("distance")) if r.get("distance") is not None else None,
            }
        )
        chunk_ids.append(r["chunk_id"])

    top_similarity = passages[0]["score"] if passages else 0.0
    elapsed = int((time.time() - start) * 1000)
    jlog({"event": "retrieval_complete", "request_id": request_id, "returned": len(passages), "top_similarity": top_similarity, "ms": elapsed})
    return {"request_id": request_id, "passages": passages, "chunk_ids": chunk_ids, "top_similarity": float(top_similarity)}


# ----- Lambda handler (thin wrapper) -----
def handler(event: Dict[str, Any], context: Any):
    try:
        res = retrieve(event)
        return res
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_unexpected", "detail": str(e)})
        return {"request_id": event.get("request_id") or f"r-{int(time.time() * 1000)}", "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "handler_exception"}


# ----- CLI parity -----
if __name__ == "__main__":
    ev = {
        "request_id": os.getenv("TEST_REQUEST_ID", "local-test"),
        "query": os.getenv("TEST_QUERY", "How to apply for a voter id?"),
        "top_k": int(os.getenv("TEST_TOP_K", str(FINAL_K))),
        "raw_k": int(os.getenv("TEST_RAW_K", str(RAW_K))),
        "filters": json.loads(os.getenv("TEST_FILTERS", "{}")),
    }
    out = retrieve(ev)
    print(json.dumps(out, indent=2, default=str))
