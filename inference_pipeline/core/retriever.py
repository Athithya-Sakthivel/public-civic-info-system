#!/usr/bin/env python3
"""
inference_pipeline/core/retriever.py (patched final)

Assumptions & external contracts (explicit):
- Python 3.9+ runtime.
- Postgres with pgvector installed and available to the DB cluster.
- psycopg (v3) used for DB connectivity.
- Bedrock runtime accessible via boto3 for embeddings.
- Environment variables below must be set (fail-fast validated).
- Embeddings from Bedrock are lists-of-floats with length == EMBED_DIM.

Idempotency / determinism goals:
- Deterministic parameter ordering for DB queries.
- Do not rely on automatic adapter behavior for pgvector types in query parameters.
- Use explicit vector literal strings with `::vector` cast in SQL to avoid adapter issues.
- Filter keys processed in sorted order for deterministic SQL and param ordering.

Behavior:
- retrieve(event) performs embed -> pgvector candidate retrieval -> rerank -> dedupe -> return.
- On DB/embedding failures, returns an error payload but does not crash the process when called via handler().

Pro-tip: keep embedding formatting consistent (use high precision, but not excessive digits). This file uses 17 significant digits.
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
from typing import Any, Dict, List, Optional, Tuple

# Structured logger (JSON-like single-line)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("core.retriever")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "svc": "core.retriever"}
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))


# -------- Fail-fast imports (log then exit with unique codes) ----------
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

# pgvector python helper is optional for other uses; we do not rely on its
# automatic DB-parameter adaptation for search queries (we use explicit vector literals)
try:
    import pgvector  # type: ignore
except Exception as e:
    jlog({"level": "WARN", "event": "pgvector_helper_missing", "detail": str(e)})
    # Do not abort; we only require the DB extension at runtime, not the python helper.


# -------- Environment knobs (centralized & validated) ----------
def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if v is not None else default


# Required
AWS_REGION = _env("AWS_REGION")
if not AWS_REGION:
    jlog({"level": "CRITICAL", "event": "env_missing", "hint": "AWS_REGION must be set"})
    raise SystemExit(10)

# Embedding model / dims
EMBED_MODEL_ID = _env("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
try:
    EMBED_DIM = int(_env("EMBED_DIM", "1024"))
except Exception:
    jlog({"level": "CRITICAL", "event": "invalid_env", "hint": "EMBED_DIM must be an integer"})
    raise SystemExit(11)

# Vector DB type (only pgvector supported)
VECTOR_DB = _env("VECTOR_DB", "pgvector").strip().lower()
if VECTOR_DB != "pgvector":
    jlog({"level": "CRITICAL", "event": "invalid_vector_db", "value": VECTOR_DB, "hint": "Only 'pgvector' is supported"})
    raise SystemExit(12)

# Postgres connection params (required for pgvector)
PG_HOST = _env("PG_HOST")
PG_PORT = int(_env("PG_PORT", "5432"))
PG_USER = _env("PG_USER", "postgres")
PG_PASSWORD = _env("PG_PASSWORD")
PG_DB = _env("PG_DB", "postgres")
PG_TABLE_NAME = _env("PG_INDEX_NAME", "civic_chunks")

# Validate presence of critical PG envs
if not PG_HOST or not PG_PASSWORD:
    jlog({"level": "CRITICAL", "event": "pg_env_missing", "hint": "PG_HOST and PG_PASSWORD required for pgvector"})
    raise SystemExit(13)

# Validate and sanitize table name: allow only alnum + underscore, no schema qualification
_TABLE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
if not _TABLE_NAME_RE.match(PG_TABLE_NAME):
    jlog({"level": "CRITICAL", "event": "invalid_table_name", "value": PG_TABLE_NAME, "hint": "PG_INDEX_NAME must match ^[A-Za-z0-9_]+$"})
    raise SystemExit(14)

# Retrieval knobs
RAW_K = int(_env("RAW_K", "50"))
FINAL_K = int(_env("FINAL_K", "5"))

# Filters key name whitelist
META_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Operational note
jlog({"event": "startup_ok", "vector_db": VECTOR_DB, "embed_model": EMBED_MODEL_ID, "embed_dim": EMBED_DIM, "pg_table": PG_TABLE_NAME, "raw_k": RAW_K, "final_k": FINAL_K})


# -------- Module-level clients (cached singletons) ----------
_bedrock = None
_pg_conn = None


def init_bedrock_client():
    """Initialize (and cache) Bedrock runtime client."""
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
    """Initialize (and cache) a psycopg connection; performs sanity check for table & index."""
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
    # sanity check (non-fatal): verify table exists and HNSW index presence
    try:
        _check_table_and_hnsw_index(conn, PG_TABLE_NAME)
    except Exception as e:
        # don't fail startup; just surface warning
        jlog({"level": "WARN", "event": "index_check_failed", "detail": str(e)})
    return _pg_conn


# -------- Index sanity check (HNSW) ----------
def _check_table_and_hnsw_index(conn, table_name: str) -> None:
    """
    Ensure table exists and that an HNSW index exists (log warning if not).
    This is a non-fatal operational check (index absent -> WARN).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) AS reg", (table_name,))
        tr = cur.fetchone()
        if not tr or not tr.get("reg"):
            raise RuntimeError(f"table_missing:{table_name}")

        # Search pg_indexes for USING hnsw
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
            jlog(
                {
                    "level": "WARN",
                    "event": "hnsw_index_missing",
                    "hint": f"Expected HNSW index on {table_name}. Example: CREATE INDEX ON {table_name} USING hnsw (embedding);",
                }
            )
        else:
            jlog({"event": "hnsw_index_ok", "count": len(rows)})


# -------- Embedding (Bedrock) ----------
def get_embedding_from_bedrock(text: str) -> List[float]:
    """
    Request embedding from Bedrock. Validate shape.
    Raises on any error.
    """
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

    # common response keys to check
    emb = mr.get("embedding") or mr.get("embeddings") or mr.get("vector")
    if not isinstance(emb, list):
        jlog({"level": "ERROR", "event": "bedrock_no_embedding", "response_sample": mr})
        raise RuntimeError("bedrock returned no embedding list")
    if len(emb) != EMBED_DIM:
        jlog({"level": "ERROR", "event": "bedrock_dim_mismatch", "expected": EMBED_DIM, "received": len(emb)})
        raise RuntimeError("embedding_dim_mismatch")
    # ensure floats
    try:
        embf = [float(x) for x in emb]
    except Exception:
        jlog({"level": "ERROR", "event": "bedrock_embedding_non_numeric", "sample": emb[:10]})
        raise
    return embf


# -------- Helpers ----------
def _normalize_text_key(s: Optional[str]) -> str:
    """Normalized, hashed text key for dedupe stability."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # Hash normalized text to fixed-size key
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _format_vector_literal(vec: List[float]) -> str:
    """
    Format embedding list into a deterministic vector literal string suitable for
    use in SQL as %s::vector. Use high-precision formatting while avoiding
    noisy trailing zeros. This function returns a string like:
       "[0.12345678901234567, -0.00012345678901234, ...]"
    """
    # Validate shape
    if not isinstance(vec, (list, tuple)):
        raise ValueError("embedding_must_be_list")
    if len(vec) != EMBED_DIM:
        raise ValueError(f"embedding_dim_mismatch_expected_{EMBED_DIM}")
    # Ensure numeric and deterministic formatting
    pieces = []
    for x in vec:
        try:
            fx = float(x)
        except Exception:
            raise ValueError("embedding_contains_non_numeric")
        # 17 significant digits gives good float roundtrip without huge strings
        pieces.append(format(fx, ".17g"))
    return "[" + ",".join(pieces) + "]"


# -------- pgvector search (filter-first, safe paramization) ----------
def pgvector_search(query_emb: List[float], filters: Dict[str, str], raw_k: int) -> List[Dict[str, Any]]:
    """
    Query pgvector with filter-first approach:
      1) apply exact meta filters (meta->>'k' = v)
      2) order by embedding distance (embedding <-> %s::vector)
      3) limit raw_k

    Important: we do NOT pass python pgvector.Vector objects as query params (some psycopg adapters
    fail to adapt them). Instead we use a textual vector literal and cast to ::vector in SQL.
    """
    conn = init_pg_connection()

    # Build deterministic where clauses and corresponding params
    where_clauses: List[str] = []
    filter_params: List[Any] = []
    if isinstance(filters, dict):
        for k in sorted(filters.keys()):
            if not isinstance(k, str) or not META_KEY_RE.match(k):
                jlog({"level": "WARN", "event": "filter_key_skipped", "key": k})
                continue
            # Use parameterized key and value; the operator meta->>%s expects a text parameter for the key.
            where_clauses.append("meta->>%s = %s")
            filter_params.extend([k, filters[k]])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Prepare vector literal and validate shape
    try:
        vec_literal = _format_vector_literal(query_emb)
    except Exception as e:
        jlog({"level": "ERROR", "event": "invalid_query_embedding", "detail": str(e)})
        raise

    # SQL uses explicit ::vector cast for the parameter to ensure Postgres interprets it as vector
    sql = f"""
    SELECT document_id, chunk_id, chunk_index, content, meta, source_url, page_number,
           (embedding <-> %s::vector) AS distance
    FROM {PG_TABLE_NAME}
    {where_sql}
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """

    # Parameter ordering must match placeholders:
    # 1st %s -> vec_literal, then filter key/value pairs in sorted order, then 2nd %s -> vec_literal, then %s -> raw_k
    final_params: List[Any] = [vec_literal]
    final_params.extend(filter_params)
    final_params.extend([vec_literal, raw_k])

    # Execute query
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(final_params))
            rows = cur.fetchall()
    except Exception as e:
        # Be careful not to log full vector in case of accidental leak, but include truncated sample
        sample_vec = vec_literal[:200] + ("..." if len(vec_literal) > 200 else "")
        jlog({"level": "ERROR", "event": "pg_search_failed", "detail": str(e), "sql_sample": sql[:200], "vec_sample": sample_vec})
        raise

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "document_id": r.get("document_id"),
                "chunk_id": r.get("chunk_id"),
                "chunk_index": r.get("chunk_index"),
                "text": r.get("content"),
                "meta": r.get("meta") or {},
                "source_url": r.get("source_url"),
                "page_number": r.get("page_number"),
                "distance": float(r.get("distance")) if r.get("distance") is not None else None,
            }
        )
    return results


# -------- ranking helpers ----------
def compute_similarity_from_distance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    try:
        # Convert metric distance to similarity in (0,1]
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0


def trust_weight_for(meta: Dict[str, Any]) -> float:
    tl = (meta or {}).get("trust_level") or (meta or {}).get("trust") or ""
    if isinstance(tl, str):
        mapping = {"gov": 1.0, "government": 1.0, "implementing_agency": 0.95, "agency": 0.95, "ngo": 0.8, "news": 0.6}
        return float(mapping.get(tl.lower(), 1.0))
    return 1.0


def re_rank_and_select(candidates: List[Dict[str, Any]], final_k: int) -> List[Dict[str, Any]]:
    scored = []
    for c in candidates:
        dist = c.get("distance")
        sim = compute_similarity_from_distance(dist)
        tw = trust_weight_for(c.get("meta") or {})
        final_score = sim * tw
        # store computed fields for downstream auditing
        c["similarity"] = sim
        c["trust_weight"] = tw
        c["final_score"] = final_score
        scored.append(c)
    # deterministic sort: primary final_score desc, secondary similarity desc, tertiary chunk_id asc
    scored.sort(key=lambda x: (-x["final_score"], -x["similarity"], x.get("chunk_id", "")))
    return scored[:final_k]


# -------- dedupe nearest-first ----------
def dedupe_candidates_keep_nearest(candidates: List[Dict[str, Any]], max_keep: int) -> List[Dict[str, Any]]:
    seen_keys = set()
    deduped: List[Dict[str, Any]] = []
    for c in candidates:
        key = _normalize_text_key(c.get("text"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(c)
        if len(deduped) >= max_keep:
            break
    return deduped


# -------- Main pure retriever function ----------
def retrieve(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event keys:
      - request_id (optional)
      - query | question (required)
      - top_k (optional)
      - raw_k (optional)
      - filters (optional dict)
    Returns:
      {
        request_id, passages: [{number, chunk_id, document_id, chunk_index, text, meta, source_url, page_number, score, distance}],
        chunk_ids, top_similarity
      }
    """
    start = time.time()
    request_id = event.get("request_id") or f"r-{int(start * 1000)}"
    query_text = (event.get("query") or event.get("question") or "").strip()
    if not query_text:
        jlog({"level": "ERROR", "event": "invalid_request", "request_id": request_id})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "invalid_request"}

    top_k = int(event.get("top_k") or FINAL_K)
    raw_k = int(event.get("raw_k") or RAW_K)
    filters = event.get("filters") or {}

    jlog({"event": "retrieve_start", "request_id": request_id, "query_len": len(query_text), "raw_k": raw_k, "top_k": top_k, "filter_keys": sorted(list(filters.keys()))})

    # 1) embed
    try:
        emb = get_embedding_from_bedrock(query_text)
    except Exception as e:
        jlog({"level": "ERROR", "event": "embed_failed", "request_id": request_id, "detail": str(e)})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "embed_failed"}

    # 2) candidate retrieval (pgvector)
    try:
        candidates = pgvector_search(emb, filters, raw_k)
    except Exception as e:
        jlog({"level": "ERROR", "event": "vector_search_error", "request_id": request_id, "detail": str(e)})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "vector_search_failed"}

    if not candidates:
        jlog({"event": "no_candidates", "request_id": request_id})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0}

    # 3) dedupe by normalized text (nearest-first preserved because PG query ordered by embedding distance)
    deduped = dedupe_candidates_keep_nearest(candidates, raw_k)

    # 4) rerank and select final_k
    ranked = re_rank_and_select(deduped, top_k)

    # 5) format passages
    passages: List[Dict[str, Any]] = []
    chunk_ids: List[str] = []
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
    elapsed_ms = int((time.time() - start) * 1000)
    jlog({"event": "retrieval_complete", "request_id": request_id, "returned": len(passages), "top_similarity": top_similarity, "ms": elapsed_ms})
    return {"request_id": request_id, "passages": passages, "chunk_ids": chunk_ids, "top_similarity": float(top_similarity)}


# -------- Lambda handler (thin wrapper) ----------
def handler(event: Dict[str, Any], context: Any):
    try:
        return retrieve(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_unexpected", "detail": str(e)})
        return {"request_id": event.get("request_id") or f"r-{int(time.time() * 1000)}", "passages": [], "chunk_ids": [], "top_similarity": 0.0, "error": "handler_exception"}


# -------- CLI parity for local testing ----------
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
