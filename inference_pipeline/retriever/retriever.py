from __future__ import annotations
import os
import sys
import json
import logging
import time
import datetime
from typing import List, Dict, Any, Optional

log = logging.getLogger("retrieval")
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
except Exception as e:
    jlog({"level":"CRITICAL","event":"psycopg_missing","detail":str(e)})
    raise SystemExit(3)

try:
    from opensearchpy import OpenSearch
except Exception:
    OpenSearch = None

VECTOR_DB = os.getenv("VECTOR_DB", "pgvector").lower()
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
AWS_REGION = os.getenv("AWS_REGION", None)

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DB = os.getenv("PG_DB", "postgres")
PG_TABLE_NAME = os.getenv("PG_INDEX_NAME", "documents")

OPENSEARCH_HOSTS = os.getenv("OPENSEARCH_HOSTS", "http://localhost:9200").split(",")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "documents")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", None)
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", None)

RAW_K = int(os.getenv("RAW_K", "50"))
FINAL_K = int(os.getenv("FINAL_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.0"))
FRESHNESS_DAYS = int(os.getenv("FRESHNESS_DAYS", "365"))
MOCK_EMBED = os.getenv("MOCK_EMBED", "0") == "1"

_bedrock = None
_pg_conn = None
_os_client = None

def init_bedrock_client():
    global _bedrock
    if _bedrock is not None:
        return _bedrock
    if not AWS_REGION:
        jlog({"level":"CRITICAL","event":"aws_region_missing","hint":"Set AWS_REGION env var"})
        raise SystemExit(10)
    _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock

def embed_text_bedrock(text: str) -> List[float]:
    if MOCK_EMBED:
        return mock_embedding(text)
    client = init_bedrock_client()
    payload = {"inputText": text, "dimensions": EMBED_DIM, "normalize": True}
    body = json.dumps(payload)
    try:
        resp = client.invoke_model(modelId=EMBED_MODEL_ID, body=body, contentType="application/json")
    except Exception as e:
        jlog({"level":"ERROR","event":"bedrock_invoke_failed","detail":str(e)})
        raise
    body_stream = resp.get("body")
    raw = body_stream.read() if hasattr(body_stream, "read") else body_stream
    try:
        mr = json.loads(raw)
    except Exception as e:
        jlog({"level":"ERROR","event":"bedrock_response_decode_failed","detail":str(e)})
        raise
    embedding = mr.get("embedding")
    if not isinstance(embedding, list) or len(embedding) != EMBED_DIM:
        jlog({"level":"ERROR","event":"bedrock_embedding_invalid","response":mr})
        raise RuntimeError("invalid_embedding")
    return embedding

def mock_embedding(text: str) -> List[float]:
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    i = 0
    while len(vals) < EMBED_DIM:
        chunk = h[i % len(h)]
        vals.append(((chunk / 255.0) * 2.0) - 1.0)
        i += 1
    return vals[:EMBED_DIM]

def init_pg_connection():
    global _pg_conn
    if _pg_conn is not None:
        return _pg_conn
    conninfo = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"
    try:
        conn = psycopg.connect(conninfo, autocommit=True)
    except Exception as e:
        jlog({"level":"CRITICAL","event":"pg_connect_failed","detail":str(e)})
        raise
    _pg_conn = conn
    return _pg_conn

def init_opensearch_client():
    global _os_client
    if _os_client is not None:
        return _os_client
    if OpenSearch is None:
        jlog({"level":"CRITICAL","event":"opensearch_client_missing","hint":"install opensearch-py"})
        raise SystemExit(20)
    http_auth = (OPENSEARCH_USER, OPENSEARCH_PASSWORD) if OPENSEARCH_USER and OPENSEARCH_PASSWORD else None
    try:
        client = OpenSearch(hosts=OPENSEARCH_HOSTS, http_auth=http_auth, timeout=60)
    except Exception as e:
        jlog({"level":"CRITICAL","event":"opensearch_connect_failed","detail":str(e)})
        raise
    _os_client = client
    return _os_client

def pgvector_search(query_emb: List[float], language: Optional[str], region: Optional[str], raw_k: int) -> List[Dict[str, Any]]:
    conn = init_pg_connection()
    emb_str = "[" + ",".join(map(str, query_emb)) + "]"
    where_clauses = []
    params = []
    if language:
        where_clauses.append("language = %s")
        params.append(language)
    if region:
        where_clauses.append("meta->>'region' = %s")
        params.append(region)
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"""
    SELECT chunk_id, content, meta, source_url, page_number, language, ingest_time, (embedding <-> %s) AS distance
    FROM {PG_TABLE_NAME}
    {where_sql}
    ORDER BY embedding <-> %s
    LIMIT %s;
    """
    params = [emb_str, emb_str, raw_k] if not where_clauses else [emb_str] + params + [emb_str, raw_k]
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    except Exception as e:
        jlog({"level":"ERROR","event":"pg_search_failed","detail":str(e)})
        raise
    results = []
    for r in rows:
        chunk_id, content, meta, source_url, page_number, language_r, ingest_time, distance = r
        results.append({
            "chunk_id": chunk_id,
            "text": content,
            "meta": meta or {},
            "source_url": source_url,
            "page_number": page_number,
            "language": language_r,
            "ingest_time": ingest_time.isoformat() if ingest_time else None,
            "distance": float(distance) if distance is not None else None
        })
    return results

def opensearch_search(query_emb: List[float], language: Optional[str], region: Optional[str], raw_k: int) -> List[Dict[str, Any]]:
    client = init_opensearch_client()
    body = {"knn": {"embedding": {"vector": query_emb, "k": raw_k}}}
    if language or region:
        filters = []
        if language:
            filters.append({"term": {"language": language}})
        if region:
            filters.append({"term": {"meta.region": region}})
        body = {"query": {"bool": {"filter": filters, "must": {"knn": {"embedding": {"vector": query_emb, "k": raw_k}}}}}}
    try:
        resp = client.search(index=OPENSEARCH_INDEX, body=body, size=raw_k)
    except Exception as e:
        jlog({"level":"ERROR","event":"opensearch_search_failed","detail":str(e)})
        raise
    hits = resp.get("hits", {}).get("hits", [])
    results = []
    for h in hits:
        src = h.get("_source", {})
        score = h.get("_score")
        results.append({
            "chunk_id": src.get("chunk_id") or h.get("_id"),
            "text": src.get("content"),
            "meta": src.get("meta", {}),
            "source_url": src.get("source_url"),
            "page_number": src.get("page_number"),
            "language": src.get("language"),
            "ingest_time": src.get("ingest_time"),
            "distance": float(score) if score is not None else None
        })
    return results

def compute_similarity_from_distance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0

def trust_weight_for(meta: Dict[str, Any]) -> float:
    tl = (meta or {}).get("trust_level") or (meta or {}).get("trust") or ""
    mapping = {"gov": 1.0, "government": 1.0, "implementing_agency": 0.9, "agency": 0.9, "ngo": 0.7, "news": 0.6}
    return float(mapping.get(tl.lower(), 0.5)) if isinstance(tl, str) else 0.5

def freshness_weight(ingest_time_iso: Optional[str]) -> float:
    if not ingest_time_iso:
        return 0.5
    try:
        then = datetime.datetime.fromisoformat(ingest_time_iso)
    except Exception:
        try:
            then = datetime.datetime.strptime(ingest_time_iso, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return 0.5
    age = (datetime.datetime.utcnow() - then).days
    if age <= 0:
        return 1.0
    w = max(0.0, 1.0 - (age / float(FRESHNESS_DAYS)))
    return float(w)

def re_rank_and_select(candidates: List[Dict[str, Any]], final_k: int):
    scored = []
    for c in candidates:
        sim = compute_similarity_from_distance(c.get("distance"))
        tw = trust_weight_for(c.get("meta"))
        fw = freshness_weight(c.get("ingest_time"))
        final_score = sim * tw * fw
        c["similarity"] = sim
        c["trust_weight"] = tw
        c["freshness_weight"] = fw
        c["final_score"] = final_score
        scored.append(c)
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:final_k]

def handler(event, context):
    start_ts = time.time()
    request_id = event.get("request_id") or event.get("requestId") or f"r-{int(start_ts*1000)}"
    query_text = event.get("query") or event.get("question") or ""
    language = event.get("language")
    region = event.get("region")
    top_k = int(event.get("top_k") or FINAL_K)
    raw_k = int(event.get("raw_k") or RAW_K)
    if not query_text or not language:
        jlog({"level":"ERROR","event":"invalid_request","request_id":request_id})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0}
    try:
        embed = embed_text_bedrock(query_text)
    except Exception as e:
        jlog({"level":"ERROR","event":"embed_failed","request_id":request_id,"detail":str(e)})
        raise
    try:
        if VECTOR_DB == "pgvector":
            candidates = pgvector_search(embed, language, region, raw_k)
        else:
            candidates = opensearch_search(embed, language, region, raw_k)
    except Exception as e:
        jlog({"level":"ERROR","event":"vector_search_failed","request_id":request_id,"detail":str(e)})
        raise
    if not candidates:
        jlog({"event":"no_candidates","request_id":request_id})
        return {"request_id": request_id, "passages": [], "chunk_ids": [], "top_similarity": 0.0}
    ranked = re_rank_and_select(candidates, top_k)
    passages = []
    chunk_ids = []
    for i, r in enumerate(ranked):
        passages.append({"number": i+1, "chunk_id": r["chunk_id"], "text": r.get("text") or "", "score": r.get("final_score", 0.0)})
        chunk_ids.append(r["chunk_id"])
    top_similarity = passages[0]["score"] if passages else 0.0
    elapsed = int((time.time() - start_ts)*1000)
    jlog({"event":"retrieval_complete","request_id":request_id,"query_len":len(query_text),"candidates":len(candidates),"returned":len(passages),"top_similarity":top_similarity,"ms":elapsed})
    return {"request_id": request_id, "passages": passages, "chunk_ids": chunk_ids, "top_similarity": float(top_similarity)}

if __name__ == "__main__":
    test_event = {
        "query": os.getenv("TEST_QUERY", "How to apply for voter id?"),
        "language": os.getenv("TEST_LANG", "en"),
        "region": os.getenv("TEST_REGION", None),
        "top_k": int(os.getenv("TEST_TOP_K", "5")),
        "raw_k": int(os.getenv("TEST_RAW_K", "50"))
    }
    print(json.dumps(handler(test_event, None), indent=2))