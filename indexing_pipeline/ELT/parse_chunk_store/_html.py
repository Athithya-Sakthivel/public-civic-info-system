#!/usr/bin/env python3
"""
indexing_pipeline/ELT/parse_chunk_store/_html.py - Idempotent HTML parser

Assumptions & external contracts:
- Runtime: Python 3.11+ with boto3 available for S3 STORAGE mode (local FS fallback supported).
- STORAGE backend chosen by STORAGE env var: 's3' or 'local'.
- Raw bytes are read from RAW_PREFIX (S3 key or local path). Each raw object has a manifest at <raw_key>.manifest.json.
- Idempotency boundary:
  * Chunk files are content-derived JSONL stored at {CHUNKED_PREFIX}/{CHUNKED_SCHEMA_DIR}/{document_id}.chunks.jsonl
  * Raw manifest updated under <raw_key>.manifest.json with a `chunked` subdocument containing chunked_sha256.
  * If the manifest's `chunked.chunked_sha256` equals computed chunk JSONL sha, no write occurs.
  * All S3 writes are done atomically via a temporary copy -> final pattern.
- Behavior: fail-fast validation of required envs when STORAGE='s3'; minimal mutation to existing metadata; robust logging of decisions.
- Logging: every important decision writes a structured log line (JSON).

Invariant: repeated runs produce no additional writes once a document's chunked_sha256 matches the computed chunk file SHA.

Returns: functions return clear status objects; parse_file(...) returns {"saved_chunks": int, ...}.

"""

from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import traceback
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# optional libs
try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import tiktoken
except Exception:
    tiktoken = None

# -------------------------
# Environment / config (explicit at top)
# -------------------------
STORAGE = os.getenv("STORAGE", "s3").strip().lower()
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
RAW_PREFIX = (os.getenv("RAW_PREFIX") or os.getenv("STORAGE_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
CHUNKED_PREFIX = (os.getenv("CHUNKED_PREFIX") or os.getenv("STORAGE_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"
PARSER_VERSION = os.getenv("PARSER_VERSION", "html-trafilatura-v1")
MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
OVERLAP_SENTENCES = int(os.getenv("OVERLAP_SENTENCES", "2"))
ENC_NAME = os.getenv("ENC_NAME", "cl100k_base")
CHUNKED_SCHEMA_DIR = os.getenv("CHUNKED_SCHEMA_DIR", "chunked_v1").rstrip("/")
FORCE_PROCESS = os.getenv("FORCE_PROCESS", "false").lower() == "true"
PUT_RETRIES = int(os.getenv("PUT_RETRIES", "3"))
PUT_BACKOFF = float(os.getenv("PUT_BACKOFF", "0.3"))

# validations (fail-fast)
if STORAGE not in ("s3", "local"):
    raise SystemExit(f"Invalid STORAGE value: {STORAGE!r}. Expected 's3' or 'local'.")

if STORAGE == "s3" and not S3_BUCKET:
    raise SystemExit("S3_BUCKET env required when STORAGE='s3'")

if not CHUNKED_SCHEMA_DIR:
    raise SystemExit("CHUNKED_SCHEMA_DIR must be set and non-empty (use e.g. 'chunked_v1').")

# -------------------------
# Logging helpers
# -------------------------
def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def log(level: str, event: str, **extra):
    obj = {"ts": now_ts(), "level": level, "event": event}
    obj.update(extra)
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if level.lower() in ("error", "err", "critical"):
        print(line, file=sys.stderr, flush=True)
    else:
        print(line, flush=True)

# -------------------------
# S3 helpers (atomic writes)
# -------------------------
def _ensure_s3_client():
    try:
        import boto3
    except Exception as e:
        raise RuntimeError("boto3 is required for S3 STORAGE backend: " + str(e))
    return boto3.client("s3")

def _s3_atomic_put(final_key: str, data: bytes, content_type: Optional[str] = "application/octet-stream"):
    """
    Atomic write to S3: put temp key, copy to final key, delete temp key.
    Retries PUT_RETRIES times on transient errors.
    final_key must be the S3 object key (no bucket prefix).
    """
    client = _ensure_s3_client()
    temp_key = f"{final_key}.tmp.{os.getpid()}.{int(time.time()*1000)}"
    last_exc = None
    for attempt in range(1, PUT_RETRIES + 1):
        try:
            client.put_object(Bucket=S3_BUCKET, Key=temp_key, Body=data, ContentType=content_type)
            client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': temp_key}, Key=final_key)
            client.delete_object(Bucket=S3_BUCKET, Key=temp_key)
            return
        except Exception as e:
            last_exc = e
            # best-effort cleanup
            try:
                client.delete_object(Bucket=S3_BUCKET, Key=temp_key)
            except Exception:
                pass
            if attempt < PUT_RETRIES:
                time.sleep(PUT_BACKOFF * attempt)
                continue
            raise last_exc

# -------------------------
# Chunked write helpers (atomic for both s3 and local)
# -------------------------
def write_text_atomic_to_chunked(target_key: str, text: str) -> None:
    target_key = target_key.lstrip("/")
    final_key = f"{CHUNKED_PREFIX.rstrip('/')}/{target_key}"
    if STORAGE == "s3":
        _s3_atomic_put(final_key, text.encode("utf-8"), content_type="application/json")
    else:
        p = Path(final_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)

def write_bytes_atomic_to_chunked(target_key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    target_key = target_key.lstrip("/")
    final_key = f"{CHUNKED_PREFIX.rstrip('/')}/{target_key}"
    if STORAGE == "s3":
        _s3_atomic_put(final_key, data, content_type=content_type)
    else:
        p = Path(final_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)

def s3_url_for_raw(key: str) -> Optional[str]:
    if STORAGE == "s3" and S3_BUCKET:
        return f"s3://{S3_BUCKET}/{key}"
    return None

def compute_sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

# -------------------------
# Text canonicalization / tokenization
# -------------------------
def canonicalize_text(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

_sentence_re = re.compile(r'(.+?[\.\?\!\n]+)|(.+?$)', re.DOTALL)

def sentence_spans(text: str) -> List[Tuple[str,int,int]]:
    spans = []
    cursor = 0
    for m in _sentence_re.finditer(text):
        s = (m.group(1) or m.group(2) or "").strip()
        if not s:
            continue
        start = text.find(s, cursor)
        if start == -1:
            start = cursor
        end = start + len(s)
        spans.append((s, start, end))
        cursor = end
    return spans

def get_tokenizer():
    if tiktoken is None:
        return None
    try:
        enc = tiktoken.get_encoding(ENC_NAME)
        return enc
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

def encode_tokens(text: str):
    enc = get_tokenizer()
    if enc is None:
        return text.split()
    try:
        return enc.encode(text)
    except Exception:
        return text.split()

def decode_tokens(tokens) -> str:
    enc = get_tokenizer()
    if enc is None:
        return " ".join(tokens)
    try:
        return enc.decode(tokens)
    except Exception:
        if isinstance(tokens, list):
            return " ".join(str(t) for t in tokens)
        return str(tokens)

def split_into_windows(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK,
                       min_tokens: int = MIN_TOKENS_PER_CHUNK,
                       overlap_sentences: int = OVERLAP_SENTENCES):
    text = canonicalize_text(text)
    if not text:
        yield {"window_index": 0, "text": "", "token_count": 0, "token_start": 0, "token_end": 0}
        return
    spans = sentence_spans(text)
    sent_items = []
    token_cursor = 0
    for s, sc, ec in spans:
        toks = encode_tokens(s)
        tok_len = len(toks) if isinstance(toks, (list,tuple)) else 1
        sent_items.append({"text": s, "start_char": sc, "end_char": ec, "token_len": tok_len, "tokens": toks})
    if not sent_items:
        toks = encode_tokens(text)
        yield {"window_index": 0, "text": text, "token_count": len(toks) if isinstance(toks, (list,tuple)) else 1, "token_start": 0, "token_end": len(toks) if isinstance(toks, (list,tuple)) else 1}
        return
    for si in sent_items:
        si["token_start_idx"] = token_cursor
        si["token_end_idx"] = token_cursor + si["token_len"]
        token_cursor = si["token_end_idx"]
    windows = []
    i = 0
    window_index = 0
    while i < len(sent_items):
        cur_token_count = 0
        chunk_sent_texts = []
        chunk_token_start = sent_items[i]["token_start_idx"]
        chunk_token_end = chunk_token_start
        start_i = i
        while i < len(sent_items):
            sent = sent_items[i]
            if cur_token_count + sent["token_len"] > max_tokens:
                break
            chunk_sent_texts.append(sent["text"])
            cur_token_count += sent["token_len"]
            chunk_token_end = sent.get("token_end_idx", chunk_token_start + cur_token_count)
            i += 1
        if not chunk_sent_texts:
            sent = sent_items[i]
            toks = sent["tokens"]
            if isinstance(toks, (list,tuple)):
                truncated = toks[:max_tokens]
                chunk_text = decode_tokens(truncated)
                cur_token_count = len(truncated)
                remaining = toks[max_tokens:]
                if remaining:
                    sent_items[i]["tokens"] = remaining
                    sent_items[i]["token_len"] = len(remaining)
                    sent_items[i]["token_start_idx"] = None
                    sent_items[i]["token_end_idx"] = None
                else:
                    i += 1
            else:
                chunk_text = sent["text"]
                cur_token_count = sent.get("token_len", 1)
                i += 1
            chunk_token_end = chunk_token_start + cur_token_count
        else:
            chunk_text = " ".join(chunk_sent_texts).strip()
        chunk_meta = {
            "window_index": window_index,
            "text": chunk_text,
            "token_count": int(cur_token_count),
            "token_start": int(chunk_token_start),
            "token_end": int(chunk_token_end),
            "start_sentence_idx": int(start_i),
            "end_sentence_idx": int(i)
        }
        window_index += 1
        if windows and chunk_meta["token_count"] < min_tokens:
            prev = windows[-1]
            prev["text"] = prev["text"] + " " + chunk_meta["text"]
            prev["token_count"] = prev["token_count"] + chunk_meta["token_count"]
            prev["token_end"] = chunk_meta["token_end"]
            prev["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
        else:
            windows.append(chunk_meta)
        new_i = max(start_i + 1, (chunk_meta["end_sentence_idx"] - overlap_sentences))
        i = new_i
    for w in windows:
        yield w

def derive_semantic_region(token_start: int, token_end: int, document_total_tokens: int) -> str:
    try:
        if document_total_tokens is None or document_total_tokens <= 0:
            return "unknown"
        ratio = float(token_start) / float(document_total_tokens)
        if ratio < 0.10:
            return "intro"
        if ratio < 0.30:
            return "early"
        if ratio < 0.70:
            return "middle"
        if ratio < 0.90:
            return "late"
        return "footer"
    except Exception:
        return "unknown"

# -------------------------
# Chunk path helpers & idempotency checks
# -------------------------
def _chunk_paths_for_document(document_id: str) -> Tuple[str, str]:
    jsonl = f"{CHUNKED_SCHEMA_DIR}/{document_id}.chunks.jsonl"
    return jsonl, ""

def _exists_chunk_for_document(document_id: str) -> bool:
    jsonl, _ = _chunk_paths_for_document(document_id)
    full_key = f"{CHUNKED_PREFIX.rstrip('/')}/{jsonl}"
    if STORAGE == "s3":
        try:
            s3 = _ensure_s3_client()
            s3.head_object(Bucket=S3_BUCKET, Key=full_key)
            return True
        except Exception:
            return False
    else:
        p = Path(full_key)
        return p.exists()

def _raw_manifest_key_for_raw_key(raw_key: str) -> str:
    # raw_key is the S3 key (relative) or local path; manifest stored alongside raw
    if STORAGE == "s3":
        # raw_key is already a key; append .manifest.json
        return raw_key + ".manifest.json"
    p = Path(raw_key)
    if p.exists():
        return str(p.with_suffix(p.suffix + ".manifest.json"))
    return str(Path(RAW_PREFIX) / (p.name + ".manifest.json"))

def _read_raw_manifest(raw_manifest_key: str) -> Dict[str, Any]:
    if STORAGE == "s3":
        s3 = _ensure_s3_client()
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=raw_manifest_key)
            body = resp["Body"].read()
            return json.loads(body.decode("utf-8"))
        except Exception:
            return {}
    else:
        try:
            with open(raw_manifest_key, "rb") as fh:
                return json.load(fh)
        except Exception:
            return {}

def _write_raw_manifest(raw_manifest_key: str, manifest_obj: Dict[str, Any]) -> None:
    if STORAGE == "s3":
        s3 = _ensure_s3_client()
        body = json.dumps(manifest_obj, ensure_ascii=False).encode("utf-8")
        _s3_atomic_put(raw_manifest_key, body, content_type="application/json")
    else:
        p = Path(raw_manifest_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(json.dumps(manifest_obj, ensure_ascii=False, indent=2).encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(p)

def _write_chunks_and_extend_raw_manifest(document_id: str, chunks: List[Dict[str, Any]], raw_key: str, raw_sha: str, original_url: Optional[str] = None) -> Tuple[str,int,str]:
    """
    Writes chunk JSONL to CHUNKED_PREFIX under CHUNKED_SCHEMA_DIR/document_id.chunks.jsonl
    Idempotency guarantee:
      - compute jsonl_text and its sha locally first
      - if existing raw manifest has identical chunked_sha256 -> skip all writes
      - otherwise, perform atomic write of chunk file then extend raw manifest atomically
    Returns: (chunk_file_location, size_bytes, chunked_sha)
    """
    jsonl_rel, _ = _chunk_paths_for_document(document_id)
    jsonl_text = "\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in chunks) + "\n"
    jsonl_bytes = jsonl_text.encode("utf-8")
    sha = compute_sha256_bytes(jsonl_bytes)
    size = len(jsonl_bytes)

    # Check raw manifest first to avoid unnecessary uploads
    raw_manifest_key = _raw_manifest_key_for_raw_key(raw_key)
    existing = _read_raw_manifest(raw_manifest_key) or {}
    existing_chunked = existing.get("chunked", {})
    if existing_chunked and existing_chunked.get("chunked_sha256") == sha and not FORCE_PROCESS:
        log("info", "raw_manifest_already_up_to_date", raw_manifest=raw_manifest_key, chunk_rel=jsonl_rel, chunked_sha256=sha)
        if STORAGE == "s3":
            chunk_file = f"s3://{S3_BUCKET}/{CHUNKED_PREFIX.rstrip('/')}/{jsonl_rel}"
        else:
            chunk_file = str(Path(CHUNKED_PREFIX) / jsonl_rel)
        return chunk_file, size, sha

    # Not up-to-date -> write chunk file atomically then update manifest
    # Attempt atomic write of chunk JSONL
    try:
        write_text_atomic_to_chunked(jsonl_rel, jsonl_text)
    except Exception as e:
        log("error", "chunk_write_failed", raw_key=raw_key, error=str(e))
        raise

    if STORAGE == "s3":
        chunk_file = f"s3://{S3_BUCKET}/{CHUNKED_PREFIX.rstrip('/')}/{jsonl_rel}"
    else:
        chunk_file = str(Path(CHUNKED_PREFIX) / jsonl_rel)

    # Extend raw manifest atomically with chunked metadata (idempotent update)
    chunked_meta = {
        "chunk_file": chunk_file,
        "chunk_format": "jsonl",
        "schema_version": CHUNKED_SCHEMA_DIR,
        "parser_version": PARSER_VERSION,
        "ingest_time": now_ts(),
        "chunk_count": len(chunks),
        "chunked_sha256": sha,
        "chunked_size_bytes": size
    }
    existing.setdefault("file_hash", existing.get("file_hash") or raw_sha)
    existing.setdefault("timestamp", existing.get("timestamp") or now_ts())
    existing["parser_version"] = PARSER_VERSION
    if original_url:
        existing["original_url"] = original_url
    existing["chunked"] = chunked_meta
    existing["saved_chunks"] = len(chunks)
    existing["chunked_manifest_written_at"] = now_ts()
    # atomic manifest write
    try:
        _write_raw_manifest(raw_manifest_key, existing)
    except Exception as e:
        log("error", "raw_manifest_write_failed", raw_manifest=raw_manifest_key, error=str(e))
        raise
    log("info", "raw_manifest_extended", raw_manifest=raw_manifest_key, chunk_file=chunk_file, chunks=len(chunks), sha256=sha, size=size)
    return chunk_file, size, sha

# -------------------------
# parse_file: main parser entry for a single raw key
# -------------------------
def parse_file(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single raw object referenced by 'key' (S3 key or local path).
    manifest: dict parsed from the raw manifest (may be empty).
    Returns: {"saved_chunks": int, ...}
    """
    start_ts = time.time()
    try:
        try:
            raw_bytes = read_raw_bytes(key)
        except Exception as e:
            tb = traceback.format_exc()
            manifest.update({"error": f"read_failed: {str(e)}", "traceback": tb})
            log("error", "read_failed", key=key, error=str(e))
            return {"saved_chunks": 0}

        raw_sha = compute_sha256_bytes(raw_bytes)
        document_id = manifest.get("file_hash") or raw_sha

        # If chunks already exist and not forcing process -> skip
        if not FORCE_PROCESS and _exists_chunk_for_document(document_id):
            duration = int((time.time() - start_ts) * 1000)
            log("info", "skip_existing_chunks", document_id=document_id, key=key, duration_ms=duration)
            return {"saved_chunks": 0, "skipped": True}

        # decode heuristics
        text_candidate = None
        try:
            text_candidate = raw_bytes.decode("utf-8")
        except Exception:
            try:
                text_candidate = raw_bytes.decode("latin-1")
            except Exception:
                text_candidate = None

        extracted_text = None
        parsed_meta = {}
        title = None

        # preferentially use trafilatura if available and input is HTML/text
        if trafilatura is not None and text_candidate is not None:
            try:
                json_doc = trafilatura.extract(text_candidate, output_format="json", with_metadata=True)
                if json_doc:
                    parsed_meta = json.loads(json_doc)
                    extracted_text = parsed_meta.get("text") or parsed_meta.get("excerpt") or parsed_meta.get("body") or ""
                    title = parsed_meta.get("title") or title
            except Exception:
                try:
                    extracted_text = trafilatura.extract(text_candidate, output_format="text")
                except Exception:
                    extracted_text = None

        # fallback to BeautifulSoup extraction
        if not extracted_text and BeautifulSoup is not None and text_candidate is not None:
            try:
                soup = BeautifulSoup(text_candidate, "html.parser")
                if not manifest.get("source_url"):
                    can = soup.find("link", rel="canonical")
                    if can and can.get("href"):
                        manifest["source_url"] = can.get("href")
                if not title:
                    t = soup.title.string if soup.title and soup.title.string else None
                    title = t or title
                paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "li"])]
                extracted_text = "\n\n".join([p for p in paras if p])
            except Exception:
                extracted_text = None

        # final fallback: treat bytes as text if available
        if not extracted_text:
            if text_candidate:
                extracted_text = text_candidate
            else:
                manifest.update({"error": "no_extractable_text"})
                log("warn", "no_text", key=key)
                return {"saved_chunks": 0}

        canonical_text = canonicalize_text(extracted_text)
        try:
            token_list = encode_tokens(canonical_text)
            total_tokens = len(token_list) if isinstance(token_list, (list, tuple)) else 1
        except Exception:
            total_tokens = len(canonical_text.split())

        windows = list(split_into_windows(canonical_text, max_tokens=MAX_TOKENS_PER_CHUNK,
                                          min_tokens=MIN_TOKENS_PER_CHUNK,
                                          overlap_sentences=OVERLAP_SENTENCES))
        chunks: List[Dict[str, Any]] = []
        headings = []
        if isinstance(parsed_meta, dict):
            h = parsed_meta.get("title")
            if h:
                headings.append(h)
        if title:
            if title not in headings:
                headings.insert(0, title)

        source_url_authoritative = manifest.get("original_url") or manifest.get("source_url") or (parsed_meta.get("url") if isinstance(parsed_meta, dict) else None) or s3_url_for_raw(key) or (key if STORAGE != "s3" else None)
        provenance_base = {"raw_sha256": raw_sha, "raw_key": key, "original_url": manifest.get("original_url")}

        for w in windows:
            idx = int(w.get("window_index", 0))
            chunk_index = idx + 1
            chunk_id = f"{document_id}_c{str(chunk_index).zfill(4)}"
            token_start = int(w.get("token_start", 0))
            token_end = int(w.get("token_end", 0))
            sem = derive_semantic_region(token_start, token_end, total_tokens)
            chunk = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_type": "token_window" if len(windows) > 1 else "page",
                "text": w.get("text", ""),
                "token_count": int(w.get("token_count", 0)),
                "token_range": [token_start, token_end],
                "document_total_tokens": int(total_tokens),
                "semantic_region": sem,
                "headings": headings,
                "heading_path": headings,
                "layout_tags": ["html"],
                "figures": [],
                "source_url": source_url_authoritative,
                "source_domain": None,
                "s3_url": s3_url_for_raw(key),
                "local_path": key if STORAGE != "s3" else None,
                "language": parsed_meta.get("language") if isinstance(parsed_meta, dict) else None,
                "region": None,
                "topic_tags": manifest.get("tags", []),
                "trust_level": manifest.get("trust_level", "gov"),
                "last_updated": manifest.get("last_updated") or None,
                "ingest_time": now_ts(),
                "parser_version": PARSER_VERSION,
                "used_ocr": False,
                "original_manifest": manifest,
                "provenance": provenance_base,
                "embedding": None
            }
            try:
                src = chunk["source_url"] or chunk["s3_url"] or chunk["local_path"]
                if src:
                    from urllib.parse import urlparse
                    parsed = urlparse(src)
                    domain = parsed.netloc or parsed.path.split("/")[0]
                    chunk["source_domain"] = domain
            except Exception:
                chunk["source_domain"] = None
            chunks.append(chunk)

        if not chunks:
            duration_ms = int((time.time() - start_ts) * 1000)
            log("info", "no_chunks", key=key, document_id=document_id, duration_ms=duration_ms)
            return {"saved_chunks": 0}

        # Race check: after windows computed, if another process wrote chunks concurrently, avoid double-writes
        if not FORCE_PROCESS and _exists_chunk_for_document(document_id):
            duration = int((time.time() - start_ts) * 1000)
            log("info", "race_skip_existing", document_id=document_id, key=key, duration_ms=duration)
            return {"saved_chunks": 0, "skipped": True}

        # Attempt to write chunk file and extend manifest (idempotent)
        try:
            chunk_file, size, sha = _write_chunks_and_extend_raw_manifest(document_id, chunks, key, raw_sha, manifest.get("original_url"))
        except Exception as e:
            tb = traceback.format_exc()
            log("error", "chunk_write_failed", key=key, error=str(e), traceback=tb)
            return {"saved_chunks": 0, "error": str(e)}

        duration_ms = int((time.time() - start_ts) * 1000)
        log("info", "html_parsed", key=key, document_id=document_id, saved_chunks=len(chunks), chunk_file=chunk_file, chunk_size_bytes=size, chunk_sha256=sha, duration_ms=duration_ms)
        return {"saved_chunks": len(chunks)}
    except Exception as e:
        tb = traceback.format_exc()
        log("error", "write_failed", key=key, error=str(e), traceback=tb)
        return {"saved_chunks": 0}

# -------------------------
# helpers used by parse_file but defined after to keep file structured
# -------------------------
def read_raw_bytes(key: str) -> bytes:
    """
    Read raw bytes from S3 or local path.
    key: for S3 - object key under S3_BUCKET; for local - filesystem path or relative path under RAW_PREFIX.
    """
    if STORAGE == "s3":
        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET env required for STORAGE=s3")
        s3 = _ensure_s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read()
    else:
        p = Path(key)
        if not p.exists():
            p2 = Path(RAW_PREFIX) / Path(key).name
            if p2.exists():
                p = p2
        with p.open("rb") as f:
            return f.read()

# -------------------------
# Validation helpers
# -------------------------
def manifest_validator_minimal(man: Dict[str, Any]) -> None:
    required = ["file_hash", "mime_ext", "original_url", "timestamp"]
    missing = [k for k in required if k not in man]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    if not isinstance(man.get("file_hash"), str) or len(man.get("file_hash", "")) < 8:
        raise ValueError("invalid file_hash")
    try:
        _ = datetime.fromisoformat(man["timestamp"].replace("Z", "+00:00"))
    except Exception:
        raise ValueError("timestamp not ISO8601")

def anchor_implies_document(href: str, anchor_text: str, attrs: Dict[str, str]) -> bool:
    txt = (anchor_text or "").lower()
    for kw in ("download", "pdf", "guideline", "application", "form", "gazette", "notification", "circular"):
        if kw in txt:
            return True
    href_l = (href or "").lower()
    if any(p in href_l for p in ("/documents/", "/uploads/", "/files/", "/pdf", "/download", "attachment")):
        return True
    if any(href_l.endswith("." + e) for e in ["pdf","doc","docx","xls","xlsx","csv","json","zip","html","htm"]):
        return True
    return False

def is_allowed_domain(url: str) -> bool:
    # placeholder: real allowed-domain logic is outside this module
    return True

# -------------------------
# When run as script, no-op (the pipeline calls parse_file)
# -------------------------
if __name__ == "__main__":
    log("info", "module_loaded", module=__file__, parser_version=PARSER_VERSION)

