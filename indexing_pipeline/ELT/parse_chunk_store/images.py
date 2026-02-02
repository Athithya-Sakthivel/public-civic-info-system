#!/usr/bin/env python3
# indexing_pipeline/ELT/images.py
"""
Image parser for indexing pipeline (S3 or local FS).

Implements:
    def parse_file(key: str, manifest: dict) -> dict

Behavior:
- Reads an image file referenced by `key`:
  - when STORAGE="s3": key is S3 object key under S3_BUCKET
  - when STORAGE="fs": key is a local path (absolute or relative)
- Runs OCR using Tesseract (pytesseract). RapidOCR removed.
- Produces page/frame-level chunks (page_number, chunk_index) with used_ocr flag true when OCR produced text.
- Writes chunk output as JSONL to CHUNKED_PREFIX/<schema>/<document_id>.chunks.jsonl
- Writes a chunk manifest CHUNKED_PREFIX/<schema>/<document_id>.chunks.manifest.json
- Returns {"saved_chunks": N} and does not delete raw files.

Env knobs used:
  STORAGE = "s3" | "fs"            (default "s3")
  S3_BUCKET                        (required if STORAGE="s3")
  RAW_PREFIX (default "data/raw/")
  CHUNKED_PREFIX (default "data/chunked/")
  PARSER_VERSION_IMAGE
  FORCE_OVERWRITE
  IMAGE_TESSERACT_LANG (e.g. "eng+tam")
  TESSERACT_CMD (optional, path to tesseract binary)
  MIN_TOKENS_PER_CHUNK, MAX_TOKENS_PER_CHUNK, NUMBER_OF_OVERLAPPING_SENTENCES
  CHUNKED_SCHEMA_VERSION
  TMPDIR
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# --- env/config ---
STORAGE = os.getenv("STORAGE", "s3").strip().lower()  # "s3" or "fs"
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
RAW_PREFIX = (os.getenv("RAW_PREFIX") or os.getenv("STORAGE_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
CHUNKED_PREFIX = (os.getenv("CHUNKED_PREFIX") or os.getenv("STORAGE_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"
PARSER_VERSION_IMAGE = os.getenv("PARSER_VERSION_IMAGE", "images-tesseract-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"

IMAGE_TESSERACT_LANG = os.getenv("IMAGE_TESSERACT_LANG", "eng")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6")

MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
OVERLAP_SENTENCES = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", "2"))
CHUNKED_SCHEMA_VERSION = os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1")
TMPDIR = os.getenv("TMPDIR", None)

# --- simple logger ---
def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def log(level: str, event: str, **extra):
    o = {"ts": now_ts(), "level": level, "event": event}
    o.update(extra)
    line = json.dumps(o, ensure_ascii=False)
    if level.lower() in ("error", "err", "critical"):
        print(line, file=sys.stderr, flush=True)
    else:
        print(line, flush=True)

# --- minimal optional deps ---
try:
    from PIL import Image
except Exception as e:
    Image = None

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    pytesseract = None

try:
    import tiktoken
except Exception:
    tiktoken = None

# --- storage helpers (S3 / FS) ---
def _ensure_s3_client():
    try:
        import boto3
    except Exception as e:
        raise RuntimeError("boto3 required for S3 backend: " + str(e))
    return boto3.client("s3")

def read_raw_bytes(key: str) -> bytes:
    if STORAGE == "s3":
        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET required when STORAGE='s3'")
        s3 = _ensure_s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read()
    else:
        p = Path(key)
        if not p.exists():
            # try under RAW_PREFIX
            candidate = Path(RAW_PREFIX) / Path(key).name
            if candidate.exists():
                p = candidate
        with p.open("rb") as f:
            return f.read()

def write_text_atomic(target_relpath: str, text: str) -> None:
    # target_relpath is path under CHUNKED_PREFIX e.g. "chunked_v1/<doc>.chunks.jsonl"
    full = Path(CHUNKED_PREFIX) / target_relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(full.parent), suffix=".tmp")
    try:
        tmp.write(text.encode("utf-8"))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        Path(tmp.name).replace(full)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

def upload_bytes_s3(key: str, data: bytes) -> None:
    s3 = _ensure_s3_client()
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)

def compute_sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

# --- tokenizer helpers ---
def encode_tokens(text: str):
    if tiktoken is None:
        return text.split()
    try:
        enc = tiktoken.get_encoding(os.getenv("ENC_NAME", "cl100k_base"))
        return enc.encode(text)
    except Exception:
        # fallback
        return text.split()

def decode_tokens(toks):
    if tiktoken is None:
        return " ".join(toks)
    try:
        enc = tiktoken.get_encoding(os.getenv("ENC_NAME", "cl100k_base"))
        return enc.decode(toks)
    except Exception:
        return " ".join(str(x) for x in toks)

# --- sentence splitter (simple) ---
import re
_SENT_RE = re.compile(r'(.+?[\.\?\!]["\']?\s+)|(.+?$)', re.DOTALL)

def sentence_spans(text: str):
    spans = []
    cursor = 0
    for m in _SENT_RE.finditer(text):
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

def split_into_windows(text: str, max_tokens=MAX_TOKENS_PER_CHUNK, min_tokens=MIN_TOKENS_PER_CHUNK, overlap_sentences=OVERLAP_SENTENCES):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        yield {"window_index": 0, "text": "", "token_count": 0, "token_start": 0, "token_end": 0}
        return
    sps = sentence_spans(text)
    sent_items = []
    token_cursor = 0
    for s, sc, ec in sps:
        toks = encode_tokens(s)
        tok_len = len(toks) if isinstance(toks, (list,tuple)) else 1
        sent_items.append({"text": s, "start_char": sc, "end_char": ec, "token_len": tok_len, "tokens": toks})
    if not sent_items:
        toks = encode_tokens(text)
        yield {"window_index": 0, "text": text, "token_count": len(toks) if isinstance(toks,(list,tuple)) else 1, "token_start": 0, "token_end": len(toks) if isinstance(toks,(list,tuple)) else 1}
        return
    for si in sent_items:
        si["token_start_idx"] = token_cursor
        si["token_end_idx"] = token_cursor + si["token_len"]
        token_cursor = si["token_end_idx"]
    windows = []
    i = 0; window_index = 0
    while i < len(sent_items):
        cur_token_count = 0; chunk_texts = []; chunk_token_start = sent_items[i]["token_start_idx"]; chunk_token_end = chunk_token_start; start_i = i
        while i < len(sent_items):
            sent = sent_items[i]; sent_tok_len = sent["token_len"]
            if cur_token_count + sent_tok_len > max_tokens:
                break
            chunk_texts.append(sent["text"]); cur_token_count += sent_tok_len
            chunk_token_end = sent.get("token_end_idx", chunk_token_start + cur_token_count)
            i += 1
        if not chunk_texts:
            # sentence > max_tokens -> truncate by tokens
            sent = sent_items[i]
            toks = sent["tokens"]
            if isinstance(toks, (list,tuple)):
                prefix = toks[:max_tokens]
                try:
                    prefix_text = decode_tokens(prefix)
                except Exception:
                    prefix_text = " ".join(str(x) for x in prefix)
                cur_token_count = len(prefix)
                remainder = toks[max_tokens:]
                if remainder:
                    sent_items[i]["tokens"] = remainder
                    sent_items[i]["token_len"] = len(remainder)
                else:
                    i += 1
                chunk_token_end = chunk_token_start + cur_token_count
                chunk_texts = [prefix_text]
            else:
                chunk_texts = [sent["text"]]
                cur_token_count = sent_tok_len
                i += 1
        chunk_text = " ".join(chunk_texts).strip()
        chunk_meta = {"window_index": window_index, "text": chunk_text, "token_count": int(cur_token_count), "token_start": int(chunk_token_start), "token_end": int(chunk_token_end), "start_sentence_idx": start_i, "end_sentence_idx": i}
        window_index += 1
        if windows and chunk_meta["token_count"] < min_tokens:
            prev = windows[-1]
            prev["text"] = prev["text"] + " " + chunk_meta["text"]
            prev["token_count"] = prev["token_count"] + chunk_meta["token_count"]
            prev["token_end"] = chunk_meta["token_end"]
            prev["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
        else:
            windows.append(chunk_meta)
        i = max(start_i + 1, chunk_meta["end_sentence_idx"] - overlap_sentences)
    for w in windows:
        yield w

def derive_semantic_region(token_start: int, token_end: int, document_total_tokens: int) -> str:
    try:
        if not document_total_tokens or document_total_tokens <= 0:
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

# --- core OCR invocation (pytesseract only) ---
def ocr_pytesseract(image_obj) -> str:
    if pytesseract is None:
        return ""
    try:
        text = pytesseract.image_to_string(image_obj, lang=IMAGE_TESSERACT_LANG, config=TESSERACT_CONFIG)
        return text or ""
    except Exception:
        return ""

# --- file download to temp ---
def write_bytes_to_tempfile(b: bytes, suffix: str = "") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMPDIR)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name

# --- write JSONL and manifest helpers ---
def write_chunks_jsonl(document_id: str, chunks: List[Dict[str, Any]]) -> Tuple[str,int]:
    schema_dir = CHUNKED_SCHEMA_VERSION
    out_basename = f"{document_id}.chunks.jsonl"
    relpath = Path(schema_dir) / out_basename
    text_lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    jsonl_text = "\n".join(text_lines) + "\n"
    write_text_atomic(str(relpath), jsonl_text)
    fullpath = str(Path(CHUNKED_PREFIX) / relpath)
    size = len(jsonl_text.encode("utf-8"))
    return fullpath, size

def write_chunk_manifest(document_id: str, count: int, chunk_file_path: str, raw_path: str, raw_sha: str) -> None:
    schema_dir = CHUNKED_SCHEMA_VERSION
    manifest_basename = f"{document_id}.chunks.manifest.json"
    rel = Path(schema_dir) / manifest_basename
    manifest = {
        "document_id": document_id,
        "chunk_count": count,
        "chunk_file": chunk_file_path,
        "chunk_format": "jsonl",
        "schema_version": CHUNKED_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION_IMAGE,
        "ingest_time": datetime.utcnow().isoformat() + "Z",
        "raw_path": raw_path,
        "raw_sha256": raw_sha
    }
    write_text_atomic(str(rel), json.dumps(manifest, ensure_ascii=False, indent=2))

# --- main parse implementation ---
def parse_file(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        # validate environment
        if STORAGE == "s3" and not S3_BUCKET:
            raise RuntimeError("S3_BUCKET required for STORAGE='s3'")

        # read raw bytes
        try:
            raw_bytes = read_raw_bytes(key)
        except Exception as e:
            tb = traceback.format_exc()
            log("error", "read_failed", key=key, error=str(e), tb=tb)
            manifest.update({"error": str(e), "traceback": tb})
            return {"saved_chunks": 0}

        raw_sha = compute_sha256_bytes(raw_bytes)
        document_id = manifest.get("file_hash") or raw_sha

        # short-circuit: if chunks already exist and not forcing, skip
        schema_basename = f"{document_id}.chunks.jsonl"
        existing_chunk_path = Path(CHUNKED_PREFIX) / CHUNKED_SCHEMA_VERSION / schema_basename
        if existing_chunk_path.exists() and not FORCE_OVERWRITE:
            log("info", "already_chunked", document_id=document_id, path=str(existing_chunk_path))
            return {"saved_chunks": 0}

        # write raw to temp file for PIL
        suffix = Path(key).suffix or ".img"
        tmpfile = write_bytes_to_tempfile(raw_bytes, suffix=suffix)
        if Image is None:
            raise RuntimeError("Pillow not installed (required to open images)")

        try:
            im = Image.open(tmpfile)
        except Exception as e:
            tb = traceback.format_exc()
            log("error", "open_image_failed", key=key, error=str(e), traceback=tb)
            try:
                os.unlink(tmpfile)
            except Exception:
                pass
            manifest.update({"error": "open_failed", "traceback": tb})
            return {"saved_chunks": 0}

        # number of frames/pages
        n_frames = getattr(im, "n_frames", 1)
        chunks: List[Dict[str, Any]] = []
        total_tokens_accum = 0
        for frame_idx in range(n_frames):
            try:
                if n_frames > 1:
                    im.seek(frame_idx)
                frame = im.convert("RGB")
            except Exception:
                frame = im.convert("RGB")

            # run OCR (pytesseract)
            ocr_text = ""
            used_ocr = False
            try:
                ocr_text = ocr_pytesseract(frame)
                if ocr_text and ocr_text.strip():
                    used_ocr = True
                    ocr_text = ocr_text.strip()
            except Exception as e:
                log("warn", "ocr_failed", key=key, frame=frame_idx+1, error=str(e))

            # if no OCR text, still emit an empty chunk for the page for provenance
            if not ocr_text:
                chunk = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_p{frame_idx+1}_1",
                    "chunk_index": 1,
                    "chunk_type": "image_page",
                    "text": "",
                    "token_count": 0,
                    "token_range": [0, 0],
                    "document_total_tokens": 0,
                    "semantic_region": "unknown",
                    "headings": [],
                    "heading_path": [],
                    "layout_tags": ["image"],
                    "figures": [],
                    "source_url": None,
                    "s3_url": f"s3://{S3_BUCKET}/{key}" if STORAGE == "s3" else None,
                    "local_path": key if STORAGE == "fs" else None,
                    "page_number": frame_idx + 1,
                    "language": None,
                    "region": None,
                    "topic_tags": manifest.get("tags", []),
                    "trust_level": manifest.get("trust_level", "gov"),
                    "last_updated": manifest.get("last_updated"),
                    "ingest_time": now_ts(),
                    "parser_version": PARSER_VERSION_IMAGE,
                    "used_ocr": False,
                    "original_manifest": manifest,
                    "provenance": {"raw_sha256": raw_sha, "raw_key": key},
                    "embedding": None
                }
                chunks.append(chunk)
                continue

            # build windows from OCR text
            windows = list(split_into_windows(ocr_text))
            doc_total_tokens = sum(w.get("token_count", 0) for w in windows) or len(ocr_text.split())
            for w in windows:
                idx = int(w.get("window_index", 0))
                chunk_idx = idx + 1
                token_start = int(w.get("token_start", 0))
                token_end = int(w.get("token_end", 0))
                sem = derive_semantic_region(token_start, token_end, doc_total_tokens)
                chunk = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_p{frame_idx+1}_{str(chunk_idx).zfill(4)}",
                    "chunk_index": chunk_idx,
                    "chunk_type": "image_page_chunk",
                    "text": w.get("text", ""),
                    "token_count": int(w.get("token_count", 0)),
                    "token_range": [token_start, token_end],
                    "document_total_tokens": int(doc_total_tokens),
                    "semantic_region": sem,
                    "headings": [],
                    "heading_path": [],
                    "layout_tags": ["image", "ocr"],
                    "figures": [],
                    "source_url": None,
                    "s3_url": f"s3://{S3_BUCKET}/{key}" if STORAGE == "s3" else None,
                    "local_path": key if STORAGE == "fs" else None,
                    "page_number": frame_idx + 1,
                    "language": None,
                    "region": None,
                    "topic_tags": manifest.get("tags", []),
                    "trust_level": manifest.get("trust_level", "gov"),
                    "last_updated": manifest.get("last_updated"),
                    "ingest_time": now_ts(),
                    "parser_version": PARSER_VERSION_IMAGE,
                    "used_ocr": True,
                    "original_manifest": manifest,
                    "provenance": {"raw_sha256": raw_sha, "raw_key": key},
                    "embedding": None
                }
                chunks.append(chunk)
                total_tokens_accum += chunk["token_count"]

        # cleanup temp file
        try:
            os.unlink(tmpfile)
        except Exception:
            pass

        # persist chunks as JSONL
        if not chunks:
            duration_ms = int((time.time() - start) * 1000)
            log("info", "no_chunks", key=key, duration_ms=duration_ms)
            return {"saved_chunks": 0}

        chunk_file_rel, size = None, None
        try:
            chunk_file_path, size = write_chunks_jsonl_and_manifest(document_id, chunks, key, raw_sha)
            duration_ms = int((time.time() - start) * 1000)
            log("info", "written_chunks", document_id=document_id, saved_chunks=len(chunks), chunk_file=chunk_file_path, size_bytes=size, duration_ms=duration_ms)
            return {"saved_chunks": len(chunks)}
        except Exception as e:
            tb = traceback.format_exc()
            log("error", "write_failed", key=key, error=str(e), traceback=tb)
            return {"saved_chunks": 0}
    except Exception as e_outer:
        tb = traceback.format_exc()
        log("error", "parse_error", key=key, error=str(e_outer), traceback=tb)
        return {"saved_chunks": 0}

# helper to write jsonl + manifest together
def write_chunks_jsonl_and_manifest(document_id: str, chunks: List[Dict[str, Any]], raw_key: str, raw_sha: str) -> Tuple[str,int]:
    schema_dir = CHUNKED_SCHEMA_VERSION
    out_basename = f"{document_id}.chunks.jsonl"
    relpath = Path(schema_dir) / out_basename
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    jsonl_text = "\n".join(lines) + "\n"
    write_text_atomic(str(relpath), jsonl_text)
    fullpath = str(Path(CHUNKED_PREFIX) / relpath)
    size = len(jsonl_text.encode("utf-8"))

    # write manifest
    manifest_basename = f"{document_id}.chunks.manifest.json"
    manifest_rel = Path(schema_dir) / manifest_basename
    manifest_obj = {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunk_file": fullpath,
        "chunk_format": "jsonl",
        "schema_version": CHUNKED_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION_IMAGE,
        "ingest_time": datetime.utcnow().isoformat() + "Z",
        "raw_path": (f"s3://{S3_BUCKET}/{raw_key}" if STORAGE == "s3" else raw_key),
        "raw_sha256": raw_sha
    }
    write_text_atomic(str(manifest_rel), json.dumps(manifest_obj, ensure_ascii=False, indent=2))
    return fullpath, size

# expose parse_file as router expects
def parse_file(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    return parse_file_impl(key, manifest) if False else parse_file_main(key, manifest)

# Define parse_file_main to avoid name conflict earlier in module
def parse_file_main(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    return parse_file(key, manifest)  # placeholder to satisfy interface

# Fix: ensure the parse_file symbol points to the actual implementation above
# Replace the previous parse_file with implementation function
def parse_file(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    return parse_file_main_impl(key, manifest)  # will be defined next

# Actual implementation aliasing to previous function body for clarity
def parse_file_main_impl(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    # call the implementation (we placed the implementation earlier under parse_file)
    # To avoid duplicating code, simply call the top-level implementation block by using a wrapper.
    # The actual implementation is contained in the big try/except above (named parse_file in that block).
    # But to keep this file clear and simple, call the core function defined as parse_file_core.
    return _parse_file_core(key, manifest)

# Move the earlier implementation to a named core function for clarity
def _parse_file_core(key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    # replicate the earlier parse logic by calling parse_file (the large block above)
    # For simplicity, call the main logic in parse_file (the first defined one). Python allows reusing names.
    # In practice the router imports parse_file from this module; ensure the function below is the final one.
    # Here we just re-run the implementation defined earlier by keeping one canonical function.
    # To avoid confusion, inline the logic:
    start = time.time()
    try:
        # validate environment
        if STORAGE == "s3" and not S3_BUCKET:
            raise RuntimeError("S3_BUCKET required for STORAGE='s3'")

        # read raw bytes
        try:
            raw_bytes = read_raw_bytes(key)
        except Exception as e:
            tb = traceback.format_exc()
            log("error", "read_failed", key=key, error=str(e), tb=tb)
            manifest.update({"error": str(e), "traceback": tb})
            return {"saved_chunks": 0}

        raw_sha = compute_sha256_bytes(raw_bytes)
        document_id = manifest.get("file_hash") or raw_sha

        # short-circuit: if chunks already exist and not forcing, skip
        schema_basename = f"{document_id}.chunks.jsonl"
        existing_chunk_path = Path(CHUNKED_PREFIX) / CHUNKED_SCHEMA_VERSION / schema_basename
        if existing_chunk_path.exists() and not FORCE_OVERWRITE:
            log("info", "already_chunked", document_id=document_id, path=str(existing_chunk_path))
            return {"saved_chunks": 0}

        # write raw to temp file for PIL
        suffix = Path(key).suffix or ".img"
        tmpfile = write_bytes_to_tempfile(raw_bytes, suffix=suffix)
        if Image is None:
            raise RuntimeError("Pillow not installed (required to open images)")

        try:
            im = Image.open(tmpfile)
        except Exception as e:
            tb = traceback.format_exc()
            log("error", "open_image_failed", key=key, error=str(e), traceback=tb)
            try:
                os.unlink(tmpfile)
            except Exception:
                pass
            manifest.update({"error": "open_failed", "traceback": tb})
            return {"saved_chunks": 0}

        # number of frames/pages
        n_frames = getattr(im, "n_frames", 1)
        chunks: List[Dict[str, Any]] = []
        for frame_idx in range(n_frames):
            try:
                if n_frames > 1:
                    im.seek(frame_idx)
                frame = im.convert("RGB")
            except Exception:
                frame = im.convert("RGB")

            ocr_text = ""
            used_ocr = False
            try:
                ocr_text = ocr_pytesseract(frame)
                if ocr_text and ocr_text.strip():
                    used_ocr = True
                    ocr_text = ocr_text.strip()
            except Exception as e:
                log("warn", "ocr_failed", key=key, frame=frame_idx+1, error=str(e))

            if not ocr_text:
                chunk = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_p{frame_idx+1}_0001",
                    "chunk_index": 1,
                    "chunk_type": "image_page",
                    "text": "",
                    "token_count": 0,
                    "token_range": [0, 0],
                    "document_total_tokens": 0,
                    "semantic_region": "unknown",
                    "headings": [],
                    "heading_path": [],
                    "layout_tags": ["image"],
                    "figures": [],
                    "source_url": None,
                    "s3_url": f"s3://{S3_BUCKET}/{key}" if STORAGE == "s3" else None,
                    "local_path": key if STORAGE == "fs" else None,
                    "page_number": frame_idx + 1,
                    "language": None,
                    "region": None,
                    "topic_tags": manifest.get("tags", []),
                    "trust_level": manifest.get("trust_level", "gov"),
                    "last_updated": manifest.get("last_updated"),
                    "ingest_time": now_ts(),
                    "parser_version": PARSER_VERSION_IMAGE,
                    "used_ocr": False,
                    "original_manifest": manifest,
                    "provenance": {"raw_sha256": raw_sha, "raw_key": key},
                    "embedding": None
                }
                chunks.append(chunk)
                continue

            windows = list(split_into_windows(ocr_text))
            doc_total_tokens = sum(w.get("token_count", 0) for w in windows) or len(ocr_text.split())
            for w in windows:
                idx = int(w.get("window_index", 0))
                chunk_idx = idx + 1
                token_start = int(w.get("token_start", 0))
                token_end = int(w.get("token_end", 0))
                sem = derive_semantic_region(token_start, token_end, doc_total_tokens)
                chunk = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_p{frame_idx+1}_{str(chunk_idx).zfill(4)}",
                    "chunk_index": chunk_idx,
                    "chunk_type": "image_page_chunk",
                    "text": w.get("text", ""),
                    "token_count": int(w.get("token_count", 0)),
                    "token_range": [token_start, token_end],
                    "document_total_tokens": int(doc_total_tokens),
                    "semantic_region": sem,
                    "headings": [],
                    "heading_path": [],
                    "layout_tags": ["image", "ocr"],
                    "figures": [],
                    "source_url": None,
                    "s3_url": f"s3://{S3_BUCKET}/{key}" if STORAGE == "s3" else None,
                    "local_path": key if STORAGE == "fs" else None,
                    "page_number": frame_idx + 1,
                    "language": None,
                    "region": None,
                    "topic_tags": manifest.get("tags", []),
                    "trust_level": manifest.get("trust_level", "gov"),
                    "last_updated": manifest.get("last_updated"),
                    "ingest_time": now_ts(),
                    "parser_version": PARSER_VERSION_IMAGE,
                    "used_ocr": True,
                    "original_manifest": manifest,
                    "provenance": {"raw_sha256": raw_sha, "raw_key": key},
                    "embedding": None
                }
                chunks.append(chunk)

        # cleanup temp file
        try:
            os.unlink(tmpfile)
        except Exception:
            pass

        if not chunks:
            duration_ms = int((time.time() - start) * 1000)
            log("info", "no_chunks", key=key, duration_ms=duration_ms)
            return {"saved_chunks": 0}

        # write chunks + manifest
        chunk_file_path, size = write_chunks_jsonl_and_manifest(document_id, chunks, key, raw_sha)
        duration_ms = int((time.time() - start) * 1000)
        log("info", "write_complete", document_id=document_id, saved_chunks=len(chunks), chunk_file=chunk_file_path, duration_ms=duration_ms)
        return {"saved_chunks": len(chunks)}
    except Exception as e:
        tb = traceback.format_exc()
        log("error", "parse_exception", key=key, error=str(e), traceback=tb)
        return {"saved_chunks": 0}

# Provide parse_file symbol (router expects parse_file)
# parse_file defined above (parse_file_main_impl) is the real implementation; ensure aliasing
# Final parse_file is the function implemented: _parse_file_core
parse_file = _parse_file_core
