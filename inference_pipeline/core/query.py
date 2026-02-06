#!/usr/bin/env python3
"""
inference_pipeline/core/query.py

Core orchestration for inference pipeline (validation, policy, retrieval, generation, audit).

Assumptions & invariants (fail-fast at import):
* Python 3.11 runtime.
* core/retriever.py exposes retrieve(event: dict) -> dict with keys:
    { request_id, passages: [{number, chunk_id, text, meta, source_url, ...}], chunk_ids, top_similarity }
* core/generator.py exposes generate(event: dict) -> dict with decisions:
    - ACCEPT -> {'request_id', 'decision':'ACCEPT', 'answer_lines':[{'text':...}], 'confidence':...}
    - NOT_ENOUGH_INFORMATION -> {'request_id', 'decision':'NOT_ENOUGH_INFORMATION'}
    - INVALID_OUTPUT -> {'request_id', 'decision':'INVALID_OUTPUT'}
* Bedrock embedding/generation budgets: embed+search <= EMBED_SEARCH_BUDGET, generation <= GEN_BUDGET (logged/enforced heuristically).
* Vector store contract (pgvector) and indexing assumed satisfied by indexing pipeline.
* Audit sink: optional S3 bucket (AUDIT_S3_BUCKET). If present, each request writes a JSON audit record under prefix `audits/{date}/{request_id}.json`.
* All env vars are read and validated at import to avoid drift.

External contracts (request/response):
* Canonical request (from adapters) into core.query.handler:
  {
    "session_id": "uuid",          # optional but recommended
    "request_id": "uuid",          # optional
    "language": "en|hi|ta",        # REQUIRED
    "channel": "web|sms|voice",    # REQUIRED
    "query": "text",               # REQUIRED
    "region": "tn",                # optional
    "top_k": int,                  # optional
    "raw_k": int,                  # optional
    "asr_confidence": float,       # required for channel=voice
  }
* Core response:
  {
    "request_id": "...",
    "resolution": "answer" | "refusal" | "not_enough_info" | "invalid_output",
    "answer_lines": [{ "text": "..." }],         # when resolution == "answer"
    "citations": [{"citation": 1, "chunk_id":"c_123","source_url":"...","meta":{...}}], # when answer
    "confidence": "high" | "low",
    "guidance_key": "..."                         # when refusal
  }

Operational knobs (envs and defaults):
* MIN_SIMILARITY=0.60
* ASR_CONF_THRESHOLD=0.75
* EMBED_SEARCH_BUDGET_SEC=2.5
* GEN_BUDGET_SEC=4.0
* AUDIT_S3_BUCKET (optional) -- write audit records when present

Design goals:
* Centralize policy decisions (intent blocklist, ASR gating) in this module.
* Deterministic control flow with explicit logs for every decision.
* Fail-closed for safety-sensitive steps.
* Keep channel adapters thin (they should only normalize/forward).
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import datetime
import re
from typing import Any, Dict, List, Optional

# Structured JSON logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("core.query")


def jlog(obj: Dict[str, Any]) -> None:
    base = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "svc": "core.query",
    }
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))


# ---- Fail-fast imports ----
try:
    import boto3
except Exception as e:
    jlog({"level": "CRITICAL", "event": "boto3_missing", "detail": str(e)})
    raise SystemExit(2)

try:
    # core modules (local package)
    from core import retriever, generator  # type: ignore
except Exception as e:
    jlog({"level": "CRITICAL", "event": "core_import_failed", "detail": str(e)})
    raise SystemExit(3)


# ---- Env knobs (centralized & validated) ----
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


AWS_REGION = _env("AWS_REGION")
if not AWS_REGION:
    jlog({"level": "CRITICAL", "event": "env_missing", "hint": "AWS_REGION must be set"})
    raise SystemExit(10)

MIN_SIMILARITY = float(_env("MIN_SIMILARITY", "0.60"))
ASR_CONF_THRESHOLD = float(_env("ASR_CONF_THRESHOLD", "0.75"))

EMBED_SEARCH_BUDGET_SEC = float(_env("EMBED_SEARCH_BUDGET_SEC", "2.5"))
GEN_BUDGET_SEC = float(_env("GEN_BUDGET_SEC", "4.0"))
METADATA_FETCH_BUDGET_SEC = float(_env("METADATA_FETCH_BUDGET_SEC", "0.3"))

AUDIT_S3_BUCKET = _env("AUDIT_S3_BUCKET")  # optional
AUDIT_S3_PREFIX = _env("AUDIT_S3_PREFIX", "audits/")

ALLOWED_LANGUAGES = set(["en", "hi", "ta"])
ALLOWED_CHANNELS = set(["web", "sms", "voice"])

# Intent blocklist (very small/safe heuristics). If matched -> deterministic refusal.
INTENT_BLOCKLIST_PATTERNS = {
    "medical": re.compile(r"\b(medic(al|ine)|prescribe|diagnos|symptom|pill|dosage)\b", re.I),
    "legal": re.compile(r"\b(attorney|sue|lawsuit|contract|custody|divorce|legal advice|crime)\b", re.I),
}

# Guidance keys returned when refusing
GUIDANCE_KEYS = {
    "medical": "refusal_medical",
    "legal": "refusal_legal",
    "asr_low_confidence": "refusal_asr_low_confidence",
    "insufficient_evidence": "refusal_insufficient_evidence",
}

# AWS clients (lazy)
_s3_client = None


def init_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    except Exception as e:
        jlog({"level": "WARN", "event": "s3_client_init_failed", "detail": str(e)})
        _s3_client = None
    return _s3_client


# ---- Helper validators ----
def _validate_request_shape(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Basic presence checks and normalization
    req_id = ev.get("request_id") or f"r-{int(time.time() * 1000)}"
    language = (ev.get("language") or "").strip().lower()
    channel = (ev.get("channel") or "").strip().lower()
    query_text = (ev.get("query") or ev.get("question") or "").strip()
    # top_k/raw_k optional ints
    try:
        top_k = int(ev.get("top_k")) if ev.get("top_k") is not None else None
    except Exception:
        top_k = None
    try:
        raw_k = int(ev.get("raw_k")) if ev.get("raw_k") is not None else None
    except Exception:
        raw_k = None

    if not language or language not in ALLOWED_LANGUAGES:
        return {"error": "invalid_language", "request_id": req_id}
    if not channel or channel not in ALLOWED_CHANNELS:
        return {"error": "invalid_channel", "request_id": req_id}
    if not query_text:
        return {"error": "empty_query", "request_id": req_id}
    if channel == "voice":
        asr_conf = ev.get("asr_confidence")
        try:
            asr_conf = float(asr_conf) if asr_conf is not None else None
        except Exception:
            asr_conf = None
        if asr_conf is None:
            return {"error": "missing_asr_confidence", "request_id": req_id}
    else:
        asr_conf = None

    return {
        "request_id": req_id,
        "session_id": ev.get("session_id"),
        "language": language,
        "channel": channel,
        "query": query_text,
        "top_k": top_k,
        "raw_k": raw_k,
        "asr_confidence": asr_conf,
        "region": ev.get("region"),
    }


def _intent_blocked(query: str) -> Optional[str]:
    """Return guidance_key if intent falls into a blocked class."""
    for k, pat in INTENT_BLOCKLIST_PATTERNS.items():
        if pat.search(query):
            return GUIDANCE_KEYS.get(k)
    return None


def _enforce_asr(asr_confidence: Optional[float]) -> Optional[str]:
    if asr_confidence is None:
        return None
    if float(asr_confidence) < ASR_CONF_THRESHOLD:
        return GUIDANCE_KEYS["asr_low_confidence"]
    return None


def _top_similarity_from_retrieval(res: Dict[str, Any]) -> float:
    try:
        return float(res.get("top_similarity") or 0.0)
    except Exception:
        return 0.0


# ---- Metadata hydration (best-effort) ----
def _hydrate_citation_metadata(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build citations array from passages. We use passage.meta and source_url returned by retriever.
    This is best-effort and intentionally light-weight (no extra network calls).
    """
    citations = []
    for p in passages:
        citations.append(
            {
                "citation": p.get("number"),
                "chunk_id": p.get("chunk_id"),
                "source_url": p.get("source_url"),
                "meta": p.get("meta") or {},
            }
        )
    return citations


# ---- Audit write (S3) ----
def _write_audit(record: Dict[str, Any]) -> None:
    if not AUDIT_S3_BUCKET:
        # audit disabled
        jlog({"event": "audit_skipped", "reason": "no_audit_bucket"})
        return
    try:
        client = init_s3_client()
        if client is None:
            jlog({"event": "audit_skipped", "reason": "s3_client_unavailable"})
            return
        date_prefix = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        key = f"{AUDIT_S3_PREFIX.rstrip('/')}/{date_prefix}/{record.get('request_id')}.json"
        body = json.dumps(record, default=str)
        client.put_object(Bucket=AUDIT_S3_BUCKET, Key=key, Body=body.encode("utf-8"))
        jlog({"event": "audit_written", "request_id": record.get("request_id"), "s3_key": key})
    except Exception as e:
        jlog({"level": "WARN", "event": "audit_write_failed", "detail": str(e), "request_id": record.get("request_id")})


# ---- Orchestration: handle a single canonical request ----
def handle(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main core entrypoint used internally by adapters: validates, enforces policy, calls retriever and generator,
    hydrates citations, writes audit, and returns a structured response.

    Returns canonical response object described in module docstring.
    """
    start_time = time.time()
    validated = _validate_request_shape(event)
    if not validated or "error" in validated:
        req_id = (validated or {}).get("request_id") or f"r-{int(time.time() * 1000)}"
        jlog({"level": "ERROR", "event": "invalid_request_shape", "detail": validated, "request_id": req_id})
        return {"request_id": req_id, "resolution": "refusal", "guidance_key": "invalid_request", "reason": validated}

    request_id = validated["request_id"]
    session_id = validated.get("session_id")
    language = validated["language"]
    channel = validated["channel"]
    query_text = validated["query"]
    top_k = validated["top_k"]
    raw_k = validated["raw_k"]

    jlog({"event": "request_start", "request_id": request_id, "session_id": session_id, "language": language, "channel": channel})

    # 1) ASR gating for voice
    if channel == "voice":
        asr_conf = validated.get("asr_confidence")
        asr_refusal = _enforce_asr(asr_conf)
        if asr_refusal:
            jlog({"event": "refuse_asr", "request_id": request_id, "asr_confidence": asr_conf})
            _write_audit(
                {
                    "session_id": session_id,
                    "request_id": request_id,
                    "language": language,
                    "channel": channel,
                    "used_chunk_ids": [],
                    "resolution": "refusal",
                    "guidance_key": asr_refusal,
                    "timing_ms": int((time.time() - start_time) * 1000),
                }
            )
            return {"request_id": request_id, "resolution": "refusal", "guidance_key": asr_refusal}

    # 2) Intent classification (blocklist)
    guidance = _intent_blocked(query_text)
    if guidance:
        jlog({"event": "intent_blocked", "request_id": request_id, "guidance_key": guidance})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": [],
                "resolution": "refusal",
                "guidance_key": guidance,
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "refusal", "guidance_key": guidance}

    # 3) Retrieval
    retrieval_ev = {
        "request_id": request_id,
        "query": query_text,
        "top_k": top_k,
        "raw_k": raw_k,
        "filters": event.get("filters", {}),
    }
    t_retr_start = time.time()
    try:
        retr_res = retriever.retrieve(retrieval_ev)
    except Exception as e:
        jlog({"level": "ERROR", "event": "retriever_exception", "request_id": request_id, "detail": str(e)})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": [],
                "resolution": "invalid_output",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "invalid_output", "error": "retrieval_failed"}
    t_retr_ms = int((time.time() - t_retr_start) * 1000)
    jlog({"event": "retriever_returned", "request_id": request_id, "retrieval_ms": t_retr_ms, "top_similarity": retr_res.get("top_similarity")})

    # Enforce retrieval budgets heuristically
    if t_retr_ms / 1000.0 > EMBED_SEARCH_BUDGET_SEC:
        jlog({"level": "WARN", "event": "retrieval_slow", "request_id": request_id, "retrieval_ms": t_retr_ms, "budget_sec": EMBED_SEARCH_BUDGET_SEC})

    # If no candidates -> deterministic not_enough_info
    passages = retr_res.get("passages") or []
    chunk_ids = retr_res.get("chunk_ids") or []
    top_similarity = _top_similarity_from_retrieval(retr_res)
    if not passages:
        jlog({"event": "no_candidates", "request_id": request_id})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": [],
                "resolution": "not_enough_info",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "not_enough_info"}

    # Minimum similarity gate
    if top_similarity < MIN_SIMILARITY:
        jlog({"event": "too_low_similarity", "request_id": request_id, "top_similarity": top_similarity, "min_similarity": MIN_SIMILARITY})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": chunk_ids,
                "resolution": "not_enough_info",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "not_enough_info", "top_similarity": top_similarity}

    # 4) Call generator (grounded)
    gen_ev = {
        "request_id": request_id,
        "language": language,
        "question": query_text,
        "passages": passages,
    }
    t_gen_start = time.time()
    try:
        gen_res = generator.generate(gen_ev)
    except Exception as e:
        jlog({"level": "ERROR", "event": "generator_exception", "request_id": request_id, "detail": str(e)})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": chunk_ids,
                "resolution": "invalid_output",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "invalid_output", "error": "generator_failed"}
    t_gen_ms = int((time.time() - t_gen_start) * 1000)
    jlog({"event": "generator_returned", "request_id": request_id, "gen_ms": t_gen_ms, "gen_decision": gen_res.get("decision")})

    if t_gen_ms / 1000.0 > GEN_BUDGET_SEC:
        jlog({"level": "WARN", "event": "generation_slow", "request_id": request_id, "gen_ms": t_gen_ms, "budget_sec": GEN_BUDGET_SEC})

    # 5) Interpret generator decision and form response
    decision = gen_res.get("decision")
    if decision == "NOT_ENOUGH_INFORMATION":
        jlog({"event": "generator_refuse_no_info", "request_id": request_id})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": chunk_ids,
                "resolution": "not_enough_info",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "not_enough_info"}

    if decision != "ACCEPT":
        jlog({"level": "ERROR", "event": "generator_invalid_output", "request_id": request_id, "decision": decision})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": chunk_ids,
                "resolution": "invalid_output",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "invalid_output"}

    # Ensure answer_lines exist
    answer_lines = gen_res.get("answer_lines") or []
    if not isinstance(answer_lines, list) or not answer_lines:
        jlog({"level": "ERROR", "event": "generator_accepted_but_no_answer", "request_id": request_id})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": chunk_ids,
                "resolution": "invalid_output",
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "invalid_output"}

    # 6) Hydrate citation metadata (best-effort)
    citations = _hydrate_citation_metadata(passages)

    # Build final answer_lines (as returned by generator) and include citations mapping
    response_answer_lines = []
    for ln in answer_lines:
        text = ln.get("text") if isinstance(ln, dict) else str(ln)
        response_answer_lines.append({"text": text})

    # Final response
    res = {
        "request_id": request_id,
        "resolution": "answer",
        "answer_lines": response_answer_lines,
        "citations": citations,
        "confidence": gen_res.get("confidence", "low"),
    }

    # 7) Audit (write record)
    _write_audit(
        {
            "session_id": session_id,
            "request_id": request_id,
            "language": language,
            "channel": channel,
            "query": query_text,
            "used_chunk_ids": chunk_ids,
            "top_similarity": top_similarity,
            "resolution": res["resolution"],
            "generator_decision": decision,
            "timing_ms": int((time.time() - start_time) * 1000),
        }
    )

    jlog({"event": "request_complete", "request_id": request_id, "resolution": res["resolution"], "returned_lines": len(response_answer_lines)})
    return res


# Lambda handler (adapter common entrypoint)
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        return handle(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_unexpected", "detail": str(e)})
        req_id = (event.get("request_id") if isinstance(event, dict) else None) or f"r-{int(time.time()*1000)}"
        # Best-effort audit
        _write_audit(
            {
                "session_id": event.get("session_id") if isinstance(event, dict) else None,
                "request_id": req_id,
                "language": event.get("language") if isinstance(event, dict) else None,
                "channel": event.get("channel") if isinstance(event, dict) else None,
                "used_chunk_ids": [],
                "resolution": "invalid_output",
                "error": str(e),
                "timing_ms": int((time.time() - (context.get_remaining_time_in_millis() / 1000) if context and hasattr(context, 'get_remaining_time_in_millis') else 0) * 1000),
            }
        )
        return {"request_id": req_id, "resolution": "invalid_output", "error": "handler_exception"}

# CLI parity for local tests
if __name__ == "__main__":
    sample = {
        "session_id": "local-sess",
        "request_id": "local-req",
        "language": "en",
        "channel": "web",
        "query": "How to apply for a voter id?",
        "top_k": 3,
        "raw_k": 20,
    }
    print(json.dumps(handle(sample), indent=2, default=str))
