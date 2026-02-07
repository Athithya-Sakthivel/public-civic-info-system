from __future__ import annotations
import os
import sys
import json
import time
import logging
import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("core.query")

def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "svc": "core.query"}
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))

try:
    import boto3
except Exception as e:
    jlog({"level": "CRITICAL", "event": "boto3_missing", "detail": str(e)})
    raise SystemExit(2)

retriever = None
generator = None
try:
    from core import retriever as retriever, generator as generator  # type: ignore
except Exception:
    try:
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        from core import retriever as retriever, generator as generator  # type: ignore
    except Exception as e:
        jlog({"level": "CRITICAL", "event": "core_import_failed", "detail": str(e)})
        raise SystemExit(3)

def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if v is not None else default

AWS_REGION = _env("AWS_REGION") or _env("AWS_DEFAULT_REGION")
if not AWS_REGION:
    jlog({"level": "CRITICAL", "event": "aws_region_missing", "hint": "Set AWS_REGION"})
    raise SystemExit(10)

MIN_SIMILARITY = float(_env("MIN_SIMILARITY", "0.35"))
ASR_CONF_THRESHOLD = float(_env("ASR_CONF_THRESHOLD", "0.35"))
EMBED_SEARCH_BUDGET_SEC = float(_env("EMBED_SEARCH_BUDGET_SEC", "2.5"))
GEN_BUDGET_SEC = float(_env("GEN_BUDGET_SEC", "4.0"))

AUDIT_S3_BUCKET = _env("AUDIT_S3_BUCKET")
AUDIT_S3_PREFIX = _env("AUDIT_S3_PREFIX", "audits/")

ALLOWED_LANGUAGES = set(["en", "hi", "ta"])
ALLOWED_CHANNELS = set(["web", "sms", "voice"])

INTENT_BLOCKLIST_PATTERNS = {
    "medical": re.compile(r"\b(medic(al|ine|ation)|prescrib|diagnos|symptom|pill|dosage|treatment)\b", re.I),
    "legal": re.compile(r"\b(attorney|sue|lawsuit|contract|custody|divorce|legal advice|crime|sentence)\b", re.I),
}
GUIDANCE_KEYS = {
    "medical": "refusal_medical",
    "legal": "refusal_legal",
    "asr_low_confidence": "refusal_asr_low_confidence",
    "insufficient_evidence": "refusal_insufficient_evidence",
    "invalid_request": "refusal_invalid_request",
}

CITATION_PAT = re.compile(r"\[(\d+)\]\s*$")
DISALLOWED_SUBSTRINGS = ("http://", "https://", "www.", "file://")

jlog({"event": "startup_ok", "region": AWS_REGION, "min_similarity": MIN_SIMILARITY, "asr_threshold": ASR_CONF_THRESHOLD})

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

def _write_audit(record: Dict[str, Any]) -> None:
    if not AUDIT_S3_BUCKET:
        jlog({"event": "audit_skipped", "reason": "no_audit_bucket"})
        return
    client = init_s3_client()
    if client is None:
        jlog({"event": "audit_skipped", "reason": "s3_unavailable"})
        return
    try:
        date_prefix = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        key = f"{AUDIT_S3_PREFIX.rstrip('/')}/{date_prefix}/{record.get('request_id')}.json"
        client.put_object(Bucket=AUDIT_S3_BUCKET, Key=key, Body=json.dumps(record, default=str).encode("utf-8"))
        jlog({"event": "audit_written", "request_id": record.get("request_id"), "s3_key": key})
    except Exception as e:
        jlog({"level": "WARN", "event": "audit_write_failed", "request_id": record.get("request_id"), "detail": str(e)})

def _validate_request_shape(ev: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    req = {}
    req_id = ev.get("request_id") or f"r-{int(time.time() * 1000)}"
    language = (ev.get("language") or "").strip().lower()
    channel = (ev.get("channel") or "").strip().lower()
    query_text = (ev.get("query") or ev.get("question") or "").strip()
    session_id = ev.get("session_id")
    try:
        top_k = int(ev.get("top_k")) if ev.get("top_k") is not None else None
    except Exception:
        top_k = None
    try:
        raw_k = int(ev.get("raw_k")) if ev.get("raw_k") is not None else None
    except Exception:
        raw_k = None

    if not language or language not in ALLOWED_LANGUAGES:
        return None, {"error": "invalid_language", "request_id": req_id}
    if not channel or channel not in ALLOWED_CHANNELS:
        return None, {"error": "invalid_channel", "request_id": req_id}
    if not query_text:
        return None, {"error": "empty_query", "request_id": req_id}
    asr_confidence = None
    if channel == "voice":
        try:
            asr_confidence = float(ev.get("asr_confidence")) if ev.get("asr_confidence") is not None else None
        except Exception:
            asr_confidence = None
        if asr_confidence is None:
            return None, {"error": "missing_asr_confidence", "request_id": req_id}

    req.update(
        {
            "request_id": req_id,
            "session_id": session_id,
            "language": language,
            "channel": channel,
            "query": query_text,
            "top_k": top_k,
            "raw_k": raw_k,
            "asr_confidence": asr_confidence,
            "region": ev.get("region"),
            "filters": ev.get("filters", {}),
        }
    )
    return req, None

def _intent_blocked(query: str) -> Optional[str]:
    for k, pat in INTENT_BLOCKLIST_PATTERNS.items():
        if pat.search(query):
            return GUIDANCE_KEYS.get(k)
    return None

def _enforce_asr(asr_confidence: Optional[float]) -> Optional[str]:
    if asr_confidence is None:
        return None
    if float(asr_confidence) < ASR_CONF_THRESHOLD:
        return "asr_low_confidence"
    return None

def _validate_generator_output_and_extract_lines(gen_res: Dict[str, Any], passages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    decision = gen_res.get("decision")
    if decision == "NOT_ENOUGH_INFORMATION":
        return "NOT_ENOUGH_INFORMATION", []

    raw_lines: List[str] = []
    if isinstance(gen_res.get("answer_lines"), list) and gen_res.get("answer_lines"):
        for el in gen_res.get("answer_lines"):
            if isinstance(el, dict) and isinstance(el.get("text"), str):
                for ln in el.get("text").splitlines():
                    if ln.strip():
                        raw_lines.append(ln.strip())
            elif isinstance(el, str):
                for ln in el.splitlines():
                    if ln.strip():
                        raw_lines.append(ln.strip())
    else:
        found = None
        for k in ("text", "output", "result", "body"):
            v = gen_res.get(k)
            if isinstance(v, str) and v.strip():
                found = v
                break
        if found is None:
            return "INVALID_OUTPUT", []
        for ln in found.splitlines():
            if ln.strip():
                raw_lines.append(ln.strip())

    if not raw_lines:
        return "INVALID_OUTPUT", []

    max_pass = 0
    for p in passages:
        try:
            n = int(p.get("number", 0))
            if n > max_pass:
                max_pass = n
        except Exception:
            continue
    if max_pass < 1:
        return "INVALID_OUTPUT", []

    validated_lines: List[Dict[str, Any]] = []
    for ln in raw_lines:
        m = CITATION_PAT.search(ln)
        if not m:
            jlog({"event": "validation_failed", "reason": "missing_citation", "line": ln})
            return "INVALID_OUTPUT", []
        cited = int(m.group(1))
        if cited < 1 or cited > max_pass:
            jlog({"event": "validation_failed", "reason": "citation_out_of_range", "line": ln, "cited": cited, "max_pass": max_pass})
            return "INVALID_OUTPUT", []
        lower = ln.lower()
        if any(s in lower for s in DISALLOWED_SUBSTRINGS):
            jlog({"event": "validation_failed", "reason": "disallowed_substring", "line": ln})
            return "INVALID_OUTPUT", []
        validated_lines.append({"text": ln})
    return "ACCEPT", validated_lines

def _hydrate_citation_metadata(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

def handle(event: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    validated, error = _validate_request_shape(event)
    if error:
        jlog({"level": "ERROR", "event": "invalid_request_shape", "detail": error, "request_id": error.get("request_id")})
        _write_audit(
            {
                "session_id": event.get("session_id"),
                "request_id": error.get("request_id"),
                "language": event.get("language"),
                "channel": event.get("channel"),
                "used_chunk_ids": [],
                "resolution": "refusal",
                "guidance_key": GUIDANCE_KEYS.get("invalid_request"),
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": error.get("request_id"), "resolution": "refusal", "guidance_key": GUIDANCE_KEYS.get("invalid_request")}

    request_id = validated["request_id"]
    session_id = validated.get("session_id")
    language = validated["language"]
    channel = validated["channel"]
    query_text = validated["query"]
    top_k = validated["top_k"]
    raw_k = validated["raw_k"]
    filters = validated.get("filters", {})

    jlog({"event": "request_start", "request_id": request_id, "session_id": session_id, "language": language, "channel": channel})

    if channel == "voice":
        asr_conf = validated.get("asr_confidence")
        asr_issue = _enforce_asr(asr_conf)
        if asr_issue:
            jlog({"event": "refuse_asr", "request_id": request_id, "asr_confidence": asr_conf})
            _write_audit(
                {
                    "session_id": session_id,
                    "request_id": request_id,
                    "language": language,
                    "channel": channel,
                    "used_chunk_ids": [],
                    "resolution": "refusal",
                    "guidance_key": GUIDANCE_KEYS.get("asr_low_confidence"),
                    "timing_ms": int((time.time() - start_time) * 1000),
                }
            )
            return {"request_id": request_id, "resolution": "refusal", "guidance_key": GUIDANCE_KEYS.get("asr_low_confidence")}

    guid = _intent_blocked(query_text)
    if guid:
        jlog({"event": "intent_blocked", "request_id": request_id, "guidance_key": guid})
        _write_audit(
            {
                "session_id": session_id,
                "request_id": request_id,
                "language": language,
                "channel": channel,
                "used_chunk_ids": [],
                "resolution": "refusal",
                "guidance_key": guid,
                "timing_ms": int((time.time() - start_time) * 1000),
            }
        )
        return {"request_id": request_id, "resolution": "refusal", "guidance_key": guid}

    retr_ev = {"request_id": request_id, "query": query_text, "top_k": top_k, "raw_k": raw_k, "filters": filters}
    t_retr_start = time.time()
    try:
        retr_res = retriever.retrieve(retr_ev)
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

    if t_retr_ms / 1000.0 > EMBED_SEARCH_BUDGET_SEC:
        jlog({"level": "WARN", "event": "retrieval_slow", "request_id": request_id, "retrieval_ms": t_retr_ms, "budget_sec": EMBED_SEARCH_BUDGET_SEC})

    passages = retr_res.get("passages") or []
    chunk_ids = retr_res.get("chunk_ids") or []
    top_similarity = float(retr_res.get("top_similarity") or 0.0)

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

    gen_ev = {"request_id": request_id, "language": language, "question": query_text, "passages": passages}
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

    validation_decision, validated_lines = _validate_generator_output_and_extract_lines(gen_res, passages)
    if validation_decision == "NOT_ENOUGH_INFORMATION":
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

    if validation_decision != "ACCEPT":
        jlog({"level": "ERROR", "event": "generator_invalid_output", "request_id": request_id, "validation_decision": validation_decision})
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

    citations = _hydrate_citation_metadata(passages)

    response_answer_lines = validated_lines
    res = {
        "request_id": request_id,
        "resolution": "answer",
        "answer_lines": response_answer_lines,
        "citations": citations,
        "confidence": gen_res.get("confidence", "high"),
    }

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
            "generator_decision": gen_res.get("decision"),
            "timing_ms": int((time.time() - start_time) * 1000),
        }
    )

    jlog({"event": "request_complete", "request_id": request_id, "resolution": res["resolution"], "returned_lines": len(response_answer_lines)})
    return res

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        return handle(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_unexpected", "detail": str(e)})
        req_id = event.get("request_id") if isinstance(event, dict) else None or f"r-{int(time.time() * 1000)}"
        _write_audit(
            {
                "session_id": event.get("session_id") if isinstance(event, dict) else None,
                "request_id": req_id,
                "language": event.get("language") if isinstance(event, dict) else None,
                "channel": event.get("channel") if isinstance(event, dict) else None,
                "used_chunk_ids": [],
                "resolution": "invalid_output",
                "error": str(e),
                "timing_ms": int((time.time() - (context.get_remaining_time_in_millis() / 1000) if context and hasattr(context, "get_remaining_time_in_millis") else 0) * 1000),
            }
        )
        return {"request_id": req_id, "resolution": "invalid_output", "error": "handler_exception"}

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
