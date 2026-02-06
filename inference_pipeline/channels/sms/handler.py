#!/usr/bin/env python3
"""
inference_pipeline/channels/sms/handler.py

Thin SMS adapter -> core.query.handle

Assumptions & invariants:
- Receives provider webhook payloads (Twilio-style form POST dict, SNS JSON, or a simple JSON with 'from' and 'message').
- This adapter does normalization only: extract `session_id`, `from`, `message` -> canonical core request.
- DEFAULT_LANGUAGE env var is used if language is not provided in message metadata.
- SMS mapping rules:
  * For core "answer" resolution, SMS sends first answer line only (short).
  * If AUDIT_SHORTLINK_BASE is configured and citations exist, append " See source: <shortlink>".
  * Avoid returning citation tokens like [n]; the SMS adapter strips them.
- Adapter enforces SMS_MAX_LENGTH (provider supports concatenation), default 1600.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import re
from typing import Any, Dict, Optional

# structured logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("channels.sms")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "svc": "channels.sms"}
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))


# ---- fail-fast imports ----
try:
    from core import query  # type: ignore
except Exception as e:
    jlog({"level": "CRITICAL", "event": "core_import_failed", "detail": str(e)})
    raise SystemExit(3)

# ---- envs ----
DEFAULT_LANG = os.getenv("DEFAULT_LANGUAGE", "en")
SMS_MAX_LENGTH = int(os.getenv("SMS_MAX_LENGTH", "1600"))
AUDIT_SHORTLINK_BASE = os.getenv("AUDIT_SHORTLINK_BASE")  # optional shortlink base; e.g. https://short.example/

# minimal provider-agnostic parser: variant shapes
def _parse_sms_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support:
      - Twilio: {'From': '+1234', 'Body': 'text'}
      - SNS: {'Message': '{"originationNumber":"+1234","messageBody":"text"}'}
      - Direct JSON: {'from': '+1234', 'message': 'text', 'language': 'en'}
    Returns canonical payload for core.
    """
    # SNS wrapper
    if "Message" in ev and isinstance(ev["Message"], str):
        try:
            inner = json.loads(ev["Message"])
            ev = inner
        except Exception:
            # leave as is
            pass

    # Twilio style
    from_num = ev.get("From") or ev.get("from") or ev.get("originationNumber") or ev.get("msisdn")
    body = ev.get("Body") or ev.get("body") or ev.get("message") or ev.get("messageBody") or ""
    # session_id best effort: use from number
    session_id = from_num
    payload = {
        "session_id": session_id,
        "language": ev.get("language") or DEFAULT_LANG,
        "channel": "sms",
        "query": body.strip(),
    }
    # allow callers to pass request_id
    if ev.get("request_id"):
        payload["request_id"] = ev.get("request_id")
    # allow filters passthrough
    if ev.get("filters"):
        payload["filters"] = ev.get("filters")
    return payload


def _sms_format_text_for_send(text: str) -> str:
    # strip citation tokens like " [1]" at end of sentences
    out = re.sub(r"\s*\[\d+\]", "", text)
    out = out.strip()
    if len(out) > SMS_MAX_LENGTH:
        out = out[: SMS_MAX_LENGTH - 3].rstrip() + "..."
    return out


# deterministic retry config
MAX_RETRIES = int(os.getenv("ADAPTER_MAX_RETRIES", "2"))
RETRY_BASE_DELAY_SEC = float(os.getenv("ADAPTER_RETRY_BASE_DELAY_SEC", "0.05"))


def _invoke_core(payload: Dict[str, Any]) -> Dict[str, Any]:
    attempt = 0
    last_exc = None
    while attempt <= MAX_RETRIES:
        try:
            return query.handle(payload)
        except Exception as e:
            last_exc = e
            jlog({"level": "ERROR", "event": "core_invoke_failed", "attempt": attempt, "detail": str(e)})
            attempt += 1
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)))
                continue
            raise last_exc
    raise last_exc


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Entrypoint for SMS adapter. Returns a dict representing the SMS message to send:
      {
        "to": "<number>",
        "body": "<text>"
      }
    The caller (provider integration) is responsible for sending the SMS using provider SDK.
    """
    start = time.time()
    try:
        payload = _parse_sms_event(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "sms_parse_failed", "detail": str(e)})
        return {"status": "error", "reason": "parse_failed"}

    jlog({"event": "sms_request", "session_id": payload.get("session_id"), "request_id": payload.get("request_id")})
    if not payload.get("query"):
        jlog({"level": "ERROR", "event": "empty_message", "session_id": payload.get("session_id")})
        return {"status": "error", "reason": "empty_message"}

    try:
        core_res = _invoke_core(payload)
    except Exception as e:
        jlog({"level": "ERROR", "event": "core_invoke_failed_final", "detail": str(e)})
        return {"status": "error", "reason": "core_error"}

    req_id = core_res.get("request_id")
    resolution = core_res.get("resolution")

    if resolution != "answer":
        # Send a short refusal/notice depending on guidance_key
        guidance = core_res.get("guidance_key") or core_res.get("reason") or "no_answer"
        body = {"to": payload.get("session_id"), "body": f"We cannot answer that request: {guidance}."}
        jlog({"event": "sms_refusal", "request_id": req_id, "session_id": payload.get("session_id"), "guidance": guidance})
        return body

    # Answer: use first answer line only, strip citations
    answer_lines = core_res.get("answer_lines") or []
    first = answer_lines[0].get("text") if answer_lines and isinstance(answer_lines[0], dict) else ""
    sms_text = _sms_format_text_for_send(first or "")
    # Optionally append shortlink if available
    citations = core_res.get("citations") or []
    if AUDIT_SHORTLINK_BASE and citations:
        # deterministic pick first citation's chunk_id
        first_chunk = citations[0].get("chunk_id")
        if first_chunk:
            shortlink = AUDIT_SHORTLINK_BASE.rstrip("/") + "/" + str(first_chunk)
            sms_text = sms_text + f" See source: {shortlink}"

    jlog({"event": "sms_send", "request_id": req_id, "session_id": payload.get("session_id"), "ms": int((time.time() - start) * 1000)})
    return {"to": payload.get("session_id"), "body": sms_text}


# CLI parity
if __name__ == "__main__":
    sample = {"From": "+911234567890", "Body": "How can I apply for voter id?"}
    print(json.dumps(handler(sample, None), indent=2))
