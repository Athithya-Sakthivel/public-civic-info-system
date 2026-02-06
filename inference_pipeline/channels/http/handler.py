#!/usr/bin/env python3
"""
inference_pipeline/channels/http/handler.py

Thin HTTP adapter (API Gateway) -> core.query.handle

Assumptions & invariants:
- Runs inside AWS Lambda (or local dev). If running in Lambda, `context.get_remaining_time_in_millis()` may be used.
- core.query.handle(event: dict) is available (importable from core.query).
- Optional auth: if HTTP_AUTH_TOKEN env var is set, incoming requests MUST send Authorization: "Bearer <token>".
- Input: API Gateway proxy event or a plain dict with keys matching canonical request fields.
- Output: returns a dict compatible with API Gateway proxy integration:
    { "statusCode": int, "headers": {...}, "body": "<json-string>" }
- This adapter performs only normalization & transport mapping. All policy decisions live in core.query.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
from typing import Any, Dict, Optional

# structured logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("channels.http")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "svc": "channels.http"}
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
HTTP_AUTH_TOKEN = os.getenv("HTTP_AUTH_TOKEN")  # optional
DEFAULT_LANG = os.getenv("DEFAULT_LANGUAGE", "en")

# ---- small deterministic retry config ----
MAX_RETRIES = int(os.getenv("ADAPTER_MAX_RETRIES", "2"))
RETRY_BASE_DELAY_SEC = float(os.getenv("ADAPTER_RETRY_BASE_DELAY_SEC", "0.05"))


def _parse_api_gateway_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support API Gateway v1/v2 proxied event (detect body JSON) or direct dict payload.
    Returns canonical request dict or raises ValueError on malformed input.
    """
    # If API Gateway style: body is a JSON string
    body = ev.get("body") if isinstance(ev, dict) else None
    headers = ev.get("headers") or {}
    if body:
        try:
            payload = json.loads(body)
        except Exception:
            # body might be already a dict-like string; treat as raw text
            raise ValueError("invalid_json_body")
    else:
        # fallback: event might already be the payload
        payload = ev

    # Authorization check (optional)
    auth = (headers.get("Authorization") or headers.get("authorization") or "").strip()
    if HTTP_AUTH_TOKEN:
        if not auth.startswith("Bearer "):
            raise ValueError("missing_auth")
        token = auth.split(" ", 1)[1].strip()
        if token != HTTP_AUTH_TOKEN:
            raise ValueError("invalid_auth")

    # Ensure language present or default
    if "language" not in payload or not payload.get("language"):
        payload["language"] = DEFAULT_LANG

    return payload


def _invoke_core(payload: Dict[str, Any], remaining_ms: Optional[int]) -> Dict[str, Any]:
    """
    Call core.query.handle synchronously with a small retry loop for transient adapter-level failures.
    """
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
                # Simple exponential backoff
                time.sleep(RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)))
                # if remaining_ms is very small, bail fast
                if remaining_ms is not None and remaining_ms < 200:
                    break
                continue
            raise last_exc
    raise last_exc


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda entrypoint for HTTP adapter. Returns API Gateway-compatible response.
    """
    start = time.time()
    req_id = None
    try:
        payload = _parse_api_gateway_event(event)
    except ValueError as ve:
        jlog({"level": "ERROR", "event": "bad_request", "detail": str(ve)})
        return {"statusCode": 400, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": str(ve)})}

    req_id = payload.get("request_id")
    remaining_ms = None
    try:
        if context and hasattr(context, "get_remaining_time_in_millis"):
            remaining_ms = context.get_remaining_time_in_millis()

        core_res = _invoke_core(payload, remaining_ms)
        # HTTP returns full core JSON
        body = json.dumps(core_res, default=str)
        jlog({"event": "http_success", "request_id": core_res.get("request_id"), "ms": int((time.time() - start) * 1000)})
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": body}
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_exception", "request_id": req_id, "detail": str(e)})
        return {"statusCode": 500, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": "internal_error"})}


# CLI parity
if __name__ == "__main__":
    sample = {"request_id": "local-http-1", "language": "en", "channel": "web", "query": "How to apply for voter id?"}
    print(json.dumps(handler(sample, None), indent=2))
