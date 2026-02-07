#!/usr/bin/env python3
"""
inference_pipeline/core/generator.py

Final converged generator:
- Pinned to qwen.qwen3-next-80b-a3b (startup fails if overridden).
- Uses Bedrock Converse API with correct `inferenceConfig`. Falls back to invoke_model on ParamValidationError.
- No validators, no sanitizers: model output returned verbatim.
- Minimal, production-oriented: retry/backoff, logging, and fail-fast env checks.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
from typing import Any, Dict, List, Optional

# -------- structured logger --------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("core.generator")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "svc": "core.generator"}
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))


# -------- imports --------
try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError, ParamValidationError
except Exception as e:
    jlog({"level": "ERROR", "event": "boto3_missing", "detail": str(e)})
    boto3 = None  # type: ignore
    botocore = None
    ClientError = Exception
    ParamValidationError = Exception

# -------- env knobs & invariants --------
def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(k) if os.getenv(k) is not None else default

# pinned model id (immutable for this runtime)
REQUIRED_BEDROCK_MODEL_ID = "qwen.qwen3-next-80b-a3b"

AWS_REGION = _env("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
BEDROCK_INFERENCE_PROFILE_ID = _env("BEDROCK_INFERENCE_PROFILE_ID")
_env_model = _env("BEDROCK_MODEL_ID")
if _env_model and _env_model != REQUIRED_BEDROCK_MODEL_ID:
    jlog({"level": "ERROR", "event": "invalid_model_override", "provided": _env_model, "required": REQUIRED_BEDROCK_MODEL_ID})
    raise RuntimeError(f"BEDROCK_MODEL_ID must be '{REQUIRED_BEDROCK_MODEL_ID}' for this deployment.")
BEDROCK_MODEL_ID = REQUIRED_BEDROCK_MODEL_ID

# conservative inference defaults (override via env if needed)
DEFAULT_TEMPERATURE = float(_env("BEDROCK_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(_env("BEDROCK_TOP_P", "0.95"))
DEFAULT_MAX_TOKENS = int(_env("BEDROCK_MAX_TOKENS", "128"))

# retry/backoff knobs
BEDROCK_MAX_RETRIES = int(_env("BEDROCK_MAX_RETRIES", "1"))
BEDROCK_RETRY_BASE_DELAY_SEC = float(_env("BEDROCK_RETRY_BASE_DELAY_SEC", "0.25"))

# fail-fast checks
if not AWS_REGION:
    jlog({"level": "ERROR", "event": "aws_region_missing", "hint": "Set AWS_REGION or AWS_DEFAULT_REGION."})
    raise RuntimeError("aws_region_missing")

jlog({
    "event": "startup_ok",
    "bedrock_model": BEDROCK_MODEL_ID,
    "bedrock_profile": BEDROCK_INFERENCE_PROFILE_ID,
    "region": AWS_REGION,
    "boto3_version": getattr(boto3, "__version__", None),
    "botocore_version": getattr(botocore, "__version__", None)
})

# -------- Bedrock client init --------
_bedrock = None

def init_bedrock_client():
    global _bedrock
    if _bedrock is not None:
        return _bedrock
    if boto3 is None:
        raise RuntimeError("boto3_unavailable")
    try:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        jlog({"level": "ERROR", "event": "bedrock_client_init_failed", "detail": str(e)})
        raise
    jlog({"event": "bedrock_client_init"})
    return _bedrock

# -------- prompt builder --------
def _system_prompt() -> str:
    return (
        "You are a helpful government office staff member. "
        "Use ONLY the PASSAGES below. Do not invent facts. "
        "If the passages explain what to do, answer clearly and directly. "
        "If the passages do not explain how to do it, say the information is not available here."
    )

def _build_prompt_text(language: str, question: str, passages: List[Dict[str, Any]]) -> str:
    parts = [f"LANGUAGE: {language}", "", "PASSAGES:"]
    for p in sorted(passages, key=lambda x: int(x.get("number", 0))):
        parts.append(f"{int(p.get('number'))}. {(p.get('text') or '').strip()}")
    parts.extend(["", "QUESTION:", question.strip()])
    return "\n".join(parts)

# -------- Bedrock call (Converse preferred) --------
def call_bedrock_once(prompt_text: str, system_override: Optional[str] = None) -> str:
    client = init_bedrock_client()
    identifier = BEDROCK_INFERENCE_PROFILE_ID or BEDROCK_MODEL_ID

    inference_config = {
        "maxTokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "topP": DEFAULT_TOP_P
    }

    system = [{"text": system_override or _system_prompt()}]
    messages = [{"role": "user", "content": [{"text": prompt_text}]}]

    # Try Converse first with correct parameter name `inferenceConfig`
    try:
        resp = client.converse(
            modelId=identifier,
            messages=messages,
            system=system,
            inferenceConfig=inference_config
        )
        out = resp.get("output", {}) or {}
        message = out.get("message") or {}
        content = message.get("content") or []
        if isinstance(content, list) and content:
            for c in content:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    return c.get("text").strip()
        return json.dumps(resp)
    except ParamValidationError as pve:
        # SDK/service shape mismatch — log and fall back to invoke_model
        jlog({"level": "ERROR", "event": "converse_param_validation", "detail": str(pve)})
    except ClientError:
        # Let caller handle client errors through retry logic
        raise

    # Fallback to invoke_model (legacy)
    body = json.dumps({
        "inputText": prompt_text,
        "inferenceParameters": {
            "temperature": DEFAULT_TEMPERATURE,
            "topP": DEFAULT_TOP_P,
            "maxTokens": DEFAULT_MAX_TOKENS
        }
    }).encode("utf-8")
    resp = init_bedrock_client().invoke_model(modelId=identifier, body=body, contentType="application/json")
    body_stream = resp.get("body")
    raw = body_stream.read() if hasattr(body_stream, "read") else body_stream
    try:
        mr = json.loads(raw)
    except Exception:
        return str(raw)
    for k in ("outputText", "generatedText", "text"):
        v = mr.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(mr)

# -------- public generate (no validators) --------
def generate(event: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    request_id = event.get("request_id") or f"g-{int(start * 1000)}"
    language = event.get("language") or "en"
    question = (event.get("question") or event.get("query") or "").strip()
    passages = event.get("passages") or []

    jlog({"event": "generate_start", "request_id": request_id, "language": language, "passage_count": len(passages)})

    if not question:
        jlog({"level": "ERROR", "event": "invalid_generate_request", "request_id": request_id, "reason": "empty_question"})
        return {"request_id": request_id, "decision": "INVALID_OUTPUT"}

    prompt_text = _build_prompt_text(language, question, passages)
    raw_output: Optional[str] = None

    attempt = 0
    last_exc = None
    while attempt <= BEDROCK_MAX_RETRIES:
        try:
            system_override = None if attempt == 0 else _system_prompt()
            raw_output = call_bedrock_once(prompt_text, system_override=system_override)
            jlog({"event": "bedrock_success", "request_id": request_id, "attempt": attempt})
            break
        except Exception as e:
            last_exc = e
            jlog({"level": "ERROR", "event": "bedrock_invoke_failed", "request_id": request_id, "attempt": attempt, "detail": str(e)})
            attempt += 1
            if attempt <= BEDROCK_MAX_RETRIES:
                time.sleep(BEDROCK_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)))
                continue
            raw_output = None

    if raw_output is None:
        jlog({"level": "ERROR", "event": "no_bedrock_output", "request_id": request_id, "last_error": str(last_exc)})
        return {"request_id": request_id, "decision": "INVALID_OUTPUT"}

    # Return model output verbatim
    return {
        "request_id": request_id,
        "decision": "ACCEPT",
        "answer_lines": [{"text": raw_output}],
        "response_type": "HELP"
    }

def handler(event: Dict[str, Any], context: Any):
    try:
        return generate(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_unexpected", "detail": str(e)})
        return {"request_id": event.get("request_id") or f"g-{int(time.time() * 1000)}", "decision": "INVALID_OUTPUT"}

if __name__ == "__main__":
    sample = {
        "request_id": "local-gen-test",
        "language": "ta",
        "question": "வாக்காளர் அடையாள அட்டை எப்படி பெறுவது?",
        "passages": [
            {
                "number": 1,
                "text": "Apply online at the official portal. You must submit address and identity proof. Use the portal's application form."
            }
        ],
    }
    # print with ensure_ascii=False so Unicode (Tamil) is readable in terminal
    print(json.dumps(handler(sample, None), indent=2, ensure_ascii=False))
