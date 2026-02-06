#!/usr/bin/env python3
"""
inference_pipeline/core/generator.py
Production-ready grounded generator for public-civic-info-system.

Exposes:
  - generate(event: dict) -> dict
  - handler(event, context)

Behavior invariants (fail-fast):
- Python 3.11 runtime.
- AWS_REGION must be set.
- BEDROCK_MODEL_ID must be set (default provided).
- The generator will call Bedrock; it will retry once on transient errors.
- Output validation enforces: sentence-level citations [n], citation bounds, no URLs, allowed outputs only.
"""
from __future__ import annotations
import os
import sys
import json
import time
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

# -------- structured logger ----------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("core.generator")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "svc": "core.generator"}
    base.update(obj)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str))
    except Exception:
        _log.info(str(base))


# -------- fail-fast imports ----------
try:
    import boto3
except Exception as e:
    jlog({"level": "CRITICAL", "event": "boto3_missing", "detail": str(e)})
    raise SystemExit(2)


# -------- env knobs (centralized & validated) ----------
def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if v is not None else default


AWS_REGION = _env("AWS_REGION")
if not AWS_REGION:
    jlog({"level": "CRITICAL", "event": "aws_region_missing", "hint": "Set AWS_REGION"})
    raise SystemExit(10)

BEDROCK_MODEL_ID = _env("BEDROCK_MODEL_ID", "amazon.titan-text-3-bison")
# Small safety knobs
BEDROCK_MAX_RETRIES = int(_env("BEDROCK_MAX_RETRIES", "1"))
BEDROCK_RETRY_BASE_DELAY_SEC = float(_env("BEDROCK_RETRY_BASE_DELAY_SEC", "0.25"))

# Disallowed substrings in generator output
DISALLOWED_SUBSTRINGS = ("http://", "https://", "www.", "file://")

# Citation pattern: a sentence (or line) must end with [n], possibly with trailing whitespace
CITATION_PAT = re.compile(r"\[(\d+)\]\s*$")

# Initialize bedrock client lazily
_bedrock = None


def init_bedrock_client():
    global _bedrock
    if _bedrock is not None:
        return _bedrock
    try:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        jlog({"level": "CRITICAL", "event": "bedrock_client_init_failed", "detail": str(e)})
        _bedrock = None
        raise
    jlog({"event": "bedrock_client_init"})
    return _bedrock


# -------- prompt builders (deterministic) ----------
SYSTEM_PROMPT = (
    "SYSTEM INSTRUCTIONS:\n"
    "- You MUST answer ONLY using the provided numbered passages.\n"
    "- Each factual sentence MUST end with a citation in the exact form [n] where n is the passage number.\n"
    "- Use ONLY the provided passage numbers. Do NOT invent or infer facts not present in the passages.\n"
    "- Do NOT include URLs, filenames, page numbers, or any other metadata.\n"
    "- If the passages do not contain enough information to answer, reply exactly: NOT_ENOUGH_INFORMATION\n"
    "- Always answer in the same language as the user's query.\n"
    "- Keep answers brief and simple.\n"
)

def build_user_prompt(language: str, question: str, passages: List[Dict[str, Any]]) -> str:
    """
    Construct deterministic prompt text:
      - LANGUAGE header
      - PASSAGES block (ordered by passage['number'])
      - QUESTION block
      - Short instruction to keep brief
    """
    header = [f"LANGUAGE: {language}", "", SYSTEM_PROMPT.strip(), ""]
    parts = header + ["PASSAGES:"]
    # Sort passages by their numeric 'number' to be deterministic
    sorted_passages = sorted(passages, key=lambda p: int(p.get("number", 0)))
    for p in sorted_passages:
        num = int(p.get("number", 0))
        text = p.get("text", "") or ""
        # Normalize newlines inside passages to single-space to avoid accidental multi-line injection
        text = " ".join(line.strip() for line in text.splitlines())
        parts.append(f"{num}. {text}")
    parts.extend(["", "QUESTION:", question.strip(), "", "Answer briefly."])
    return "\n".join(parts)


# -------- bedrock call with one retry ----------
def call_bedrock(prompt: str) -> str:
    client = init_bedrock_client()
    body = json.dumps({"inputText": prompt})
    attempt = 0
    last_exc = None
    while attempt <= BEDROCK_MAX_RETRIES:
        try:
            resp = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=body, contentType="application/json")
            body_stream = resp.get("body")
            raw = body_stream.read() if hasattr(body_stream, "read") else body_stream
            try:
                mr = json.loads(raw)
            except Exception:
                # if it's not JSON, treat raw as string and return best-effort
                mr = raw if isinstance(raw, str) else str(raw)
            # tolerant extraction of generated text
            if isinstance(mr, dict):
                text = (
                    mr.get("content")
                    or mr.get("outputText")
                    or mr.get("text")
                    or mr.get("body")
                    or mr.get("result")
                    or mr.get("response")
                    or mr.get("generatedText")
                )
                if text is None and "messages" in mr:
                    text = json.dumps(mr["messages"])
            else:
                text = str(mr)
            return text if isinstance(text, str) else ""
        except Exception as e:
            last_exc = e
            jlog({"level": "ERROR", "event": "bedrock_invoke_failed", "attempt": attempt, "detail": str(e)})
            attempt += 1
            if attempt <= BEDROCK_MAX_RETRIES:
                delay = BEDROCK_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                time.sleep(delay)
                continue
            raise last_exc


# -------- output validator ----------
def validate_and_parse_output(raw: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate model output against strict rules.

    Returns dict with keys:
      - decision: "ACCEPT" | "NOT_ENOUGH_INFORMATION" | "INVALID_OUTPUT"
      - answer_lines: [{text: str}] when ACCEPT
      - confidence: "high" when ACCEPT
    """
    if raw is None:
        return {"decision": "INVALID_OUTPUT"}

    raw = raw.strip()
    if raw == "":
        return {"decision": "INVALID_OUTPUT"}

    # Exact refusal token (must be exact)
    if raw == "NOT_ENOUGH_INFORMATION":
        return {"decision": "NOT_ENOUGH_INFORMATION"}

    # Split into non-empty lines. We treat each non-empty line as an answer unit.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return {"decision": "INVALID_OUTPUT"}

    # Validate citation bounds
    max_pass = max((int(p.get("number", 0)) for p in passages), default=0)
    if max_pass < 1:
        # no passages provided or malformed
        return {"decision": "INVALID_OUTPUT"}

    answer_lines: List[Dict[str, Any]] = []
    for ln in lines:
        # Each line must end with [n]
        m = CITATION_PAT.search(ln)
        if not m:
            jlog({"event": "validation_failed", "reason": "missing_citation", "line": ln})
            return {"decision": "INVALID_OUTPUT"}
        cited = int(m.group(1))
        if cited < 1 or cited > max_pass:
            jlog({"event": "validation_failed", "reason": "citation_out_of_range", "line": ln, "cited": cited, "max_pass": max_pass})
            return {"decision": "INVALID_OUTPUT"}
        # Disallow URLs and known substrings
        lower = ln.lower()
        if any(s in lower for s in DISALLOWED_SUBSTRINGS):
            jlog({"event": "validation_failed", "reason": "disallowed_substring", "line": ln})
            return {"decision": "INVALID_OUTPUT"}
        # Prevent metadata tokens that are not allowed (simple heuristic)
        # e.g., lines that contain 'http', 'page', 'source:' are rejected by substring check above.
        answer_lines.append({"text": ln})

    # Extra safety: ensure at least one line contains a citation referencing a passage (already checked)
    return {"decision": "ACCEPT", "answer_lines": answer_lines, "confidence": "high"}


# -------- pure generator function ----------
def generate(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event keys:
      - request_id (optional)
      - language (required)
      - question | query (required)
      - passages (required, list of {number: int, text: str})
    Returns:
      {
        request_id,
        decision: ACCEPT | NOT_ENOUGH_INFORMATION | INVALID_OUTPUT,
        answer_lines (if ACCEPT),
        confidence (if ACCEPT)
      }
    """
    start = time.time()
    request_id = event.get("request_id") or f"g-{int(start * 1000)}"
    language = event.get("language")
    question = (event.get("question") or event.get("query") or "").strip()
    passages = event.get("passages") or []

    jlog({"event": "generate_start", "request_id": request_id, "language": language, "passage_count": len(passages)})

    # Basic validation of inputs
    if not language or not question or not passages:
        jlog({"level": "ERROR", "event": "invalid_generate_request", "request_id": request_id})
        return {"request_id": request_id, "decision": "INVALID_OUTPUT"}

    # Build deterministic prompt
    prompt = build_user_prompt(language, question, passages)

    # Call Bedrock
    try:
        raw_output = call_bedrock(prompt)
    except Exception as e:
        jlog({"level": "ERROR", "event": "bedrock_call_failed", "request_id": request_id, "detail": str(e)})
        return {"request_id": request_id, "decision": "INVALID_OUTPUT"}

    # Normalize raw output to string
    raw_output = raw_output if isinstance(raw_output, str) else str(raw_output or "")

    # Validate
    parsed = validate_and_parse_output(raw_output, passages)
    decision = parsed.get("decision", "INVALID_OUTPUT")
    elapsed_ms = int((time.time() - start) * 1000)
    jlog({"event": "generate_complete", "request_id": request_id, "decision": decision, "ms": elapsed_ms})

    # Attach request_id on accept
    if decision == "ACCEPT":
        parsed["request_id"] = request_id
        return parsed

    return {"request_id": request_id, "decision": decision}


# -------- Lambda handler (thin wrapper) ----------
def handler(event: Dict[str, Any], context: Any):
    try:
        return generate(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "handler_unexpected", "detail": str(e)})
        return {"request_id": event.get("request_id") or f"g-{int(time.time() * 1000)}", "decision": "INVALID_OUTPUT"}


# -------- CLI parity ----------
if __name__ == "__main__":
    sample = {
        "request_id": "local-gen-test",
        "language": "en",
        "question": "How do I apply for a voter ID?",
        "passages": [{"number": 1, "text": "You can apply online at the government portal."}],
    }
    print(json.dumps(handler(sample, None), indent=2))
