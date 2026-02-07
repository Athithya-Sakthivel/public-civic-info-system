from __future__ import annotations
import json
import logging
import sys
from typing import Any, Dict, Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("channels.voice")

def jlog(msg: str) -> None:
    _log.info(msg)

try:
    from inference_pipeline.core.core import handle as core_handle
except Exception:
    try:
        pkg_root = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        from inference_pipeline.core.core import handle as core_handle
    except Exception as e:
        jlog(f"critical: core import failed: {e}")
        raise

def _extract_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body") or {}
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = {}
    return body

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    body = _extract_body(event)
    transcript = body.get("transcript")
    asr_confidence = body.get("asr_confidence")
    language = (body.get("language") or "en").strip().lower()
    session_id = body.get("session_id")
    request_id = body.get("request_id")
    if not transcript:
        jlog("voice handler: missing transcript")
        return {"statusCode": 400, "body": json.dumps({"error": "missing_transcript"})}
    if asr_confidence is None:
        jlog("voice handler: missing asr_confidence")
        return {"statusCode": 400, "body": json.dumps({"error": "missing_asr_confidence"})}
    core_ev = {"channel": "voice", "language": language, "query": transcript, "session_id": session_id, "request_id": request_id, "asr_confidence": float(asr_confidence)}
    jlog(f"voice request | lang={language} | asr_confidence={asr_confidence} | qlen={len(transcript)}")
    try:
        core_res = core_handle(core_ev)
    except Exception as e:
        jlog(f"voice core.handle exception: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": "core_failure"})}
    out = {}
    if core_res.get("resolution") == "answer":
        lines = core_res.get("answer_lines") or []
        texts = []
        for i, el in enumerate(lines):
            if i >= 2:
                break
            texts.append(el.get("text") if isinstance(el, dict) else str(el))
        tts_text = " ".join(texts)
        out = {"request_id": core_res.get("request_id"), "resolution": "answer", "tts_text": tts_text}
        jlog(f"voice ready | request_id={core_res.get('request_id')} | tts_len={len(tts_text)}")
    else:
        reason = core_res.get("resolution") or "refusal"
        out = {"request_id": core_res.get("request_id"), "resolution": reason, "message": "Not enough information" if reason == "not_enough_info" else "Unable to answer"}
        jlog(f"voice refusal | request_id={core_res.get('request_id')} | reason={reason}")
    return {"statusCode": 200, "body": json.dumps(out, ensure_ascii=False)}
