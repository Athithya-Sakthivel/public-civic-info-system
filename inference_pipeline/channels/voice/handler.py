#!/usr/bin/env python3
"""
inference_pipeline/channels/voice/handler.py

Thin Voice adapter -> core.query.handle

Assumptions & invariants:
- Adapter accepts telephony provider payloads. Two modes:
  1) PASS_THROUGH (default): provider supplies 'transcript' and 'asr_confidence' in the webhook payload.
  2) TRANSCRIBE_SYNC: adapter will call AWS Transcribe synchronously if 's3_audio_uri' is provided and TRANSCRIBE_MODE=sync.
     - TRANSCRIBE_SYNC is NOT enabled by default to avoid introducing heavy infra in the adapter.
- DEFAULT_LANGUAGE env var used if language is not provided.
- Voice mapping rules:
  * Keep audio responses short; prefer returning core-provided 'tts_url' if present (core may supply signed TTS).
  * If no tts_url, return a small 'speech_text' (first 1-2 sentences).
  * Adapter does not perform ASR gating: it forwards `asr_confidence` and core.enforces ASR thresholds.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
from typing import Any, Dict, Optional, List

# structured logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("channels.voice")


def jlog(obj: Dict[str, Any]) -> None:
    base = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "svc": "channels.voice"}
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
TRANSCRIBE_MODE = os.getenv("TRANSCRIBE_MODE", "pass_through")  # options: pass_through, sync (sync not recommended)
TRANSCRIBE_REGION = os.getenv("TRANSCRIBE_REGION")  # required for sync mode
MAX_SPEECH_SENTENCES = int(os.getenv("MAX_SPEECH_SENTENCES", "2"))

# optional AWS clients only used if transcribe sync is enabled
_transcribe_client = None
if TRANSCRIBE_MODE == "sync":
    try:
        import boto3
        _transcribe_client = boto3.client("transcribe", region_name=TRANSCRIBE_REGION)
    except Exception as e:
        jlog({"level": "CRITICAL", "event": "transcribe_client_failed", "detail": str(e)})
        raise SystemExit(4)


def _parse_voice_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supported inputs:
      - provider supplies transcript & asr_confidence: {'call_id': 'x','transcript': 'text', 'asr_confidence': 0.9, 'language': 'en'}
      - provider supplies s3_audio_uri (and optionally content_type); in TRANSCRIBE_MODE=sync we will transcribe synchronously.
    """
    if not isinstance(ev, dict):
        raise ValueError("invalid_event")

    # s3_audio_uri mode
    s3_uri = ev.get("s3_audio_uri") or ev.get("audio_s3")
    if s3_uri and TRANSCRIBE_MODE == "sync":
        # synchronous transcribe (blocking) - basic implementation; production should use async jobs
        # NOTE: We implement a small, conservative path but recommend pass_through in production.
        try:
            job_name = f"job-{int(time.time() * 1000)}"
            # This is a placeholder; real implementation depends on AWS Transcribe API and audio format
            resp = _transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                LanguageCode=ev.get("language", DEFAULT_LANG),
                Media={"MediaFileUri": s3_uri},
                OutputBucketName=None,  # optional
            )
            # Wait poll (short): rely on environment timeout for safety
            while True:
                st = _transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                status = st.get("TranscriptionJob", {}).get("TranscriptionJobStatus")
                if status in ("COMPLETED", "FAILED"):
                    break
                time.sleep(0.5)
            if status != "COMPLETED":
                raise RuntimeError("transcription_failed")
            transcript_uri = st["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            # fetch the transcript URI (simple GET)
            import urllib.request
            with urllib.request.urlopen(transcript_uri, timeout=5) as resp2:
                body = resp2.read()
                j = json.loads(body)
                transcript_text = j.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
            transcript = transcript_text
            asr_confidence = ev.get("asr_confidence", 0.9)
        except Exception as e:
            jlog({"level": "ERROR", "event": "transcribe_sync_failed", "detail": str(e)})
            raise
    else:
        # pass-through mode expects 'transcript' and optional 'asr_confidence'
        transcript = ev.get("transcript") or ev.get("text") or ev.get("speech_text") or ""
        asr_confidence = ev.get("asr_confidence")
    # Build canonical request
    payload = {
        "session_id": ev.get("call_id") or ev.get("session_id"),
        "request_id": ev.get("request_id"),
        "language": ev.get("language") or DEFAULT_LANG,
        "channel": "voice",
        "query": transcript.strip(),
        "asr_confidence": asr_confidence,
    }
    # allow filters passthrough
    if ev.get("filters"):
        payload["filters"] = ev.get("filters")
    return payload


def _pick_speech_text_from_core(core_res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map core response to voice action:
      - if core_res contains 'tts_url' -> return {'type':'play_tts','tts_url': ...}
      - else return {'type':'speak_text','speech_text': '<short text>'}
    """
    if core_res.get("resolution") != "answer":
        guidance = core_res.get("guidance_key") or core_res.get("reason") or "no_answer"
        return {"type": "speak_text", "speech_text": f"Sorry, I cannot answer that: {guidance}."}

    # prefer tts_url
    if core_res.get("tts_url"):
        return {"type": "play_tts", "tts_url": core_res.get("tts_url")}

    # fallback: use first MAX_SPEECH_SENTENCES from answer_lines (strip citations like [1])
    lines = core_res.get("answer_lines") or []
    if not lines:
        return {"type": "speak_text", "speech_text": "Sorry, I cannot provide an answer right now."}
    # join first MAX_SPEECH_SENTENCES lines (they are typically 1 per line)
    texts: List[str] = []
    for i, ln in enumerate(lines):
        if i >= MAX_SPEECH_SENTENCES:
            break
        txt = ln.get("text") if isinstance(ln, dict) else str(ln)
        # strip citation tokens like " [1]"
        txt = re.sub(r"\s*\[\d+\]\s*$", "", txt)
        texts.append(txt.strip())
    speech = " ".join(texts).strip()
    if not speech:
        speech = "Sorry, I cannot provide an answer right now."
    return {"type": "speak_text", "speech_text": speech}


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
    Voice Lambda handler. Returns a dict instructing telephony provider what to do next:
      - {'type':'play_tts','tts_url': '...'}
      - {'type':'speak_text','speech_text': '...'}
    """
    start = time.time()
    try:
        payload = _parse_voice_event(event)
    except Exception as e:
        jlog({"level": "ERROR", "event": "voice_parse_failed", "detail": str(e)})
        return {"type": "speak_text", "speech_text": "Sorry, I could not process your audio."}

    if not payload.get("query"):
        jlog({"level": "ERROR", "event": "empty_transcript", "session_id": payload.get("session_id")})
        return {"type": "speak_text", "speech_text": "I didn't hear anything. Please try again."}

    try:
        core_res = _invoke_core(payload)
    except Exception as e:
        jlog({"level": "ERROR", "event": "core_invoke_failed_final", "detail": str(e)})
        return {"type": "speak_text", "speech_text": "Sorry, an internal error occurred."}

    voice_action = _pick_speech_text_from_core(core_res)
    jlog({"event": "voice_response", "request_id": core_res.get("request_id"), "action": voice_action.get("type"), "ms": int((time.time() - start) * 1000)})
    return voice_action


# CLI parity
if __name__ == "__main__":
    sample = {"call_id": "local-call-1", "transcript": "How do I apply for voter id?", "asr_confidence": 0.9, "language": "en"}
    print(json.dumps(handler(sample, None), indent=2))
