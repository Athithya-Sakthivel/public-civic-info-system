#!/usr/bin/env python3
"""
e2e_local_run.py — single-file end-to-end local runner for public-civic-info-system.

Purpose
- Provide a single-process, deterministic e2e flow: query -> embed -> retrieve -> generate -> format -> audit
- Minimal dependencies (stdlib only)
- Two operation modes:
    - mock (default): uses deterministic pseudo-embeddings and a mock generator that returns citation-marked answers
    - bedrock (optional): will attempt to call Amazon Bedrock if boto3 is available and env vars set

Usage
- python3 e2e_local_run.py --mode=mock --query "How to apply for voter id?"
- See bottom for self-tests and example runs.

Design notes
- Keeps the same high-level contracts as the refactored system: retriever returns passages[], generator consumes passages[] and returns lines with citations [n].
- Core validation and citation checks are run in `pipeline_run`.
- Audit logs are written to ./audits/<date>/<request_id>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

# ---------- structured logging ----------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
_log = logging.getLogger("e2e")

def jlog(d: Dict[str, Any]) -> None:
    base = {"ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "svc": "e2e"}
    base.update(d)
    try:
        _log.info(json.dumps(base, sort_keys=True, default=str, ensure_ascii=False))
    except Exception:
        _log.info(str(base))

# ---------- configuration ----------
EMBED_DIM = int(os.getenv("E2E_EMBED_DIM", "128"))  # small for local runs
RAW_K = int(os.getenv("E2E_RAW_K", "50"))
FINAL_K = int(os.getenv("E2E_FINAL_K", "5"))
MIN_SIMILARITY = float(os.getenv("E2E_MIN_SIM", "0.2"))  # permissive for mock data
AUDIT_DIR = os.getenv("E2E_AUDIT_DIR", "./audits")
BEDROCK_MODE = os.getenv("E2E_BEDROCK_MODE", "false").lower() in ("1","true","yes")

# Generator citation and disallowed substrings
CITATION_PAT = re.compile(r"\[(\d+)\]\s*$")
DISALLOWED_SUBSTRINGS = ("http://","https://","file://","www.")

# ---------- deterministic pseudo-embedding (stable across runs) ----------
def deterministic_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """
    Produce a deterministic pseudo-embedding from text.
    Mechanism: SHA256(text) -> use chunks of the hex digest to seed float values, then L2-normalize.
    This is deterministic and fast; suitable for local E2E testing.
    """
    if not text:
        return [0.0] * dim
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vals: List[float] = []
    # Use repeated hashing to get enough bytes if needed
    acc = h
    while len(acc) < dim * 8:
        acc += hashlib.sha256(acc.encode("utf-8")).hexdigest()
    # Turn each 8-hex chunk into a float in [-1,1]
    for i in range(dim):
        hex_chunk = acc[i*8:(i+1)*8]
        v = int(hex_chunk, 16)
        # map to [-1, 1]
        vals.append(((v / 0xFFFFFFFF) * 2.0) - 1.0)
    # normalize
    norm = math.sqrt(sum(x*x for x in vals) or 1.0)
    return [x / norm for x in vals]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    # assume same dims
    dot = sum(x*y for x,y in zip(a,b))
    # both vectors should be normalized by construction; clamp
    return max(min(dot, 1.0), -1.0)

# ---------- simple chunking and indexing ----------
def chunk_text_simple(doc_text: str, max_chunk_chars: int = 800) -> List[str]:
    """
    Very simple chunker: split on sentences; aggregate until max_chunk_chars is reached.
    """
    if not doc_text:
        return []
    # naive sentence split by punctuation
    parts = re.split(r"(?<=[\.\?\!])\s+", doc_text.strip())
    chunks = []
    cur = []
    cur_len = 0
    for p in parts:
        l = len(p)
        if cur_len + l > max_chunk_chars and cur:
            chunks.append(" ".join(cur).strip())
            cur = [p]
            cur_len = l
        else:
            cur.append(p)
            cur_len += l
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]

class InMemoryIndex:
    """
    Build a simple in-memory vector index with metadata.
    """
    def __init__(self, embed_dim: int = EMBED_DIM):
        self.embed_dim = embed_dim
        self.rows: List[Dict[str, Any]] = []  # each row: {chunk_id, doc_id, text, meta, emb}
        self._next_chunk = 1

    def add_document(self, doc_id: str, text: str, meta: Optional[Dict[str,Any]] = None) -> List[str]:
        """Chunk document and add chunks to the index. Returns list of added chunk_ids."""
        chunk_ids = []
        chunks = chunk_text_simple(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_c{i:04d}"
            emb = deterministic_embedding(chunk, self.embed_dim)
            self.rows.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "meta": meta or {},
                "embedding": emb,
            })
            chunk_ids.append(chunk_id)
            self._next_chunk += 1
        return chunk_ids

    def search(self, query_emb: List[float], raw_k: int = RAW_K, filters: Optional[Dict[str,str]] = None) -> List[Dict[str,Any]]:
        """
        Filter-first search (filters are applied to meta key equality), then sorted by cosine similarity desc.
        """
        # apply filters if any
        rows = self.rows
        if filters:
            def match(r):
                for k,v in filters.items():
                    if str(r.get("meta",{}).get(k)) != str(v):
                        return False
                return True
            rows = [r for r in rows if match(r)]
        # compute similarity
        scored = []
        for r in rows:
            sim = cosine_similarity(query_emb, r["embedding"])
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [dict(r, similarity=sim) for sim, r in scored[:raw_k]]
        return top

# ---------- dedupe & re-rank ----------
def normalize_text_key(s: str) -> str:
    s2 = re.sub(r"\s+", " ", s or "").strip().lower()
    s2 = re.sub(r"[^\w\s]", "", s2)
    return hashlib.sha256(s2.encode("utf-8")).hexdigest()

def dedupe_keep_nearest(candidates: List[Dict[str,Any]], max_keep: int) -> List[Dict[str,Any]]:
    seen = set()
    out = []
    for c in candidates:
        key = normalize_text_key(c.get("text",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_keep:
            break
    return out

def trust_weight(meta: Dict[str,Any]) -> float:
    t = (meta or {}).get("trust_level") or (meta or {}).get("trust") or ""
    if isinstance(t, str):
        m = {"gov": 1.0, "agency": 0.9, "ngo": 0.8, "news": 0.6}
        return float(m.get(t.lower(), 1.0))
    return 1.0

def rerank_and_select(candidates: List[Dict[str,Any]], final_k: int) -> List[Dict[str,Any]]:
    scored = []
    for c in candidates:
        sim = float(c.get("similarity", 0.0))
        tw = trust_weight(c.get("meta", {}))
        final = sim * tw
        c2 = dict(c)
        c2["final_score"] = final
        scored.append(c2)
    scored.sort(key=lambda x: (-x["final_score"], -x.get("similarity",0.0), x.get("chunk_id","")))
    return scored[:final_k]

# ---------- mock generator (creates citation-marked lines) ----------
def mock_generator(question: str, passages: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Deterministic mock: compose one-sentence answers using phrases from top passages, end each sentence with [n].
    If passages do not seem to contain the query (very naive check), return NOT_ENOUGH_INFORMATION.
    """
    # naive check: does any passage contain one or more keywords from the question?
    q_words = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 3)
    match_scores = []
    for p in passages:
        txt = (p.get("text") or "").lower()
        common = q_words.intersection(set(re.findall(r"\w+", txt)))
        match_scores.append((len(common), p))
    # if no overlap, refuse
    best_common = max((s for s,_ in match_scores), default=0)
    if best_common == 0:
        return {"decision": "NOT_ENOUGH_INFORMATION"}
    # produce up to 2 lines using top 2 passages
    match_scores.sort(key=lambda x: -x[0])
    lines = []
    used = 0
    for _, p in match_scores[:3]:
        snippet = p.get("text","").split(".")[0].strip()
        if not snippet:
            continue
        # make short line
        num = p.get("number") or 1
        line = f"{snippet}. [{num}]"
        # ensure it's not too long
        if len(line) > 300:
            line = line[:297].rstrip() + "..."
        lines.append(line)
        used += 1
        if used >= 2:
            break
    if not lines:
        return {"decision": "NOT_ENOUGH_INFORMATION"}
    return {"decision": "ACCEPT", "answer_lines": [{"text": l} for l in lines], "confidence": "high"}

# ---------- optional Bedrock generator (best-effort) ----------
def bedrock_generator(question: str, passages: List[Dict[str,Any]], request_id: str, model_id: Optional[str] = None) -> Dict[str,Any]:
    """
    If BEDROCK_MODE enabled and boto3 present, try to call Bedrock in a tolerant way.
    The function is intentionally minimal: if bedrock fails, it returns INVALID_OUTPUT.
    """
    try:
        import boto3
    except Exception as e:
        jlog({"event":"bedrock_unavailable","detail":str(e)})
        return {"decision":"INVALID_OUTPUT"}
    model = model_id or os.getenv("BEDROCK_MODEL_ID")
    if not model:
        jlog({"event":"bedrock_no_model"})
        return {"decision":"INVALID_OUTPUT"}
    # Build prompt similar to other generator
    prompt_parts = ["PASSAGES:"]
    for p in sorted(passages, key=lambda x: int(x.get("number",0))):
        prompt_parts.append(f"{int(p.get('number',0))}. {p.get('text')}")
    prompt_parts.append("")
    prompt_parts.append("QUESTION:")
    prompt_parts.append(question)
    prompt = "\n".join(prompt_parts)
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
    body = json.dumps({"inputText": prompt})
    try:
        resp = client.invoke_model(modelId=model, body=body, contentType="application/json")
        body_stream = resp.get("body")
        raw = body_stream.read() if hasattr(body_stream,"read") else body_stream
        mr = json.loads(raw)
        text = mr.get("outputText") or mr.get("generatedText") or mr.get("text") or mr.get("content")
        if not text:
            return {"decision":"INVALID_OUTPUT"}
        # naive parsing into lines
        lines = [l.strip() for l in str(text).splitlines() if l.strip()]
        return {"decision":"ACCEPT","answer_lines":[{"text":l} for l in lines], "confidence":"high"}
    except Exception as e:
        jlog({"event":"bedrock_call_failed","detail":str(e)})
        return {"decision":"INVALID_OUTPUT"}

# ---------- retriever wrapper ----------
def retrieve(index: InMemoryIndex, query: str, raw_k: int = RAW_K, final_k: int = FINAL_K, filters: Optional[Dict[str,str]] = None) -> Dict[str,Any]:
    t0 = time.time()
    q_emb = deterministic_embedding(query, index.embed_dim)
    candidates = index.search(q_emb, raw_k=raw_k, filters=filters)
    if not candidates:
        jlog({"event":"no_candidates","query":query})
        return {"request_id": f"r-{int(t0*1000)}", "passages": [], "chunk_ids": [], "top_similarity": 0.0}
    deduped = dedupe_keep_nearest(candidates, raw_k)
    ranked = rerank_and_select(deduped, final_k)
    passages = []
    chunk_ids = []
    for i, r in enumerate(ranked):
        passages.append({
            "number": i+1,
            "chunk_id": r["chunk_id"],
            "document_id": r.get("document_id"),
            "chunk_index": r.get("chunk_index"),
            "text": r.get("text",""),
            "meta": r.get("meta",{}),
            "source_url": r.get("meta",{}).get("source_url"),
            "page_number": None,
            "score": float(r.get("final_score",0.0)),
            "distance": None,
        })
        chunk_ids.append(r["chunk_id"])
    top_sim = passages[0]["score"] if passages else 0.0
    jlog({"event":"retrieval_complete","returned":len(passages),"top_similarity":top_sim,"ms":int((time.time()-t0)*1000)})
    return {"request_id": f"r-{int(t0*1000)}", "passages": passages, "chunk_ids": chunk_ids, "top_similarity": float(top_sim)}

# ---------- validation of generator output (same rules as core.query) ----------
def validate_generator_output(gen_res: Dict[str,Any], passages: List[Dict[str,Any]]) -> Tuple[str, List[Dict[str,Any]]]:
    """
    Return (decision, validated_lines)
    """
    if not isinstance(gen_res, dict):
        return "INVALID_OUTPUT", []
    if gen_res.get("decision") == "NOT_ENOUGH_INFORMATION":
        return "NOT_ENOUGH_INFORMATION", []
    # assemble lines
    raw_lines = []
    if isinstance(gen_res.get("answer_lines"), list) and gen_res.get("answer_lines"):
        for el in gen_res["answer_lines"]:
            if isinstance(el, dict) and isinstance(el.get("text"), str):
                for ln in el["text"].splitlines():
                    if ln.strip():
                        raw_lines.append(ln.strip())
            elif isinstance(el, str):
                for ln in el.splitlines():
                    if ln.strip():
                        raw_lines.append(ln.strip())
    else:
        # try common keys
        for k in ("text","output","result","body"):
            v = gen_res.get(k)
            if isinstance(v, str) and v.strip():
                for ln in v.splitlines():
                    if ln.strip():
                        raw_lines.append(ln.strip())
                break
    if not raw_lines:
        return "INVALID_OUTPUT", []
    max_pass = max((int(p.get("number",0)) for p in passages), default=0)
    if max_pass < 1:
        return "INVALID_OUTPUT", []
    validated = []
    for ln in raw_lines:
        m = CITATION_PAT.search(ln)
        if not m:
            jlog({"event":"validation_failed","reason":"missing_citation","line":ln})
            return "INVALID_OUTPUT", []
        cited = int(m.group(1))
        if cited < 1 or cited > max_pass:
            jlog({"event":"validation_failed","reason":"citation_out_of_range","line":ln,"cited":cited,"max_pass":max_pass})
            return "INVALID_OUTPUT", []
        low = ln.lower()
        if any(s in low for s in DISALLOWED_SUBSTRINGS):
            jlog({"event":"validation_failed","reason":"disallowed_substring","line":ln})
            return "INVALID_OUTPUT", []
        validated.append({"text": ln})
    return "ACCEPT", validated

# ---------- formatters ----------
def format_for_web(res: Dict[str, Any]) -> Dict[str, Any]:
    return res

def format_for_sms(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a minimal SMS body: first answer line, stripped of citation tokens, and trimmed to 1600 chars.
    """
    if res.get("resolution") != "answer":
        guidance = res.get("guidance_key") or "no_answer"
        return {"to": None, "body": f"Cannot answer: {guidance}."}
    lines = res.get("answer_lines") or []
    first = lines[0]["text"] if lines and isinstance(lines[0], dict) else ""
    sms_text = re.sub(r"\s*\[\d+\]\s*$", "", first).strip()
    if len(sms_text) > 1600:
        sms_text = sms_text[:1597].rstrip() + "..."
    return {"to": None, "body": sms_text}

def format_for_voice(res: Dict[str,Any]) -> Dict[str,Any]:
    """
    Prefer returning a short TTS string (first 1-2 sentences without citations).
    """
    if res.get("resolution") != "answer":
        guidance = res.get("guidance_key") or "no_answer"
        return {"type":"speak","text": f"Sorry, I cannot answer: {guidance}."}
    texts = []
    for ln in res.get("answer_lines", [])[:2]:
        txt = ln["text"] if isinstance(ln, dict) else str(ln)
        txt = re.sub(r"\s*\[\d+\]\s*$", "", txt).strip()
        texts.append(txt)
    speech = " ".join(texts) or "Sorry, I cannot provide an answer right now."
    return {"type":"speak","text":speech}

# ---------- audit ----------
def write_audit(record: Dict[str,Any]) -> None:
    try:
        date_prefix = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        p = os.path.join(AUDIT_DIR, date_prefix)
        os.makedirs(p, exist_ok=True)
        key = f"{record.get('request_id')}.json"
        path = os.path.join(p, key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        jlog({"event":"audit_written","path":path,"request_id":record.get("request_id")})
    except Exception as e:
        jlog({"event":"audit_failed","detail":str(e)})

# ---------- top-level pipeline ----------
def pipeline_run(index: InMemoryIndex, query: str, mode: str = "mock", channel: str = "web", request_id: Optional[str]=None, top_k:int=FINAL_K, raw_k:int=RAW_K) -> Dict[str,Any]:
    """
    Run a single E2E pipeline:
    - validate inputs (language assumed en for local run)
    - retrieve
    - call generator (mock or bedrock)
    - validate generator output (strict)
    - format final response
    - audit
    """
    start = time.time()
    request_id = request_id or f"r-{int(start*1000)}"
    language = "en"
    # 1) retrieve
    retr = retrieve(index, query, raw_k=raw_k, final_k=top_k, filters=None)
    passages = retr.get("passages", [])
    chunk_ids = retr.get("chunk_ids", [])
    top_similarity = float(retr.get("top_similarity", 0.0))
    # similarity gate (local run, permissive)
    if not passages or top_similarity < MIN_SIMILARITY:
        res = {"request_id": request_id, "resolution": "not_enough_info"}
        write_audit({
            "request_id": request_id,
            "query": query,
            "used_chunk_ids": chunk_ids,
            "resolution": res["resolution"],
            "top_similarity": top_similarity,
            "ms": int((time.time()-start)*1000)
        })
        return res

    # 2) generator
    if mode == "bedrock" and BEDROCK_MODE:
        gen_res = bedrock_generator(query, passages, request_id)
    else:
        gen_res = mock_generator(query, passages)
    # 3) validate generator output
    decision, validated_lines = validate_generator_output(gen_res, passages)
    if decision == "NOT_ENOUGH_INFORMATION":
        res = {"request_id": request_id, "resolution": "not_enough_info"}
        write_audit({
            "request_id": request_id,
            "query": query,
            "used_chunk_ids": chunk_ids,
            "resolution": res["resolution"],
            "ms": int((time.time()-start)*1000)
        })
        return res
    if decision != "ACCEPT":
        res = {"request_id": request_id, "resolution": "invalid_output"}
        write_audit({
            "request_id": request_id,
            "query": query,
            "used_chunk_ids": chunk_ids,
            "resolution": res["resolution"],
            "gen_res": gen_res,
            "ms": int((time.time()-start)*1000)
        })
        return res

    # 4) build final response with citations hydrated
    citations = [{"citation": ln_idx+1, "chunk_id": passages[ln_idx]["chunk_id"], "source_url": passages[ln_idx].get("source_url"), "meta": passages[ln_idx].get("meta", {})} for ln_idx,_ in enumerate(validated_lines)]
    res = {
        "request_id": request_id,
        "resolution": "answer",
        "answer_lines": validated_lines,
        "citations": citations,
        "confidence": gen_res.get("confidence", "high"),
    }
    # 5) write audit
    write_audit({
        "request_id": request_id,
        "query": query,
        "used_chunk_ids": chunk_ids,
        "top_similarity": top_similarity,
        "resolution": res["resolution"],
        "gen_res_decision": gen_res.get("decision"),
        "ms": int((time.time()-start)*1000)
    })

    # 6) map to channel format
    if channel == "sms":
        return format_for_sms(res)
    if channel == "voice":
        return format_for_voice(res)
    return format_for_web(res)

# ---------- small demo corpus builder ----------
def build_demo_index() -> InMemoryIndex:
    idx = InMemoryIndex(embed_dim=EMBED_DIM)
    docs = [
        {
            "doc_id": "doc_myscheme",
            "text": (
                "myScheme is a National Platform that aims to offer one-stop search and discovery of Government schemes. "
                "It provides search based on eligibility and guides citizens on how to apply for different Government schemes. "
                "To apply, visit the official myScheme portal and use the online application form."
            ),
            "meta": {"trust_level":"gov","source_url":"https://myscheme.gov.in"}
        },
        {
            "doc_id": "doc_pmkisan",
            "text": (
                "PM-KISAN is a scheme for farmers. eKYC is mandatory for PM-KISAN Registered Farmers. "
                "OTP based eKYC is available on PMKISAN portal or by visiting a CSC location."
            ),
            "meta": {"trust_level":"gov","source_url":"https://pmkisan.gov.in"}
        },
        {
            "doc_id": "doc_csc",
            "text": (
                "Common Service Centers (CSC) provide government-to-citizen e-services in rural areas. "
                "They are access points for delivering G2C services and help make service delivery transparent."
            ),
            "meta": {"trust_level":"gov","source_url":"https://csc.gov.in"}
        },
    ]
    for d in docs:
        idx.add_document(d["doc_id"], d["text"], meta=d["meta"])
    jlog({"event":"demo_index_built","rows":len(idx.rows)})
    return idx

# ---------- CLI and self-tests ----------
def run_interactive(idx: InMemoryIndex, args: argparse.Namespace):
    print("E2E local runner (mode={}). Type queries or Ctrl-C to exit.".format(args.mode))
    try:
        while True:
            q = input("Query> ").strip()
            if not q:
                continue
            out = pipeline_run(idx, q, mode=args.mode, channel=args.channel)
            print(json.dumps(out, indent=2, ensure_ascii=False))
    except KeyboardInterrupt:
        print("\nexiting.")

def self_test(idx: InMemoryIndex) -> None:
    """
    Run a few deterministic assertions to ensure pipeline is functioning.
    """
    jlog({"event":"self_test_start"})
    # 1) simple query that should succeed
    r1 = pipeline_run(idx, "How to apply for voter id?", mode="mock", channel="web")
    assert r1.get("resolution") == "answer", f"expected answer, got {r1}"
    # 2) query with no overlap -> not_enough_info
    r2 = pipeline_run(idx, "What is the chemical formula for water (H2O)?", mode="mock", channel="web")
    assert r2.get("resolution") in ("not_enough_info","invalid_output"), f"unexpected {r2}"
    # 3) sms output shape
    s = pipeline_run(idx, "How to apply for voter id?", mode="mock", channel="sms")
    assert isinstance(s.get("body",""), str) and len(s.get("body",""))>0
    # 4) voice output shape
    v = pipeline_run(idx, "How to apply for voter id?", mode="mock", channel="voice")
    assert v.get("type") == "speak"
    jlog({"event":"self_test_ok"})
    print("self-test passed.")

def parse_args():
    p = argparse.ArgumentParser(description="E2E local runner for public-civic-info-system")
    p.add_argument("--mode", choices=["mock","bedrock"], default="mock", help="generator mode")
    p.add_argument("--query", type=str, help="one-off query")
    p.add_argument("--channel", choices=["web","sms","voice"], default="web", help="output channel mapping")
    p.add_argument("--interactive", action="store_true", help="run interactive REPL")
    p.add_argument("--self-test", action="store_true", help="run self tests and exit")
    return p.parse_args()

def main():
    args = parse_args()
    idx = build_demo_index()
    if args.self_test:
        self_test(idx)
        return
    if args.interactive:
        run_interactive(idx, args)
        return
    if args.query:
        out = pipeline_run(idx, args.query, mode=args.mode, channel=args.channel)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    print("No query provided. Use --query or --interactive or --self-test. Exiting.")

if __name__ == "__main__":
    main()
