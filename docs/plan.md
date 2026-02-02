

## Overall architecture (two decoupled systems)

### A — **Indexing / Ingestion Pipeline** (async, batch-first)

Purpose: turn raw public documents into citationable indexed chunks.

Flow:

1. Raw sources → S3 (ingest bucket).
2. S3 event or scheduled job → ingestion Lambda (starter).
3. Step Functions orchestrates per-document pipeline:

   * Language detection (Amazon Comprehend or heuristic)
   * OCR: **Textract** for supported languages; **Indic OCR / Tesseract** via container Lambda for Indic scripts if required.
   * Text normalization, removal of noise, table extraction (container Lambda if native tools needed).
   * Chunking (300–700 tokens), metadata enrichment (source_id, title, url, region, language, last_updated).
   * Embedding generation (Amazon Bedrock/Titan embeddings).
   * Write embeddings → OpenSearch Serverless (vector index); write chunk metadata → DynamoDB; store processed text → S3.
4. Verification job: sample-check ingestion (human/volunteer) — store verification status.

Storage:

* S3 (raw + processed)
* OpenSearch Serverless / OpenSearch (vector)
* DynamoDB (chunk metadata, source map)
* (Optional) S3 versioning for source provenance

Idempotency & retries: use idempotent chunk IDs; Step Functions with retry policies and DLQ (SQS).

---

### B — **Inference / Access Pipeline** (runtime, low latency, ZIP Lambdas)

Purpose: fast user queries, voice/text, RAG → grounded answers with citations.

Flow:

1. Client (Web PWA / SMS / IVR) → API Gateway.
2. API Gateway → `query-router` ZIP Lambda:

   * Detect channel, language, region; normalize query.
   * If voice: call Amazon Transcribe (or pass audio to Transcribe).
   * If SMS: parse short command (stateless).
3. `query-router` → `embed-query` (use Bedrock embeddings) or call Bedrock directly as needed.
4. Vector search → OpenSearch (top-k chunks, filtered by language/region).
5. Build RAG prompt: include retrieved chunk contents with chunk IDs and metadata. Inject strict instruction: “Only use the supplied chunks. Cite chunk_ids after paragraphs.”
6. Call Bedrock LLM for grounded answer.
7. Confidence & safety checks:

   * Compute retrieval confidence (based on similarity, chunk last_updated).
   * If confidence < threshold or blocked intent (medical/legal/emergency), return safe fallback (helpline or “I don’t know”).
8. Post-processing: format answer + map chunk_ids → (title, source_url, last_updated) for UI/SMS/Voice.
9. If voice: synthesize with Amazon Polly; if SMS: ensure ≤ 160–320 chars summary; include short source text.
10. Return to user.

Services used at runtime:

* API Gateway
* Lambda (ZIP) — query-router, answer-gen
* Amazon Bedrock — embeddings + LLM
* OpenSearch Serverless — vector retrieval
* Amazon Transcribe (voice → text)
* Amazon Polly (TTS)
* Amazon SNS / Pinpoint (SMS fallback)
* CloudWatch (metrics + logs), QuickSight (optional dashboards)

Deployment model:

* ZIP Lambdas for all runtime components.
* Container Lambdas only if OCR ingestion requires native binaries (kept entirely in indexing pipeline).

---

## Data & citation model

* **Chunk schema** (must be implemented): `chunk_id`, `content` (300–700 tokens), `embedding`, `metadata` (`source_id`, `title`, `source_url`, `source_type`, `language`, `region`, `topic`, `last_updated`, `ingest_time`, `extraction_method`).
* **Citation policy:** Every factual statement must be traceable to one or more `source_id`. UI must display at least title + source_url + last_updated for each cited source. If web retrieval is used, include fetch timestamp and domain.
* **Trust hierarchy:** gov > ngo > structured public API > news > web (fallback). LLM answers must prefer higher-trust chunks.

---

## UX requirements & flows (must satisfy judges)

* **Zero-learning curve:** one primary action (Ask / Speak / SMS). No signup. No long forms.
* **Language-first:** first screen or first prompt asks for language or detects automatically.
* **Voice-first:** push-to-talk supported; short answers with offer “Would you like more?”.
* **Low-bandwidth:** PWA with offline cache + SMS fallback.
* **Actionable results:** every answer ends with a next step (visit CSC, call helpline).
* **Transparency:** show sources, last_updated, and a “why this answer” 1-line rationale.
* **Safety:** block medical/legal advice; provide referrals.

---

## Evaluation & metrics (what to collect and present)

* **Accuracy**: manual annotation of 100 test queries per language; report precision, recall.
* **Resolution rate**: % queries answered confidently (no escalation).
* **Latency**: median, p95 for text and voice paths.
* **Hallucination rate**: % answers with false claims vs. ground truth.
* **User comprehension**: post-use 1–2 question survey (5-point scale).
* **Adoption proxy**: number of unique users in pilot.

Map these explicitly to judging rubric: Ideation/Impact/Technical/Completeness.

---

## Security, privacy & responsible AI

* Do not store PII unnecessarily; anonymize logs. Document retention policy.
* Safety filters: block/redirect medical/legal queries.
* Model guardrails: require that LLM responses include source_ids; refuse when no credible source.
* Maintain provenance: store `source_url` and ingest timestamp for every chunk.

---

## Minimal deliverables (hackathon submission)

1. Public demo URL (PWA) + SMS number (or documented SMS flow).
2. GitHub repo with `frontend/`, `backend/zip/`, `backend/container/` (if used), `infra/` (SAM or minimal CloudFormation), README.
3. Running prototype (deployed or local instructions) that demonstrates:

   * Voice Q → answer with citation (1 regional language)
   * SMS Q → short answer with citation
   * Admin view: show cited chunks and ingestion metadata
4. 3-minute video demo (Problem, Demo scenarios, Architecture, Metrics from test users).
5. Technical blog / ARCHITECTURE.md describing ingestion pipeline, chunk schema, citation policy and evaluation metrics.
6. Evaluation dataset (100 queries per language) + results.

Acceptance criteria for demo:

* End-to-end voice answer with citation in target local language.
* SMS text reply showing source and next step within 160–320 chars.
* Admin page showing chunk metadata and verification status.

---

## Roles & team split (1–4 people)

* Product / Domain researcher — source selection, user testing, persona.
* Frontend / UX — PWA, offline, SMS integration.
* Backend / ML engineer — ingestion pipeline, OpenSearch, Bedrock calls.
* Documentation / Presentation — video, blog, README, deck.

---

## Minimal implementation timeline (opinionated; adapt to remaining days)

* Day 0: Finalize beneficiary & 3 demo scenarios.
* Day 1–3: Curate 50–200 authoritative documents, implement ingestion (S3 → chunk → metadata) and write to OpenSearch.
* Day 4–6: Implement ZIP Lambdas (query router, embed + retrieval, answer generation) + Bedrock integration mock (if access limited, simulate with smaller LLM, but document clearly).
* Day 7: Voice path (Transcribe + Polly), SMS integration (SNS). Run internal tests.
* Day 8: User tests (10–20 users), collect metrics, iterate.
* Day 9: Finalize video + deck + blog + repo cleanup.
  (Adjust days to your schedule; priority order must be preserved.)

---

## Demo script (exact three scenarios to record)

1. Voice (regional language): ask eligibility for a scheme → system replies with short answer + “Source: [title] (gov.in) — Next step: Visit CSC with Aadhaar”.
2. SMS: user texts `SCHEME FARMER` → system replies with 1–2 SMS lines with source.
3. Safety fallback: user asks medical prescription → system replies with refusal + nearest government hospital contact.

Show admin: chunk metadata for the cited source, last_updated, extraction_method.

---

## Risks & mitigations

* **OCR accuracy for Indic scripts** → use IndicOCR or manual transcription for critical docs; show extraction_method in citation.
* **Hallucinations** → RAG only from retrieved chunks; LLM instructed to cite chunk_ids and refuse when no credible chunk found.
* **Rate limits / Bedrock access** → caching answers for repeated queries; degrade gracefully to cached KB responses.
* **SMS/IVR limitations** → document SMS demo as mocked if telco integration unavailable; still show full flow.

---

## What to say to judges (exact phrasing, 1–2 sentences)

* Architecture pitch:
  “We implemented a decoupled, serverless architecture: an asynchronous ingestion pipeline builds auditable, multilingual vector indexes from verified public documents, and a low-latency ZIP-Lambda inference pipeline performs retrieval-grounded generation with mandatory citations, voice and SMS fallbacks — optimized for inclusion and responsible AI.”
* Why this is correct:
  “This design delivers low latency and operational simplicity while ensuring answers are verifiable, localized, and safe.”

---
