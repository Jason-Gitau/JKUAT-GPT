# Full design brief for the **AI Study Assistant (RAG)**

This is a complete, actionable brief. It contains the product goals, our current assumptions, the feature list, the detailed architecture and pipelines, data & API schemas, prompt templates, ops/monitoring needs, testing & evaluation, cost/scale strategy, and a concrete pilot timeline. Build to these specs unless something is impossible — then propose alternatives with tradeoffs.

---

## 1) High-level goals (what we must deliver)

* **Primary goal:** Build an AI study assistant that lets students upload personal notes & lecture handouts, ask the AI to revise/quiz/clarify using those documents, join anonymous shared study groups, and receive simplified research paper summaries tailored to their level.
* **Secondary goals:** Mobile-first UX, transparent citation of sources, fast answers, low cost-per-student, easy campus adoption (JKUAT pilot).
* **Non-goals (initial):** Replace professors, host copyrighted paid journals (only open/allowed content), or build heavy proprietary models at launch.

---

## 2) Current assumptions (explicit)

> These are the assumptions to design against — they’re what we used in earlier planning and should be parameterized in the implementation.

**User and usage**

* Pilot population: JKUAT ≈ 44,000 students.
* MVP pilot target: 500–1,000 early adopters; expansion target: 10k → 44k students.
* Typical usage scenarios (three tiers, parametric):

  * Light: 50 queries / student / month (≈ 1–2/day)
  * Moderate: 150 queries / student / month (≈ 5/day)
  * Heavy (exam season): 600 queries / student / month (≈ 20/day)
* Average tokens per query: 300–800 tokens (prompt + retrieved context + response). Use 500 tokens as baseline.

**Model & infra**

* Initial inference: use cheap Chinese model APIs (example rates we used for planning: **DeepSeek ≈ \$0.10 / 1M tokens**, Qwen ≈ \$0.20 / 1M tokens). Treat these as adjustable parameters.
* Embedding generation: similar providers or local open-source embeddings (768-dim baseline).
* Vector DB: start with **Postgres + pgvector** for cost-efficiency; switch to dedicated vector DB (Weaviate, Milvus, Pinecone, Qdrant) if throughput/latency requires it.
* Frontend: React PWA (mobile-first).
* Backend: FastAPI.

**Cost & switching**

* Start API-first (no GPU CAPEX) to reduce up-front risk. Monitor token spend.
* Define decision rule to self-host models when local hosting cost < projected API cost at scale — threshold will be configurable (we can use tokens/month × API price vs monthly local infra cost).

---

## 3) User-facing feature spec (MVP & roadmap)

### MVP (must-have)

1. **Upload & Ingest**: Upload PDFs, DOCX, PPT, plain text. Basic OCR for scanned PDFs.
2. **Ask from your notes**: Chat UI — ask a question, system retrieves from *your* uploaded notes and answers with citations to chunks.
3. **Auto-generated quizzes**: Generate practice MCQs or short-answer questions from uploaded notes.
4. **Research paper summaries**: Pull papers from open sources (arXiv/HuggingFace/Semantic Scholar APIs) and produce student-level summaries.
5. **Anonymous study groups**: Ability to create or join a group, share notes, and have an AI that can summarize group notes or generate group quizzes.
6. **Feedback**: “Was this helpful?” thumbs-up/down and optional correction input for continuous improvement.

### Phase 2 (near-term)

* Reranking improvement (cross-encoder); improved summarization modes; personalized study path; progress dashboard; classroom/lecturer admin features.

### Phase 3 (longer)

* Fine-tune a domain model on local academic content; host locally on GPU cluster; advanced tutor mode; integrations (WhatsApp, Telegram, LMS).

---

## 4) System architecture (high-level components)

1. **Frontend (React PWA)**

   * Chat UI, Upload UI, Group UI, Dashboard, Admin panel.
2. **API layer (FastAPI)**

   * Auth, ingestion endpoints, query endpoints, group endpoints, feedback endpoints.
3. **Worker layer** (background jobs)

   * Document processing, embedding creation, re-indexing, OCR, summarization jobs. Use Celery/RQ.
4. **Storage**

   * Postgres (users, metadata, app data), storage bucket for raw docs (S3 / local / Cloudpap), Redis for caching/session data.
5. **Vector store**

   * Start: Postgres + pgvector. Later: Milvus / Weaviate / Pinecone / Qdrant if needed.
6. **Embedding & LLM services**

   * Initially: external API (DeepSeek / Qwen / Kimi) for both embeddings & LLM calls (configurable).
   * Later: local model hosts on university/Atlancis GPU cluster (via containerized inference servers).
7. **Monitoring & Observability**

   * Prometheus + Grafana, OpenTelemetry traces, Sentry for errors, custom dashboards for token spend and retrieval quality.
8. **Security & Governance**

   * TLS, encryption at rest, RBAC for admin functions, student consent for uploaded documents, data retention policies.

---

## 5) Data model (core tables / documents) — JSON schemas

### User

```json
{
  "user_id": "uuid",
  "email": "string",
  "student_id": "string",
  "display_name": "string",
  "created_at": "timestamp",
  "role": "student|admin|lecturer",
  "consent_uploads": true
}
```

### Document (raw upload)

```json
{
  "doc_id": "uuid",
  "owner_id": "uuid",
  "title": "string",
  "filename": "string",
  "source_type": "lecture|paper|book|pastpaper",
  "uploaded_at": "timestamp",
  "storage_path": "s3://... or local path",
  "language": "en",
  "hash": "sha256",
  "status": "processing|ready|error"
}
```

### Chunk (what we store in vector DB)

```json
{
  "chunk_id":"uuid",
  "doc_id":"uuid",
  "text":"string",
  "tokens": 312,
  "metadata": {
    "page": 3,
    "section": "Introduction",
    "source_title": "Biology lecture 3"
  },
  "embedding_ref": "vector in pgvector",
  "created_at":"timestamp",
  "checksum":"sha1"
}
```

### Group

```json
{
  "group_id":"uuid",
  "name":"string",
  "is_anonymous": true,
  "members": ["user_id", ...],
  "created_at":"timestamp"
}
```

### Query / Session Log

```json
{
  "query_id":"uuid",
  "user_id":"uuid",
  "group_id":"uuid|null",
  "query_text":"string",
  "retrieved_chunk_ids":["uuid", ...],
  "llm_response":"string",
  "tokens_input": 400,
  "tokens_output": 300,
  "cost_usd": 0.00012,
  "rating": "up|down|null",
  "created_at":"timestamp"
}
```

---

## 6) Ingestion pipeline (detailed)

1. **Upload** (frontend → /ingest endpoint)

   * Validate type & size, store raw file in object storage, enqueue ingestion job.
2. **Processing worker**

   * OCR (if needed), text extraction (pdfminer/poetry/Apache Tika fallback), clean text, remove boilerplate.
3. **Chunking**

   * Strategy: 200–500 tokens per chunk (baseline 300), overlap 50 tokens. Keep chunks semantically coherent (paragraph boundaries when possible).
4. **Metadata enrichment**

   * Title, author, page, course code (if provided), tags, language.
5. **Duplicate detection**

   * Hash entire doc and store. Within-chunk near-duplicate removal using shingling or simhash to avoid storage bloat.
6. **Embeddings**

   * Generate embeddings for each chunk using selected embedding model (API call or local). Use 768-dim embeddings initially.
7. **Store**

   * Insert chunk + embedding vector + metadata into pgvector table; store chunk text references in Postgres.
8. **Indexing**

   * Ensure HNSW indexes created (via pgvector) for fast ANN retrieval.
9. **Post-process**

   * Create inverted index / term map for hybrid search.

---

## 7) Query & RAG pipeline (detailed flow)

1. **User query → /query endpoint**

   * Authenticate user; optional group context.
2. **Query preprocessing**

   * Normalize, detect language, optionally rewrite/clarify short queries (lightweight).
3. **Query embedding**

   * Create embedding for query.
4. **Retrieve (ANN)**

   * Retrieve top-K (K = 8–12) chunks by cosine similarity from vector DB.
   * Optionally do keyword filter by metadata (e.g., only user’s documents or group documents when requested).
5. **Rerank (optional but recommended)**

   * Use a light cross-encoder reranker (small transformer) to rank the top-K results by semantic match to query. This improves precision and reduces tokens we send to LLM.
6. **Context assembly**

   * Filter duplicates, keep high-quality chunks (max total context tokens <= threshold), add short metadata citations for each chunk.
   * Optionally compress / summarize retrieved chunks (if tokens too big).
7. **Prompt template composition**

   * Use a strict prompt that instructs the LLM to use only the provided context and to cite chunks. (See sample prompts below.)
8. **LLM call**

   * Call selected LLM (external API) with the assembled prompt & context. Use streaming where possible for UX.
9. **Post-process**

   * Parse LLM output; extract citations; if answer is low-confidence, add fallback message or ask clarification.
10. **Return response**

* Save query log (tokens, cost), return answer & citations to user.

---

## 8) Prompt templates (practical)

**Short answer + citations**

```
You are an academic study assistant. Use ONLY the information in the following sources to answer. If the answer is not present, say "I don't know; please provide more notes."

SOURCES:
[1] SOURCE_META: {title, page, author}
{chunk 1}
---
[2] ...
---

QUESTION:
{user_query}

INSTRUCTIONS:
- Answer concisely in plain English.
- Provide numbered citations like (source #1, p.3).
- If you mention facts not found in sources, label them "assumption".
- End with 3 suggested practice questions.
```

**Summarization (paper -> student level)**

```
You are a study summarizer. Summarize the following research paper for a third-year undergraduate in simple language. Include: 1) 3-sentence summary, 2) 5 bullet takeaways, 3) 3 study questions. Cite section or page numbers where possible.
PAPER:
{paper text or key chunks}
```

**Reranker scoring prompt (for LLM-based reranker)**

```
Given the query: "{query}", and this candidate passage: "{passage}", score relevance 0-10 and return only the number and short reason. Higher score = more relevant.
```

---

## 9) API endpoints (core)

### Auth

* `POST /auth/login` — returns JWT
* `POST /auth/refresh`

### Ingestion

* `POST /api/v1/ingest` — upload file metadata; returns job id

  * payload: `{user_id, doc_title, file_url, tags, course_code, public|private}`
* `GET /api/v1/ingest/{job_id}` — status

### Query

* `POST /api/v1/query`

  * payload:

    ```json
    {
      "user_id":"uuid",
      "group_id": "uuid|null",
      "query":"string",
      "max_tokens_context": 2000,
      "response_mode":"concise|detailed|quiz"
    }
    ```
  * response: `{answer: str, citations: [{chunk_id, doc_id, score}], tokens_in, tokens_out, cost_usd, query_id}`

### Groups

* `POST /api/v1/groups` — create
* `POST /api/v1/groups/{id}/join`
* `GET /api/v1/groups/{id}/notes` — retrieve group-shared notes

### Feedback

* `POST /api/v1/feedback`

  * payload: `{query_id, rating: up|down, correction: optional string}`

### Admin / Metrics

* `GET /admin/metrics?from=&to=` — returns token usage, costs, DAU/MAU, avg latency, retrieval quality

---

## 10) Caching & optimization (must implement)

* **Query result cache** (Redis) keyed by canonicalized query + doc-scope. TTL = 1–7 days depending on content volatility.
* **Prompt & response cache** for repeated/FAQ queries.
* **Embedding cache** for repeated identical chunks & queries to reduce embedding API calls.
* **Batched embedding & generation** for high-throughput to exploit provider batch endpoints and lower per-call overhead.
* **Pagination & truncation** when building context to keep LLM token usage efficient.

---

## 11) Reliability, latency & performance targets

* **Cold query latency (end-to-end)**: ≤ 2.5s for retrieval (ANN) + ≤ 1–3s for LLM streaming first tokens (depends on API). Aim for <5s for concise answers.
* **99th percentile response**: <15s (for long, summarized answers).
* **Retrieval precision\@5**: > 0.7 (goal to measure).
* **Hallucination target**: keep "unsupported claim" rate < 5% on human-evaluated sample.

---

## 12) Monitoring & metrics (essential)

* **Business metrics**: DAU, MAU, retention (7/30-day), paid conversion %, churn.
* **Performance metrics**: avg latency, 95/99p latency, tokens-in/out per minute, embedding calls/min.
* **Cost metrics**: tokens/month, \$/month by provider, cost per query, cost per student per month.
* **RAG quality metrics**: precision\@k, recall\@k (use labeled test set), reranker accuracy, human rating scores for answers.
* **Ops metrics**: error rates, ingestion pipeline failures, queue backlog length.

---

## 13) Security, privacy & compliance

* **Encryption**: TLS in transit; encrypt PII/storage at rest for uploaded docs (AES-256).
* **Access control**: JWT + roles; group anonymity enforced by not storing mapping for anonymous groups unless explicitly requested.
* **Data retention**: configurable by user; default remove raw uploads after N months unless user flags keep.
* **Consent**: explicit checkbox for document upload + terms explaining that documents are used to generate embeddings (and possibly used in aggregated analytics).
* **Copyright/DMCA**: provide takedown flows; only auto-index public/open content for research feeds.
* **University SSO**: integrate SAML/OAuth2 with JKUAT if possible (optional).

---

## 14) Testing & evaluation plan

* **Unit tests** for ingestion, chunking, DB ops.
* **Integration tests**: ingestion → embedding → retrieval → end-to-end response.
* **Load testing**: simulate spikes (exam season); test retrieval and API rate-limits.
* **Human evaluation**: sample N=200 queries per week, human raters judge faithfulness, relevance, hallucination.
* **Automated LLM-judge**: complement human evaluation with LLM scoring for scale, but always cross-check for bias.

---

## 15) Ops & deployment (recommended stack)

* **Language & framework**: Python + FastAPI.
* **Queue**: Redis + RQ or Celery.
* **DB**: Postgres + pgvector (start), migrate to Milvus/Weaviate if needed.
* **Cache**: Redis.
* **Containerization**: Docker + Kubernetes (or K3s on local infra).
* **CI/CD**: GitHub Actions -> deploy to staging -> promote to prod.
* **Observability**: Prometheus, Grafana, Sentry, OpenTelemetry.
* **Storage**: S3-compatible (DigitalOcean Spaces or local Ceph/MinIO on JKUAT).

---

## 16) Cost & scale strategy (actionable)

* **Phase A (prototype / pilot)**: API-based embeddings + LLM (DeepSeek) + Postgres+pgvector on cheap VPS. No GPU investment.

  * Keep aggressive caching, reranking, hybrid search to minimize LLM calls.
* **Instrumentation**: track tokens/month per user and total. Use billing alarms.
* **Decision rule (switching to self-host)**:

  * Let `API_monthly_cost = tokens_per_month * api_price_per_1M / 1M`.
  * If `API_monthly_cost > Local_monthly_infra_cost * (1 + ops_buffer)`, evaluate migration.
  * Example params: ops\_buffer = 0.2 (20% margin for ops costs).
* **Rough threshold**: This will depend on model prices. Monitor continuously; target moving to local GPU cluster when predictable steady usage > 10k–50k active students (model-specific).

---

## 17) Pilot plan with JKUAT (6–10 weeks)

**Goal:** 500–1,000 students, measure DAU, retention, usefulness, and token spend.

**Week 0–1**: Infra & skeleton API + React demo, ingestion pipeline skeleton.
**Week 2–4**: Build ingestion, retrieval, sample LLM flows for notes Q\&A and paper summarization. Instrument metrics.
**Week 5–6**: Enroll pilot cohort, integrate auth (email + student id), run pilot. Collect QA labels & feedback.
**Week 7–8**: Iterate on prompts, reranker thresholds, UX. Produce pilot report: adoption, retention, infra cost, top failure modes.
**Deliverables:** working demo, admin dashboard, pilot report with metrics and recommendations for full roll-out.

---

## 18) Acceptance criteria (MVP)

* Students can upload a doc and within 5 minutes it is searchable.
* Query endpoint returns relevant answer with citations for 80% of the test set (human judged).
* End-to-end median latency < 5s for concise answers.
* Token spend per 1,000 pilot users < budget threshold (configurable).
* Feedback loop in place (user ratings saved and sampled weekly).

---

## 19) Deliverables I expect from you (AI system designer)

1. Architecture diagram (boxes & arrows) + deployment plan.
2. DB schema and vector table definitions, index configs.
3. Implementation of these endpoints: `/ingest`, `/query`, `/groups`, `/feedback`, `/admin/metrics`.
4. Chunking & embedding worker with configurable chunk size, overlap, embedding model.
5. Retrieval + rerank pipeline implementation and tuning knobs.
6. Sample prompt templates (we provided); create a prompt-versioning mechanism.
7. Monitoring dashboards (tokens/month, retrieval precision, latency).
8. CI/CD pipeline and deployment on a test cluster.
9. Pilot run plan and metrics reporting format.

---

## 20) Immediate action items (start here)

* Provision a small cluster / VPS and Postgres + pgvector.
* Implement the ingestion worker and a simple retrieval + LLM query flow using DeepSeek API (configurable provider).
* Implement the `/query` endpoint and a minimal React chat UI.
* Add logging for tokens, latency, and cost per request.
* Run ingestion for 10 sample lecture notes and test queries & summaries.

---

## 21) Notes, constraints & edge cases

* **Anonymity:** for anonymous groups we must avoid storing mappings that deanonymize members unless explicitly allowed. Provide opt-in for identity disclosure.
* **Copyright:** don’t index/serve paywalled copyrighted content unless licensed. For research paper feeds, rely on open sources (arXiv, institutional repositories, or licensed content).
* **Hallucination safety:** require the model to explicitly cite sources and include a “confidence” statement — and if no supporting chunk exists, respond "I don’t know" or ask for clarification.
* **Data leakage:** ensure that private uploads are not used to train external models; state this clearly in TOS.

---

### Closing (tone)

Build modular, measurable, and configurable. Keep costs and token usage visible from day 1. The product’s value is the *workflow* (notes → AI-driven revision → group study → research digests), not raw model power. Focus on reliable retrieval and crisp UI — that’s where students feel value.


