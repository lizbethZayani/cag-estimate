# CAG Estimate

**Context-Augmented Generation for Project Estimation** - An intelligent API and a real-time Streamlit chat UI that generate detailed project estimations from meeting transcriptions using Anthropic's Claude AI.

## Overview

CAG Estimate uses the **Context-Augmented Generation (CAG)** architecture pattern to analyze meeting transcriptions and generate comprehensive project estimations including:
- Detailed task breakdown
- Hour estimates per task
- Cost calculations
- Team size recommendations
- Project timeline estimates
- Key assumptions and dependencies

The project ships with three ways to use it:
- **FastAPI backend** (`/api/v1/estimate`, `/api/v1/estimate/stream`) — see [API Documentation](#api-documentation)
- **Streamlit chat UI** (`src/ui/streamlit_app.py`) — a conversational interface with real-time, token-by-token streaming; see [💬 Streamlit Chat Interface](#-streamlit-chat-interface)
- **Rails web UI** (`../estimator-web/`) — a form-based UI over the structured endpoint; see [Web UI (Rails)](#web-ui-rails)

## 📦 What's Included

#### 1. **API Service** - `/api/v1/estimate`
Context-Augmented Generation endpoint that:
- Accepts a meeting transcription and an hourly rate
- Runs input guardrails, an exact cache, a semantic cache, a structured LLM call and output guardrails (see [Request pipeline](#request-pipeline))
- Returns a validated `ProjectEstimation` (tasks, hours, costs, summary) plus `prompt_version` and `cached`

#### 2. **Configuration System**
- `config.py` - Pydantic `BaseSettings`, loads `.env`
- Every setting is listed in [Configuration](#configuration)

#### 3. **Context Examples**
- Reference estimation projects in `context/examples.py`
- Rendered into the system prompt as benchmarks for the model

#### 4. **Prompt templates, schemas, guardrails, caches**
- Versioned Jinja2 prompts, Pydantic schemas with validators, input/output guardrails, exact (Redis) and semantic (Redis Stack) caches; see [Architecture](#architecture)

#### 5. **FastAPI Endpoints**
- `POST /api/v1/estimate` - Structured estimation (JSON)
- `POST /api/v1/estimate/stream` - Markdown estimation as Server-Sent Events
- `GET /health` - Service health check
- `GET /` - API information
- `/docs` - Interactive Swagger UI
- `/redoc` - ReDoc documentation

#### 6. **Streamlit Chat Interface** - `src/ui/streamlit_app.py`
- Two response modes (streaming markdown, structured table); see [Streamlit Chat Interface](#-streamlit-chat-interface)

#### 7. **Tests**
- Offline unit tests (no network, no Redis): run with `uv run pytest`; see [Running tests](#running-tests)
- For end-to-end checks against the running API, the evals and the Rails tests see [VERIFICATION.md](VERIFICATION.md)

### 🚀 Quick Start

**1. Start the API:**
```bash
.venv/bin/python -m uvicorn cag_estimate.main:app --host 127.0.0.1 --port 8001
```

**2. Open API Documentation:**
```
http://127.0.0.1:8001/docs
```

**3. Make an Estimation Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "Your meeting transcription...",
    "hourly_rate": 40
  }'
```

**4. Verify Results:** follow the end-to-end checklist in [VERIFICATION.md](VERIFICATION.md).

**5. Launch the Streamlit Chat UI:**

The UI reads the `API_BASE_URL` environment variable (default `http://localhost:8000`). Make sure the API is reachable there first (e.g. `docker-compose up`, or `uvicorn cag_estimate.main:app --host 0.0.0.0 --port 8000`), or set `API_BASE_URL` to the dev port used in step 1 (`API_BASE_URL=http://127.0.0.1:8001`).

```bash
./run_streamlit.sh
# or manually:
.venv/bin/python -m streamlit run src/ui/streamlit_app.py
```
Then open http://localhost:8501 and chat with the estimation assistant in real time.

### 📊 Example: Grocery Price Comparison App (iOS, output from an earlier run)

**Input (Meeting Transcription):**
```
"we have to estimate a new feature for the mobile app in iOS only, than most include 
a new chat with an agent than support the grocery shopping carts creation base on the 
grocery shops around where you live as food is quite becoming expensive we most save 
money on food so we most copare the prices of the produces than i want to eat, so we 
most compare also the labels with the ingredientes to filter out the most natural food 
and organic so estimate this project"
```

**Output (Generated Estimation):**
```
Project: Grocery Price Comparison & Smart Shopping Mobile App (iOS)
Total Hours: 384
Total Cost: $15,360
Team Size: 2 developers (1 iOS, 1 Backend) + 1 QA Engineer
Duration: 10 weeks
Hourly Rate: $40/hour

Tasks Included:
  1. Project Setup & Architecture Design (Medium) - 12h
  2. User Authentication & Profile Management (Medium) - 14h
  3. Location Services & Store Discovery (Medium) - 16h
  4. Live Chat Agent Integration (High) - 24h
  5. Grocery Store API Integration (High) - 28h
  ... and 13 more tasks
```

**Token Usage & Cost:**
```
Input Tokens: 3,625
Output Tokens: 3,275
Total Tokens: 6,900
API Cost: $0.0200 (about 2 cents at $1 / $5 per 1M tokens)
```

**Verification Result:** ✅ ALL VALIDATIONS PASSED

### 🎯 URLs & Access Points

| URL | Purpose |
|-----|---------|
| `http://127.0.0.1:8001/` | API root with info |
| `http://127.0.0.1:8001/health` | Health check endpoint |
| `http://127.0.0.1:8001/api/v1/estimate` | Main estimation endpoint |
| `http://127.0.0.1:8001/docs` | **Swagger UI** (interactive) |
| `http://127.0.0.1:8001/redoc` | ReDoc documentation |
| `http://127.0.0.1:8001/openapi.json` | OpenAPI schema |
| `http://localhost:8501` | **Streamlit Chat UI** (real-time chat) |
| `http://localhost:8000/api/v1/estimate/stream` | SSE streaming endpoint used by the chat UI |

### ✨ Key Features

✅ **Context-Augmented Generation** - LLM uses reference examples for consistent, high-quality estimations  
✅ **Real-Time Streaming Chat UI** - Token-by-token responses in a WhatsApp-style Streamlit interface  
✅ **Validated Output** - Pydantic validators, guardrails and retries on every structured estimate  
✅ **Two-level Cache** - Exact (Redis) and semantic (Redis Stack) cache, fail-soft  
✅ **Cost Transparent** - See exact token costs for every request  
✅ **Reasonable latency** - typically 15-30 seconds per structured estimate (longer for multi-turn or reviewed estimates)  
✅ **Affordable** - roughly 1-2 cents per structured estimate with Haiku 4.5 (see Token Pricing)  
✅ **Production Ready** - CORS enabled, error handling, comprehensive logging  
✅ **Well Documented** - README, verification guide, interactive API docs  
✅ **Tested** - offline pytest suite (`uv run pytest`)  

### 📁 Project Structure

```
cag-estimate/
├── src/
│   ├── cag_estimate/
│   │   ├── main.py                      # FastAPI app, logging setup
│   │   ├── config.py                    # Pydantic BaseSettings
│   │   ├── logging_config.py            # structlog level from LOG_LEVEL
│   │   ├── dependencies.py              # FastAPI dependency factories
│   │   ├── routers/estimations.py       # HTTP only: /estimate + /estimate/stream
│   │   ├── services/
│   │   │   ├── estimation.py            # EstimationService (pipeline)
│   │   │   ├── llm_wrapper.py           # LiteLLM Router + Instructor
│   │   │   └── cache.py                 # Exact-match Redis cache
│   │   ├── prompts/
│   │   │   ├── loader.py                # render_estimation_prompt()
│   │   │   └── estimation/v1/           # system.j2, user.j2, examples.j2
│   │   ├── schemas/estimation.py        # Request/response models + validators
│   │   ├── guardrails/                  # input.py, output.py, errors.py
│   │   ├── cache/semantic.py            # Semantic cache (Redis Stack)
│   │   ├── context/examples.py          # Reference estimation examples
│   │   └── embedding_pipeline/          # Session 7: chunk + embed historical budgets
│   │       ├── schemas.py               # Budget in, Chunk / EmbeddedChunk / IngestResponse out
│   │       ├── chunker.py               # JSONStructuralChunker (one component = one chunk)
│   │       ├── embedder.py              # OpenAIEmbedder, cost estimate, cosine_similarity
│   │       ├── service.py               # EmbeddingIngestService (chunk -> embed -> stats)
│   │       └── router.py                # POST /embeddings/ingest
│   └── ui/
│       ├── streamlit_app.py             # Chat UI (streaming + structured)
│       ├── estimate_client.py           # HTTP client with safe error messages
│       └── view_models.py               # Pure presentation helpers
├── tests/                               # Offline pytest suite
├── scripts/compare.py                   # CLI: cosine similarity between two texts
├── data/                                # Sample budgets (budgets_sample.json, see Embedding pipeline)
├── session4/NOTES.md                    # Session 4 changes vs the reference repo
├── run_streamlit.sh                     # Launches the chat UI
├── Dockerfile                           # API container image
├── docker-compose.yml                   # api + redis-stack + postgres + estimator-web
├── docker-compose.override.yml          # Dev overrides (hot-reload)
├── DOCKER.md / VERIFICATION.md          # Docker and verification guides
└── pyproject.toml                       # Project configuration
```

### 💡 For Managers: Confidence Levels

**✅ GREEN** - Estimation Approved
- All validations pass
- Mathematically correct
- Reasonable ranges
- Ready to use

**⚠️ YELLOW** - Review Required
- Minor validation warnings
- May need team discussion
- Check assumptions

**❌ RED** - Estimation Rejected
- Critical validation failures
- Unrealistic values
- Request re-estimation

## Architecture

### CAG Pattern (Context-Augmented Generation)

```
┌─────────────────────────────────────────────────────────┐
│ System Prompt                                           │
│ ├─ Role: Software Estimation Expert                    │
│ ├─ Instructions: Estimation guidelines                 │
│ └─ Context: Reference examples from previous projects  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ User Message: Meeting Transcription                     │
│ (The actual project to be estimated)                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Claude AI (Haiku 4.5) - Fast & Efficient               │
│ Analyzes context + instructions + transcription        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Structured Output: Project Estimation                   │
│ ├─ Tasks with hours & costs                            │
│ ├─ Team composition                                     │
│ ├─ Timeline                                            │
│ └─ Assumptions                                          │
└─────────────────────────────────────────────────────────┘
```

### Layering

```
routers (HTTP only)  ->  services (orchestration)  ->  prompts / guardrails / cache / schemas
```

- `routers/estimations.py` parses the request, calls `EstimationService` and maps errors to HTTP status codes. It holds no business logic and no prompt text.
- `services/estimation.py` runs the pipeline below. The LLM wrapper, caches and moderation client are injected (`dependencies.py`), so tests replace them with fakes.
- `prompts/`, `guardrails/`, `cache/` and `schemas/` are leaf modules.

### Request pipeline

`POST /api/v1/estimate` runs these steps in order (`EstimationService.estimate`):

1. **Input guardrail** (moderation, prompt injection, PII). A rejected input never reaches a cache or the LLM.
2. **Exact cache** lookup (Redis, SHA-256 key of transcription, hourly rate, prompt version and model).
3. **Semantic cache** lookup (Redis Stack vector search), only on an exact miss and only when enabled.
4. **Render prompt** from the versioned Jinja2 templates and call the **structured LLM**.
5. **Output guardrail** (leak check, total-cost correction).
6. **Store** the result in both caches.

A hit in step 2 or 3 returns `cached: true` and skips the LLM. `POST /api/v1/estimate/stream` runs only the input guardrail and then streams a Markdown answer; it does not use the caches, structured validation or the output guardrail.

### Prompt templates

Prompts live in `src/cag_estimate/prompts/estimation/<version>/` as Jinja2 files: `system.j2`, `user.j2` and the `examples.j2` partial included by `system.j2`. `render_estimation_prompt(request, version, output_format)` renders them with `StrictUndefined`, so a missing variable fails loudly. The allowed hour ranges, task counts and rate bounds in the prompt come from the schema constants. `output_format` is `"json"` (structured endpoint) or `"markdown"` (stream endpoint).

To add a prompt version:

1. Copy `prompts/estimation/v1/` to `prompts/estimation/v2/`.
2. In `v2/system.j2`, change the include to `{% include "estimation/v2/examples.j2" %}` (it is hardcoded to `v1`).
3. Edit the templates.
4. Change `PROMPT_VERSION = "v1"` to `"v2"` in `services/estimation.py`. The version is part of the exact-cache key and the semantic-cache bucket, so v1 entries are not served for v2.

Cache keys use the request inputs, not the rendered prompt text, so editing wording within a version keeps existing entries.

### Structured output and validators

`LLMWrapper.complete_structured()` calls the model through Instructor with `ProjectEstimation` as `response_model` and up to 3 retries; Instructor feeds validator errors back to the model. The call goes through the LiteLLM Router, so the primary-to-fallback model behaviour is kept. Validators in `schemas/estimation.py`:

| Rule | Value |
|------|-------|
| Task count | 3 to 20 |
| Total hours | 8 to 500; derived from the task hours (see below) |
| Hourly rate (in the result) | 30 to 150 |
| Hours per complexity | Simple 4-24, Medium 10-64, High 16-112 |
| `complexity` | `Simple`, `Medium` or `High` |

`summary.total_hours` is derived, not trusted: when it differs from the sum of the task hours it is replaced by the sum (logged as `estimate_total_hours_corrected` with `claimed` and `computed`), and the 8 to 500 range is checked on the corrected value. Models often slip on this arithmetic and repeat the slip on every Instructor retry, which used to end as a 502. For the same reason a `summary` sent as a JSON-encoded string is parsed into the object. Every other rule still raises and triggers a retry.

If retries are exhausted, the exception reaches the router and becomes a generic 502.

### Guardrails and HTTP mapping

| Where | Reason / failure | HTTP result |
|-------|------------------|-------------|
| Input | `moderation` (OpenAI moderation, only when `OPENAI_API_KEY` is set; fails open on errors) | 400 `{"reason", "message"}` |
| Input | `prompt_injection` (regexes for phrases such as "ignore previous instructions") | 400 |
| Input | `pii` (email, IBAN, phone with 9-13 digits) | 400 |
| Request body | missing or invalid fields (FastAPI validation) | 422 |
| Output | system-prompt markers in the result | 502 |
| Output | `total_cost_usd` different from `total_hours x hourly_rate` | corrected silently, logged |
| LLM / structured retries exhausted / any other error | - | 502 with a generic message |

Error bodies never echo the offending text or the exception message. On the stream endpoint a rejected input is a single SSE `error` event with HTTP 200 (carrying `reason`), so the Streamlit client can show the guardrail message.

### Semantic cache

`cache/semantic.py` stores results in a RediSearch vector index (redisvl, cosine distance, embeddings from `EMBEDDING_MODEL` through the OpenAI API). A stored result is reused only when the bucket (`<prompt_version>:<hourly_rate>`) matches exactly and the similarity is at least `SEMANTIC_CACHE_THRESHOLD` (default `0.90`).

- **Requires Redis Stack** (RediSearch). The compose file uses `redis/redis-stack:7.4.0-v0`; plain `redis:7-alpine` does not work.
- **Requires `OPENAI_API_KEY`** for the embeddings. Without it the semantic cache is disabled and the pipeline still works.
- `SEMANTIC_CACHE_ENABLED=false` disables it. `SEMANTIC_CACHE_LOG_ONLY=true` logs would-be hits without serving them (useful to calibrate the threshold). `SEMANTIC_CACHE_TTL` is the entry lifetime in seconds.
- **Fail-soft:** setup, lookup and store errors are logged and treated as a miss; they never fail a request.

## 💬 Streamlit Chat Interface

A conversational frontend, located at [`src/ui/streamlit_app.py`](src/ui/streamlit_app.py), that talks to the FastAPI backend. A **Response mode** radio in the sidebar picks the mode. The default, **Conversation (memory)**, is documented in [Conversation memory (sessions)](#conversation-memory-sessions); the two single-shot modes are:

- **Streaming (markdown)**: calls `POST /api/v1/estimate/stream` and shows the answer token by token (SSE). Token and cost metrics are shown after each answer.
- **Structured (JSON)**: calls `POST /api/v1/estimate` and renders a summary, a task table (task, complexity, hours, cost) and the totals. A badge shows whether the answer was served from cache or freshly generated. No token or cost metrics are shown in this mode.

Both modes use an hourly rate of 40. Messages are kept in `st.session_state` (**New Chat** clears them); in these two modes the backend is single-shot and does not receive the history as context. The API URL comes from the `API_BASE_URL` environment variable (default `http://localhost:8000`).

If the API rejects the input, the UI shows a friendly message per guardrail reason (`moderation`, `prompt_injection`, `pii`); any other failure shows a generic message and the raw server text is never displayed.

Why Markdown for streaming? Streaming raw JSON cannot be displayed or validated until it is complete, so the stream endpoint asks the model for readable Markdown (`output_format="markdown"` in the prompt templates). The structured endpoint is the one with schema validation, guardrails on the output and caching.

### Running it

```bash
# 1. Start the API first (the UI expects it at http://localhost:8000)
docker-compose up
# or: uvicorn cag_estimate.main:app --host 0.0.0.0 --port 8000

# 2. Start the chat UI
./run_streamlit.sh
# or manually:
.venv/bin/python -m streamlit run src/ui/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

## Conversation memory (sessions)

Sessions let several estimates in one conversation share context: the model remembers the project across turns, and PDF, Word (`.docx`) or text attachments become part of the input.

### Start the project and run the tests

```bash
# API on port 8000
docker-compose up                      # or, locally from this directory:
PYTHONPATH=src uv run uvicorn cag_estimate.main:app --host 0.0.0.0 --port 8000

# Streamlit UI on http://localhost:8501 (from this directory)
API_BASE_URL=http://localhost:8000 PYTHONPATH=src uv run streamlit run src/ui/streamlit_app.py

# Tests (offline: LLM, Redis and moderation are faked)
PYTHONPATH=src uv run pytest -q
```

`PYTHONPATH=src` is needed on macOS: `uv run` re-syncs the editable install and the generated `.pth` file gets the hidden flag, so Python skips it and `cag_estimate` is not importable without it.

The Streamlit default mode, **Conversation (memory)**, creates a session lazily, offers a transcript box plus a multi-file uploader (`pdf`, `docx`, `txt`, `md`, `csv`, `json`), shows the accumulated `project_metadata` in a sidebar expander, and has a **New conversation** button that opens a fresh session.

### Endpoints

| Method and path | Purpose |
|-----------------|---------|
| `POST /api/v1/sessions` | Create a session. `201` with `{"session_id": "<uuid4>"}` |
| `GET /api/v1/sessions/{id}` | Snapshot: `session_id`, `project_metadata`, `turns`, `max_turns` (window size), `message_count` (messages in the window, `turns * 2`), `created_at` and `updated_at` (UTC ISO datetimes). `404` if unknown or expired |
| `DELETE /api/v1/sessions/{id}` | Discard a session now. `204` with no body; `404` if unknown or expired |
| `POST /api/v1/sessions/{id}/estimate` | Estimate within the session (`multipart/form-data`) |
| `POST /api/v1/sessions/{id}/estimate-acb` | Same contract, with an independent critic review ([Reviewed estimates](#reviewed-estimates-actor-critic)) |

Fields of `POST /api/v1/sessions/{id}/estimate`:

- `transcript` (required): up to 80,000 characters, not blank.
- `hourly_rate` (optional, default `40`): between the same bounds as `/estimate`.
- `attachments` (optional, repeatable): PDF, `.docx`, `.txt`, `.md`, `.csv` or `.json` (legacy `.doc` is not supported). Defaults: at most 5 files, 5,000,000 bytes per file, 80,000 characters for transcript plus attachment text combined.

The response is the regular estimation (`result`, `prompt_version`, `cached`, which is always `false` here) plus `session_id`, `project_metadata`, `turns` and `attachments` (the file names used).

| Status | When |
|--------|------|
| 400 | Input guardrail rejection: `{"reason", "message"}` (moderation, prompt injection, PII), applied to the transcript and the attachment text together |
| 404 | Unknown session |
| 413 | Too many files, a file or the combined text over the limits |
| 415 | Unsupported attachment type |
| 422 | Blank transcript, hourly rate out of bounds, or an unreadable attachment (corrupt PDF or `.docx`, or a document with no extractable text) |
| 502 | Any other failure; the message is generic and never carries upstream detail |

```bash
SID=$(curl -s -X POST http://localhost:8000/api/v1/sessions | python3 -c 'import sys, json; print(json.load(sys.stdin)["session_id"])')

curl -s -X POST "http://localhost:8000/api/v1/sessions/$SID/estimate" \
  -F 'transcript=Project Falcon: a bookstore web shop in Django with Stripe payments.' \
  -F 'hourly_rate=40' \
  -F 'attachments=@requirements.pdf'

# Second turn: the project name is not repeated, the session remembers it
curl -s -X POST "http://localhost:8000/api/v1/sessions/$SID/estimate" \
  -F 'transcript=Now add an admin dashboard.'

curl -s "http://localhost:8000/api/v1/sessions/$SID"
```

### Attachment path: B (extract the text in the service)

The practice brief offers two paths: A, upload files to a provider Files API and reference them in the message, or B, extract the text in our service and append it to the transcript. This project uses **Path B** ([`services/attachments.py`](src/cag_estimate/services/attachments.py)), because:

- **Provider-agnostic.** The LLM layer is LiteLLM with an Anthropic to OpenAI fallback. A Files API is provider-specific, so a fallback call would not see the files.
- **Offline-testable.** Text extraction (`pypdf` for PDFs, `python-docx` for `.docx`, UTF-8 decoding for the text formats) runs without network, so the whole flow is covered by tests with a fake LLM.
- **Guardrails see the document.** Moderation, prompt-injection and PII checks run on the transcript plus the attachment text combined, so an attachment cannot smuggle in what the transcript could not.

Each attachment is appended after the transcript as `--- attachment: <filename> ---` followed by its text. File names are sanitized (path components and control characters removed, runs of three or more dashes collapsed) so a name cannot fake a separator. Limits are enforced while reading (a file is never fully buffered past its limit) and again on the combined text. For `.docx`, the text of non-empty paragraphs comes first, followed by the table rows (cells joined with ` | `, one row per line); this keeps the code simple at the cost of losing the exact position of tables. Scanned PDFs have no text layer and are rejected with `422`; there is no OCR.

### How `project_metadata` is extracted

`project_metadata` has `project_name`, `assumed_team_size`, `mentioned_technologies` and `agreed_scope`. After each successful turn it is updated by a **deterministic heuristic** ([`services/metadata_extractor.py`](src/cag_estimate/services/metadata_extractor.py)) over the structured estimate the model already returned:

- `project_name` and `assumed_team_size` (first number in the team-size text) come from the estimate's fields.
- `agreed_scope` is the estimate's meeting summary, whitespace-collapsed and truncated to 300 characters.
- `mentioned_technologies` come from scanning the transcript (attachments included) and the task names, descriptions and includes against a curated vocabulary of about 35 technologies.

Merging keeps non-empty new values (the latest name, team size and scope win) and takes the union of technologies, case-insensitively.

Why not a second LLM call as extractor: it adds a call, cost, latency and a new failure mode to every turn, and it would make the memory tests depend on a live model. The heuristic is free, deterministic and testable offline. Its limits: only technologies in the vocabulary are detected (ambiguous names such as `Go` and `R` are excluded on purpose), and the scope is a truncated summary rather than a curated statement of what was agreed.

### Memory versus history

- **Memory (`project_metadata`)** holds long-term facts. It is injected into the system prompt as a `<project_metadata>` block (rendered by prompt `v1`, which `v2` includes); with empty metadata, the first call of a session, no block is rendered and the prompt is identical to a non-session one.
- **History** is a sliding window of the last `SESSION_MAX_TURNS` user/assistant pairs (default 6). When it overflows, the oldest pairs leave the window, always as whole pairs, so there is never an orphan assistant message. They are not simply lost, see *History compression* below.
- **History compression.** Evicted pairs are split in two. A user message that states a hard constraint becomes an **anchor** and stays for the whole session (only its compact user text, at most 500 characters, is kept, never the assistant JSON); at most `SESSION_MAX_ANCHORS` (default 5) are pinned and the oldest extra one is folded into the summary. Every other evicted pair is folded into one cumulative **summary**, capped at `SESSION_SUMMARY_MAX_CHARS` (default 2000). `SESSION_SUMMARY_MODE=llm` (default) asks the LLM to merge the previous summary with the evicted turns (the user text plus a short digest of each estimate: project name, total hours, total cost, task count); `extractive` writes one bullet per user message, dropping the oldest bullets first, and is also the automatic fallback when the LLM call fails or returns nothing. Set `SESSION_SUMMARY_MODEL` to a cheap model to keep the cost down. The conversation is passed to the summarizer as data, with an instruction not to follow anything written in it. Anchor kinds: `nda`, `signed_contract`, `scope_frozen`, `budget_locked`, `compliance` (HIPAA, GDPR, SOX, PCI, FDA, ISO 27001), `deadline_hard`, `contractual`, `explicit_commitment`. When there are anchors or a summary, the model receives one extra user message right after the system prompt, answered by a short assistant "Understood." so roles keep alternating:

  ```text
  [Earlier conversation context - the turns below are the live thread]
  Pinned constraints from earlier turns:
  - (nda) This project is under NDA ...
  Summary of earlier turns:
  - first evicted request ...
  ```

  `SESSION_COMPRESSION_ENABLED=false` restores the plain drop. Limitations: anchors are regex heuristics (they miss paraphrases and only skip simple negations such as "no NDA needed"), and the LLM summary can lose detail or omit a fact (the extractive mode is the deterministic alternative). Cost and latency: the summary costs one extra short LLM call (at most 600 output tokens) only on turns that evict from the window, and it runs inside the session lock, so that turn is slower and the next request on the same session waits for it. The summary prompt contains conversation text, so these calls bypass the exact cache and nothing is stored in Redis. A compaction problem never fails the request. `GET /api/v1/sessions/{id}` reports `anchors_count` and `summary_chars`.
- **The system prompt is not part of the window.** It is regenerated from the current metadata on every call, so it is always present and never trimmed.
- **Compact user turn.** The full text of a turn (transcript plus attachments) is sent to the model once, in the current message. What is stored in history is the transcript truncated to 4,000 characters plus the names of the attachments, so a large document does not repeat in every later turn. The assistant side stores the estimate as JSON.
- **A failed turn changes nothing.** History and metadata are updated only after the estimate succeeds and passes the output guardrail.

### Audience tier and conversational prompt v2

Sessions use prompt **`v2`** (`prompts/estimation/v2/`); the non-session `/estimate` endpoints keep **`v1`**, which is unchanged (a test pins its exact bytes). `v2` includes the whole `v1` system prompt (schema, rules, `<project_metadata>`) and adds:

- `<conversation_rules>`: the history is established context, the metadata block is the source of truth unless the user explicitly corrects it, and the model must always return the **updated FULL estimate** for everything agreed so far, never a delta, keeping earlier tasks unless the user removes scope and keeping `project_name` stable.
- `<audience>`: a short framing chosen by the tier (executive: business outcomes, plain language, risks in business terms; pm: delivery, dependencies, milestones; developer: technical detail, components, integration points; default: balanced).
- A closing line in the user message asking for the full updated estimate.

Why: in a live session each follow-up ("Add a barcode scanning page") was estimated almost like a separate small project (total hours went 182, 124, 126, 172), instead of the updated full estimate. The response reports `prompt_version: "v2"`.

The tier comes from `services/tier_resolver.py`, a **heuristic** (vocabulary cues, not a classifier; it can be wrong). It runs on the combined text of the turn (transcript plus attachments) and the metadata known before the turn. The first matching rule decides:

1. `override`: the optional multipart field `tier` (`executive`, `pm`, `developer`, `default`) on `POST /sessions/{id}/estimate`; any other value is a `422`.
2. `executive_cues`: CEO, CFO, board, investors, budget approval, ROI, executive summary...
3. `pm_cues`: sprint, backlog, roadmap, milestone, stakeholder, timeline, deliverable...
4. `developer_cues`: API, endpoint, schema, microservice, refactor, framework names...
5. `technology_count`: the project already mentions 3 or more technologies.
6. otherwise `default`.

Cues are matched case-insensitively on whole words. A mixed message goes to its most senior reader (executive beats pm beats developer). The resolved tier and the rule that decided it are stored on the session after a successful turn only: `GET /sessions/{id}` returns `last_tier` and `last_tier_rule` (`null` before the first turn) and the estimate response returns `resolved_tier` and `tier_rule`. In the Streamlit "Conversation (memory)" mode, the **Audience** selector (Auto, Executive, PM, Developer) sends `tier` only when not on Auto, and the memory panel shows the resolved tier.

### Reviewed estimates (actor-critic)

`POST /api/v1/sessions/{id}/estimate-acb` takes the same multipart fields and returns the same errors (`400`, `404`, `413`, `415`, `422`, generic `502`) as `/estimate`, and audits each draft before it is returned. The response is the usual session response plus an `acb` object:

```json
"acb": {
  "iterations": [{"iteration": 0, "decision_after": "accept", "critic_verdict": "accept",
                  "critic_confidence": 90, "critic_failed": false, "issue_summary": []}],
  "final_decision": "accept",
  "iterations_run": 1,
  "caveats": []
}
```

**Cost.** One reviewed request costs up to `ACB_MAX_ITERATIONS` x (draft + review) LLM calls (2 drafts and 2 reviews with the default), and like every session estimate it bypasses the caches. Use `/estimate` when speed or cost matters.

**The session changes once.** Drafts are never written to the history. Only the final estimate is committed, with exactly the history format `/estimate` stores, so the two routes can be mixed in one conversation. A failure before that point leaves the session untouched.

- **Actor**: the normal session estimator. On a retry it receives the critic's latest review in an optional `<critic_feedback>` block of the `v2` user prompt (severity, category, field path, description and suggested fix of each issue).
- **Critic** (`services/critic.py`): an independent auditor, one structured LLM call per draft, with the transcript and the estimates treated as data, never instructions. It returns a verdict (`accept`, `needs_iteration`, `reject`) and issues in eight categories: `math_error`, `hallucination`, `scope_mismatch`, `missing_task`, `history_inconsistency`, `unrealistic_estimate`, `missing_assumption`, `tier_mismatch`. `needs_iteration` requires at least one critical or major issue. If the critic call fails for any reason, the draft is accepted and the trace flags it (`critic_failed`).
- **Boss** (`services/boss.py`): makes no LLM call. `accept` returns the draft; `needs_iteration` asks the actor for a new draft while iterations remain; `reject`, or an exhausted budget, returns the LAST draft unmodified and lists the open issues in `BossTrace.caveats` (at most 5).
- `ACB_MAX_ITERATIONS` (default `2`) is one draft plus one directed retry, so the worst case is 2 actor and 2 critic calls. `CRITIC_MODEL` selects a cheaper model for the critic.
- **Fail-open critic.** If the critic call fails, the draft is accepted and its round is flagged with `critic_failed: true` in `acb.iterations` (confidence `0`).
- **Caveats are not part of the estimate.** When the budget runs out or the critic rejects, the last draft is returned unmodified and the open issues (at most 5) are listed in `acb.caveats`; `result` and the stored history stay clean.

In Streamlit, the **Independent review (actor-critic)** checkbox (off by default) of the conversation mode uses this route and shows a **Review trace** expander with the rounds, the final decision and the caveats as a warning.

### Caching and volatility

- **Session estimates bypass the exact and semantic caches.** The answer depends on the history and the accumulated metadata, not only on the submitted text, so a cache keyed on the text could serve an answer from another context.
- **Sessions live in process memory.** They are lost on restart, are not shared between workers or processes, and are never persisted. `SESSION_MAX_SESSIONS` (default 1000) caps the store; beyond it the least recently used session is evicted and its id then answers `404`. `SESSION_IDLE_TTL_SECONDS` (default `0`, disabled) expires a session left idle for that long: its id then answers `404`. Use is what counts, so every request on a session restarts its timer; expiry is lazy (checked when the store is used, no background thread) and a session that is mid-request is never expired. `DELETE /api/v1/sessions/{id}` frees one explicitly, and the Streamlit **New conversation** button calls it for the old session. A later stage would move sessions to Redis or a database with a shared store so any worker can serve any session.
- Requests on the same session are serialized with a lock, so concurrent turns cannot interleave.

### Configuration

`SESSION_MAX_TURNS`, `SESSION_MAX_SESSIONS`, `SESSION_IDLE_TTL_SECONDS`, `SESSION_COMPRESSION_ENABLED`, `SESSION_MAX_ANCHORS`, `SESSION_SUMMARY_MAX_CHARS`, `SESSION_SUMMARY_MODE`, `SESSION_SUMMARY_MODEL`, `ATTACHMENT_MAX_FILES`, `ATTACHMENT_MAX_BYTES`, `ATTACHMENT_MAX_CHARS_TOTAL`, `TRANSCRIPT_MAX_CHARS`, `ACB_MAX_ITERATIONS`, `CRITIC_MODEL` and `LLM_STRUCTURED_MAX_TOKENS` are listed in [Environment Variables](#environment-variables). `docker-compose.yml` passes them to the `api` service with the defaults shown there. The structured token limit matters in conversations: later turns can produce larger estimates, and a limit that is too low truncates the JSON and fails the request.

### Tests

The Step 7 integration tests are in [`tests/test_session_integration.py`](tests/test_session_integration.py) and use `httpx.AsyncClient` against the real app with a recording fake LLM: chained requests update the metadata, an attached PDF changes the estimate (a qualitative test of the plumbing: the fake LLM reacts to the text it receives), and eight turns never put more than `SESSION_MAX_TURNS` pairs in the history sent to the LLM. Endpoint tests are in `tests/test_sessions_endpoints.py`.

## Evals

A small golden dataset of scripted conversations, run against a **running** API and scored by deterministic metrics. Code: `src/cag_estimate/evals/` (`dataset.py`, `golden_dataset.json`, `metrics.py`, `runner.py`, `run.py`); the JSON ships inside the wheel.

**What they are.** Structural and regression checks: totals add up, counts and hours stay in loose ranges, expected vocabulary appears, guardrails fire with the right reason and without echoing the input, the audience tier matches, and conversation memory stays consistent across turns.

**What they are NOT.** They do not judge whether an estimate is good, and there is no LLM judge: scores from a second model would add cost, variance and a new thing to trust. For that reason `deepeval` is not a dependency. The runner never fakes a result: every turn is a real HTTP request, and a transport error is an errored case, never a pass.

### Run

```bash
# 1. See the plan and the number of LLM calls. Sends nothing.
PYTHONPATH=src uv run python -m cag_estimate.evals.run --dry-run

# 2. Run it against the running stack. COST: this calls the real LLM
#    (one estimate per turn; --mode acb adds up to ACB_MAX_ITERATIONS drafts and reviews per turn).
PYTHONPATH=src uv run python -m cag_estimate.evals.run --base-url http://localhost:8000 --output report.json
```

Options: `--mode standard|acb`, `--limit N`, `--case ID` (repeatable), `--output PATH` (JSON report), `--timeout SECONDS`, `--dry-run`. A cost banner is printed before the first request. Exit codes: `0` every case passed every metric, `1` some case failed or errored, `2` usage problem or the API's `/health` is unreachable (nothing was estimated).

### Metrics

| Metric | Checks (final turn unless stated) |
|--------|-----------------------------------|
| `schema_consistency` | `total_hours` equals the sum of task hours; `total_cost_usd` equals hours x rate (1 USD tolerance); 3 to 20 tasks; valid complexity; `prompt_version` present |
| `range_bounds` | task count and total hours inside the case's ranges |
| `content_recall` | every `expected_in_text` term appears (case-insensitive) in project name, summary, task names and descriptions; at least one `expected_technologies_any_of` appears there or in `project_metadata` |
| `guardrail_behavior` | expected case: HTTP 400 with exactly that `reason` and no echo of the input; any other case: a 400 is a failure |
| `tier_expectation` | `resolved_tier` equals `expected_resolved_tier` |
| `memory_consistency` | only with `check_memory`: every turn is 200, `turns` counts 1, 2, 3..., `project_name` is the same on every turn (case-insensitive), total hours never decrease |

A metric that does not apply to a case reports `n/a` and passes. An errored case fails every metric.

### Add a case

Append an object to `golden_dataset.json`: `id` (3 to 64 characters, unique), `description` (say what regression it guards), `turns` (one or more messages of 20 to 80,000 characters sent in order to ONE session) and any expectations (`hourly_rate`, `tier`, `expect_guardrail_reason`, `expected_in_text`, `expected_technologies_any_of`, `expected_task_count_range`, `expected_hours_range`, `expected_resolved_tier`, `check_memory`). The offline test suite checks that guardrail cases really trigger `check_input` with the declared reason, that tier cases really resolve to their tier with `resolve_tier`, that other cases do not trip a guardrail, and that ranges are sane.

### Limitations

- **Flaky by nature.** The model is not deterministic and the ranges are loose on purpose (they catch regressions, not exact values), so a rare failure may be noise. Re-run before concluding anything; tighten or loosen a range with the evidence.
- **Tier cases depend on the heuristic.** They test `services/tier_resolver.py` (cue words and rule order); changing the cues can change the expected tier.
- **Memory rules are strict.** A renamed project or fewer total hours after an additive follow-up fails the case, even if a human might accept it.
- **Cost and time.** A full run makes about 20 estimate calls; `--mode acb` multiplies that.

## Embedding pipeline (session 7 pre-exercise)

A minimal pipeline that chunks historical budgets structurally and embeds each chunk with OpenAI `text-embedding-3-small`. Code: `src/cag_estimate/embedding_pipeline/`.

**What it is.** `POST /embeddings/ingest` takes budgets, builds one chunk per budget component and returns the chunks with their embedding vectors and run stats. `scripts/compare.py` embeds two texts and prints their cosine similarity.

**What it is NOT.** The vectors are returned over HTTP and are **not persisted**. Persistence, retrieval and other chunking strategies come in later sessions.

### How it works

1. `JSONStructuralChunker` makes one chunk per `BudgetComponent`, with no overlap and no splitting. The chunk text starts with a header from the parent budget (project summary, client sector, year, main technology) so a component keeps its context once embedded alone. `chunk_id` is `{budget_id}::{component_id}`; a duplicate id raises an error. `metadata` (budget id, component id, sector, main technology, year, complexity, estimated hours) is returned with the chunk but **not embedded**. `token_count` is measured with tiktoken.
2. `OpenAIEmbedder` sends the chunk texts in batches of 100. A rate-limit error is retried up to 3 times, waiting 1s, 2s and 4s. Out-of-credit errors (`insufficient_quota` / `credit_balance_exhausted`) are **not** retried, because waiting does not fix them. No `dimensions` is sent, so vectors have the model default of 1536 dimensions.
3. `stats` come from the chunks' token counts and the constant `EMBEDDING_PRICE_PER_M_TOKENS_USD` (`0.02`, USD per 1M tokens) in `embedder.py`. The price changes over time, so treat `estimated_cost_usd` as an estimate.

| File | Role |
|------|------|
| `schemas.py` | Pydantic models: `Budget`, `BudgetComponent`, `Chunk`, `EmbeddedChunk`, `IngestRequest`, `IngestResponse`, `IngestStats` |
| `chunker.py` | `JSONStructuralChunker`, `TiktokenCounter` |
| `embedder.py` | `OpenAIEmbedder` (batching, retries), cost estimate, hand-written `cosine_similarity` |
| `service.py` | `EmbeddingIngestService`: chunk, embed, build stats |
| `router.py` | `POST /embeddings/ingest` |
| `scripts/compare.py` | CLI that compares two texts |

### Endpoint: `POST /embeddings/ingest`

The route is `/embeddings/ingest`, **not** under `/api/v1`. Request body: `{"budgets": [...]}` with at least one budget, and each budget needs at least one component. Unknown fields are ignored.

```json
{
  "budgets": [
    {
      "budget_id": "B-001",
      "client_metadata": {"name": "Acme Retail", "sector": "retail", "country": "ES"},
      "project_summary": "Online store with catalog and checkout",
      "main_technology": "Python",
      "year": 2024,
      "total_estimated_hours": 120,
      "components": [
        {
          "component_id": "C-01",
          "name": "Checkout API",
          "description": "REST endpoints for cart and payment",
          "tech_stack": ["FastAPI", "PostgreSQL"],
          "estimated_hours": 120,
          "complexity": "medium",
          "dependencies": []
        }
      ]
    }
  ]
}
```

Response `200`: `{"chunks": [...], "stats": {...}}`. Each chunk has `chunk_id`, `text`, `metadata`, `token_count` and `embedding`. `stats` has `total_budgets`, `total_chunks`, `total_tokens` and `estimated_cost_usd`.

| Status | When |
|--------|------|
| `200` | Chunks embedded |
| `422` | Invalid body (FastAPI validation), e.g. an empty `budgets` list |
| `500` | The embeddings API failed. Generic body, details only in the logs. This is a documented exception to the 502 rule (see `AGENTS.md`) that follows the exercise brief |
| `503` | `OPENAI_API_KEY` is not configured |

Try it with curl (save the example body above as `/tmp/budget.json` first):

```bash
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d @/tmp/budget.json
```

It is also in Swagger at **http://localhost:8000/docs** under the `embeddings` tag.

### Compare two texts

`scripts/compare.py` embeds both texts and prints the cosine similarity (computed by hand, no numpy). `scripts/` and `data/` are mounted read-only into the `api` container.

Inside the container (run from the directory with `docker-compose.yml`):

```bash
docker compose exec api python scripts/compare.py --text-a "Login form with OAuth" --text-b "Sign-in page using Google OAuth"
```

Outside the container, with a complete `.env` (`OPENAI_API_KEY`, and also `ANTHROPIC_API_KEY` because `Settings` requires it):

```bash
cd cag-estimate
PYTHONPATH=src uv run python scripts/compare.py --text-a "Login form with OAuth" --text-b "Sign-in page using Google OAuth"
```

`PYTHONPATH=src` is there because plain `uv run` can lose the editable install on macOS. Output is three lines: `Text A: ...`, `Text B: ...` and `Cosine similarity: 0.xxxx`. Exit codes: `0` success, `1` the embedding request failed, `2` bad usage (empty text) or no OpenAI key.

### Sample data

`data/budgets_sample.json` (15 historical budgets) is provided by the program and is **not committed by default**. Put it in the project directory at `data/budgets_sample.json`, next to `docker-compose.yml`. The endpoint expects `{"budgets": [...]}`. If the file is a bare list, wrap it and send it:

```bash
python3 -c 'import json; print(json.dumps({"budgets": json.load(open("data/budgets_sample.json"))}))' > /tmp/ingest_body.json
curl -s -X POST http://localhost:8000/embeddings/ingest -H "Content-Type: application/json" -d @/tmp/ingest_body.json
```

If the file already has a top-level `budgets` key, send it directly with `-d @data/budgets_sample.json`.

### Dependencies and tests

After changing dependencies (for example `tiktoken`), rebuild only the API image:

```bash
docker compose up -d --build --no-deps api
```

Tests for this module are offline (fake OpenAI client):

```bash
PYTHONPATH=src uv run pytest -q tests/test_embedding_*.py tests/test_compare_script.py
```

### Limitations

- Real calls need OpenAI credit; without it OpenAI answers `429` (`insufficient_quota`) and the endpoint returns `500`.
- tiktoken downloads its encoding file on first use, so the first run needs network access.
- A chunk above 8191 tokens (the model's input limit) is only logged as a warning, not split.
- Nothing is stored: every call embeds again and costs tokens.

## Web UI (Rails)

A Rails 8 web UI lives in [`../estimator-web/`](../estimator-web/README.md). Its root page is a **conversation** over the session endpoints (`/api/v1/sessions/...`): follow-up messages, file attachments, an audience selector and a memory panel (project, team, technologies, scope, turns), with a "New conversation" button that deletes the session. The original one-shot form (`POST /api/v1/estimate`, not the streaming one) stays at `/estimations/new` as "Single estimate", and successful estimates are kept as History in Postgres. Sessions live in the API process memory, so an API restart ends the conversation and the UI starts a new one. The Streamlit chat above, with its streaming markdown mode, is still available.

```bash
# from this directory
docker compose up -d --build estimator-web postgres api redis
```

Open **http://localhost:3000**. See the [estimator-web README](../estimator-web/README.md) for the architecture, tests and known notes, and [DOCKER.md](./DOCKER.md) for the compose services.

## Getting Started

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Configuration

Create or update `.env` file:

```env
ANTHROPIC_API_KEY=your_api_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
APP_ENV=development
LOG_LEVEL=DEBUG
```

### Run the API

```bash
# Development (with auto-reload)
uvicorn cag_estimate.main:app --reload --host 127.0.0.1 --port 8001

# Production
uvicorn cag_estimate.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, access the interactive API docs:
- **Swagger UI**: http://127.0.0.1:8001/docs
- **ReDoc**: http://127.0.0.1:8001/redoc
- **OpenAPI Schema**: http://127.0.0.1:8001/openapi.json

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8001/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "CAG Estimate API"
}
```

### Generate Estimation

**Endpoint:** `POST /api/v1/estimate`

#### Example: Grocery Price Comparison App

```bash
curl -X POST http://127.0.0.1:8001/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "we have to estimate a new feature for the mobile app in iOS only, than most include a new chat with an agent than support the grocery shopping carts creation base on the grocery shops around where you live as food is quite becoming expensive we most save money on food so we most copare the prices of the produces than i want to eat, so we most compare also the labels with the ingredientes to filter out the most natural food and organic so estimate this project",
    "hourly_rate": 40
  }'
```

#### Response Structure

```json
{
  "result": {
    "project_name": "string",
    "meeting_summary": "string",
    "tasks": [
      {
        "task_id": 1,
        "name": "string",
        "description": "string",
        "estimated_hours": 24,
        "estimated_cost_usd": 960,
        "complexity": "Medium",
        "includes": ["deliverable1", "deliverable2"]
      }
    ],
    "summary": {
      "total_hours": 168,
      "total_cost_usd": 6720,
      "team_size": "2 developers",
      "estimated_duration_weeks": 5,
      "hourly_rate": 40,
      "assumptions": ["assumption1", "assumption2"]
    }
  },
  "prompt_version": "v1",
  "cached": false
}
```

Token usage and cost metrics are no longer part of the structured response; they are reported by the stream endpoint in its final `done` event. Errors: `400 {"reason", "message"}` for rejected input, `422` for an invalid body, generic `502` otherwise.

## Verification Pipeline

The checks below are now enforced in the code: the schema validators (`schemas/estimation.py`) and the output guardrail run on every structured request. For an end-to-end check of the current API see [VERIFICATION.md](VERIFICATION.md). The ranges listed here are the original guidance; the enforced values are in [Structured output and validators](#structured-output-and-validators).

### 1. Schema Validation ✓
- Verify response matches expected JSON structure
- Validate all required fields are present
- Check data types (hours as int, costs as float, etc.)

### 2. Business Logic Validation ✓
- Verify `total_hours = sum(task hours)`
- Verify `total_cost = sum(task costs)` or `total_hours * hourly_rate`
- Check `estimated_duration_weeks ≥ total_hours / (40 * team_members)`
- Ensure complexity levels are valid (Simple|Medium|High)

### 3. Reasonableness Checks ✓
- Hours per task should be within realistic bounds:
  - Simple: 4-8 hours
  - Medium: 12-24 hours
  - High: 24-48+ hours
- Total project hours should be reasonable (8-500 hours)
- Hourly rate should be between $30-$150

### 4. Content Quality Validation ✓
- Task descriptions should be detailed and clear
- Includes list should have 3-8 items per task
- Assumptions should be realistic and specific
- Meeting summary should capture key requirements

### 5. Consistency Checks ✓
- Task complexity should match estimated hours
- Team size should match estimated hours and duration
- All assumptions should be relevant to the project

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required) | - |
| `OPENAI_API_KEY` | Fallback model, moderation and semantic-cache embeddings | unset |
| `LLM_PROVIDER` | LLM provider | `anthropic` |
| `LLM_MODEL` | Primary model | `claude-haiku-4-5-20251001` |
| `FALLBACK_MODEL` | Fallback model | `gpt-4o-mini` |
| `LLM_TIMEOUT_SECONDS` | LLM call timeout | `60` |
| `LLM_NUM_RETRIES` | Router retries | `2` |
| `LLM_STRUCTURED_MAX_TOKENS` | Output token limit for structured (estimate) calls | `8192` |
| `SESSION_MAX_TURNS` | User/assistant pairs kept in a session's history window | `6` |
| `SESSION_MAX_SESSIONS` | Cap on in-memory sessions (oldest-used evicted first) | `1000` |
| `SESSION_IDLE_TTL_SECONDS` | Seconds of inactivity after which a session expires (`0` = never) | `0` |
| `SESSION_COMPRESSION_ENABLED` | Pin anchor turns and summarize evicted history (`false` = plain drop) | `true` |
| `SESSION_MAX_ANCHORS` | Anchor (hard-constraint) turns pinned per session | `5` |
| `SESSION_SUMMARY_MAX_CHARS` | Longest cumulative summary of evicted turns, in characters | `2000` |
| `SESSION_SUMMARY_MODE` | How evicted turns are summarized: `llm` (one short LLM call, extractive fallback on failure) or `extractive` (no LLM) | `llm` |
| `SESSION_SUMMARY_MODEL` | Model for the summary call; use a cheap one. Empty = the primary model | empty |
| `ATTACHMENT_MAX_FILES` | Attachments per session estimate | `5` |
| `ATTACHMENT_MAX_BYTES` | Bytes per attachment | `5000000` |
| `ATTACHMENT_MAX_CHARS_TOTAL` | Characters of transcript plus attachment text | `80000` |
| `TRANSCRIPT_MAX_CHARS` | Longest transcript accepted by the session estimate endpoint | `80000` |
| `ACB_MAX_ITERATIONS` | Reviewed estimates: drafts the actor may produce (`2` = one draft plus one directed retry; worst case 2 actor + 2 critic calls) | `2` |
| `CRITIC_MODEL` | Model for the critic calls; use a cheap one. Empty = the primary model | empty |
| `REDIS_URL` | Redis (Stack) URL | `redis://localhost:6379/0` |
| `CACHE_TTL_SECONDS` | Exact-cache TTL | `86400` |
| `EMBEDDING_MODEL` | Embedding model for the semantic cache | `text-embedding-3-small` |
| `SEMANTIC_CACHE_ENABLED` | Enable the semantic cache | `true` |
| `SEMANTIC_CACHE_THRESHOLD` | Minimum cosine similarity for a hit | `0.90` |
| `SEMANTIC_CACHE_TTL` | Semantic entry TTL in seconds | `86400` |
| `SEMANTIC_CACHE_LOG_ONLY` | Log would-be hits without serving them | `false` |
| `APP_ENV` | Application environment | `development` |
| `LOG_LEVEL` | structlog level (unknown names fall back to `INFO`) | `DEBUG` |
| `API_BASE_URL` | Used by the Streamlit UI only | `http://localhost:8000` |

`.env.example` does not list the `EMBEDDING_MODEL` and `SEMANTIC_CACHE_*` variables; add them to your `.env` if you want to change the defaults.

## Token Pricing (Claude Haiku 4.5)

- **Input tokens**: $1 per 1M tokens
- **Output tokens**: $5 per 1M tokens (these are the prices in `MODEL_COSTS` in `services/llm_wrapper.py`)
- **Example cost**: about 1-2 cents per structured estimate (roughly 3-4k input and 3-4k output tokens). Conversation turns cost more as the history grows, and reviewed estimates (actor-critic) make extra calls per turn.

## Features

✅ **Context-Augmented Generation** - Uses reference examples to improve estimation quality  
✅ **Detailed Task Breakdown** - Each task includes hours, costs, and deliverables  
✅ **Team Composition** - Recommends optimal team size and roles  
✅ **Timeline Estimates** - Calculates project duration in weeks  
✅ **Cost Analysis** - Provides both LLM API costs and project costs  
✅ **Assumption Tracking** - Documents key assumptions made during estimation  
✅ **Interactive Documentation** - Swagger UI for easy testing  
✅ **Streaming Chat UI** - Real-time, token-by-token chat interface built with Streamlit  
✅ **Production Ready** - CORS enabled, error handling, logging  

## Development

### Running Tests

```bash
uv run pytest          # whole suite, offline (no network, no Redis, no API keys)
uv run pytest -k pipeline_order -v
uv run ruff check .    # lint
```

The suite fakes the LLM, Redis (fakeredis) and the RediSearch index, so it does not prove the real Redis Stack behaviour; use [VERIFICATION.md](VERIFICATION.md) for that.

### Code Quality

```bash
uv run ruff check .
```

## Troubleshooting

### ModuleNotFoundError: No module named 'cag_estimate'

```bash
# Reinstall in editable mode
uv pip install -e . --force-reinstall
```

### API Port Already in Use

```bash
# Use a different port
uvicorn cag_estimate.main:app --host 127.0.0.1 --port 8002
```

### ANTHROPIC_API_KEY Error

```bash
# Verify .env file exists and has valid API key
cat .env

# Test API key
curl https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY"
```

## Performance

- **Latency**: typically 15-30 seconds per structured estimate (measured on the local stack with Haiku 4.5); cache hits answer in well under a second
- **Cost**: about 1-2 cents per structured estimate (see Token Pricing)
- **Efficient**: Haiku 4.5 is the cheap, fast default model
- **Scalable**: Stateless API design supports horizontal scaling

## License

MIT

## Support

For issues, feature requests, or contributions, please open an issue or submit a pull request.

---

**Built with ❤️ using [Anthropic Claude API](https://docs.anthropic.com)**
