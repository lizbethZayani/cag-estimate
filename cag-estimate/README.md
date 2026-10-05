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
- `tests/verify_api.py` and `tests/test_verification.py` are the older verification scripts; see [VERIFICATION.md](VERIFICATION.md)

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
API Cost: $0.000060 (less than a penny!)
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
✅ **Fast Results** - 3-5 seconds per estimation  
✅ **Affordable** - ~$0.00006 per request using Haiku 4.5  
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
│   │   └── context/examples.py          # Reference estimation examples
│   └── ui/
│       ├── streamlit_app.py             # Chat UI (streaming + structured)
│       ├── estimate_client.py           # HTTP client with safe error messages
│       └── view_models.py               # Pure presentation helpers
├── tests/                               # Offline pytest suite (+ legacy verify_api.py)
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
| Total hours | 8 to 500, and exactly the sum of the task hours |
| Hourly rate (in the result) | 30 to 150 |
| Hours per complexity | Simple 4-24, Medium 10-64, High 16-112 |
| `complexity` | `Simple`, `Medium` or `High` |

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

A conversational frontend, located at [`src/ui/streamlit_app.py`](src/ui/streamlit_app.py), that talks to the FastAPI backend. A **Response mode** radio in the sidebar picks one of two modes:

- **Streaming (markdown)**: calls `POST /api/v1/estimate/stream` and shows the answer token by token (SSE). Token and cost metrics are shown after each answer.
- **Structured (JSON)**: calls `POST /api/v1/estimate` and renders a summary, a task table (task, complexity, hours, cost) and the totals. A badge shows whether the answer was served from cache or freshly generated. No token or cost metrics are shown in this mode.

Both modes use an hourly rate of 40. Messages are kept in `st.session_state` (**New Chat** clears them); the backend is single-shot and does not receive the history as context. The API URL comes from the `API_BASE_URL` environment variable (default `http://localhost:8000`).

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

## Web UI (Rails)

A Rails 8 web UI lives in [`../estimator-web/`](../estimator-web/README.md). It calls the structured `POST /api/v1/estimate` endpoint (not the streaming one), shows the estimate (summary, task table, totals) and keeps a history of successful estimates in Postgres. The Streamlit chat above, with its streaming markdown mode, is still available.

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

The checks below are now enforced in the code: the schema validators (`schemas/estimation.py`) and the output guardrail run on every structured request. `tests/test_verification.py` and `tests/verify_api.py` are the older stand-alone scripts and still describe the previous response shape. For an end-to-end check of the current API see [VERIFICATION.md](VERIFICATION.md). The ranges listed here are the original guidance; the enforced values are in [Structured output and validators](#structured-output-and-validators).

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

- **Input tokens**: $3 per 1M tokens
- **Output tokens**: $15 per 1M tokens
- **Example cost**: ~$0.00005 per estimation request

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

- **Fast estimation**: ~3-5 seconds per request
- **Low cost**: ~$0.00005 per estimation (using Haiku 4.5)
- **Efficient**: Haiku 4.5 model optimized for speed and cost
- **Scalable**: Stateless API design supports horizontal scaling

## License

MIT

## Support

For issues, feature requests, or contributions, please open an issue or submit a pull request.

---

**Built with ❤️ using [Anthropic Claude API](https://docs.anthropic.com)**
