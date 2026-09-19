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

The project ships with two ways to use it:
- **FastAPI backend** (`/api/v1/estimate`, `/api/v1/estimate/stream`) — see [API Documentation](#api-documentation)
- **Streamlit chat UI** (`src/ui/streamlit_app.py`) — a conversational interface with real-time, token-by-token streaming; see [💬 Streamlit Chat Interface](#-streamlit-chat-interface)

## 📦 Complete Setup Summary

### ✅ What's Included

#### 1. **API Service** - `/api/v1/estimate`
Context-Augmented Generation endpoint that:
- Accepts meeting transcriptions as input
- Returns detailed project estimations with task breakdown
- Includes token usage and cost tracking
- Uses Haiku 4.5 model (fast, affordable, $0.00006 per request)

#### 2. **Configuration System**
- `config.py` - Pydantic BaseSettings
- Loads environment variables from `.env`
- Settings: ANTHROPIC_API_KEY, LLM_PROVIDER, LLM_MODEL, APP_ENV, LOG_LEVEL

#### 3. **Context Examples**
- Two complete reference estimation projects
- Includes detailed Hyrox workout tracking app example
- Used by LLM as benchmarks for better estimations

#### 4. **LLM Service Integration**
- CAG architecture pattern implementation
- System prompt + reference examples + user transcription
- Structured JSON output with full estimation details

#### 5. **FastAPI Endpoints**
- `POST /api/v1/estimate` - Main estimation endpoint
- `GET /health` - Service health check
- `GET /` - API information
- `/docs` - Interactive Swagger UI
- `/redoc` - ReDoc documentation

#### 6. **Streamlit Chat Interface** - `src/ui/streamlit_app.py`
A conversational, chat-app-style frontend for the estimation API:
- WhatsApp/iMessage-style bubbles: your messages on the right (sky blue), assistant replies on the left (gray)
- True real-time, token-by-token streaming — the LLM is prompted to respond in Markdown directly (not JSON), so every token can be shown to the user the instant it's generated
- "🤖 Thinking" animated indicator while waiting for the first token
- Sidebar dashboard: New Chat, Conversation Stats, Session Metrics (calls/tokens/cost), How to Use, and the CAG reference examples used as context

#### 7. **Verification Pipeline** ✅ *NEW*
Five-stage validation for manager confidence:
1. **Schema Validation** - Correct JSON structure
2. **Business Logic** - Mathematical accuracy (±15% tolerance)
3. **Reasonableness** - Values within realistic ranges
4. **Quality Check** - Meaningful descriptions and documentation
5. **Consistency** - Related fields logically align

**Status:** ✅ All 9 verification tests passing

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

**4. Verify Results:**
```bash
.venv/bin/python tests/verify_api.py
```

**5. Launch the Streamlit Chat UI:**

The UI's `API_BASE_URL` is hardcoded to `http://localhost:8000`, so make sure the API is reachable there first (e.g. `docker-compose up`, or `uvicorn cag_estimate.main:app --host 0.0.0.0 --port 8000`) — not the `8001` dev port used in step 1 above.

```bash
./run_streamlit.sh
# or manually:
.venv/bin/python -m streamlit run src/ui/streamlit_app.py
```
Then open http://localhost:8501 and chat with the estimation assistant in real time.

### 📊 Real Example: Grocery Price Comparison App (iOS)

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
✅ **Automated Verification Pipeline** - Managers validate estimates with confidence scores  
✅ **Cost Transparent** - See exact token costs for every request  
✅ **Fast Results** - 3-5 seconds per estimation  
✅ **Affordable** - ~$0.00006 per request using Haiku 4.5  
✅ **Production Ready** - CORS enabled, error handling, comprehensive logging  
✅ **Well Documented** - README, verification guide, interactive API docs  
✅ **Fully Tested** - 9 validation tests, API integration tests  

### 📁 Project Structure

```
cag-estimate/
├── src/
│   ├── cag_estimate/
│   │   ├── __init__.py
│   │   ├── config.py                    # Pydantic BaseSettings
│   │   ├── main.py                      # FastAPI app setup
│   │   ├── routers/estimations.py       # /api/v1/estimate + /estimate/stream
│   │   ├── services/llm_service.py      # LLM integration with CAG
│   │   └── context/examples.py          # Reference estimation examples
│   └── ui/
│       └── streamlit_app.py             # Real-time streaming chat UI
├── tests/
│   ├── test_verification.py             # Verification pipeline tests
│   └── verify_api.py                    # API integration tests
├── .streamlit/config.toml               # Streamlit UI configuration
├── run_streamlit.sh                     # Launches the chat UI
├── Dockerfile                           # API container image
├── docker-compose.yml                   # API service (port 8000)
├── docker-compose.override.yml          # Dev overrides (hot-reload)
├── README.md                            # This file
├── VERIFICATION.md                      # Verification pipeline guide
├── .env                                 # Environment variables (local)
├── .env.example                         # Environment template
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

## 💬 Streamlit Chat Interface

A conversational frontend, located at [`src/ui/streamlit_app.py`](src/ui/streamlit_app.py), that talks to the FastAPI backend and streams the assistant's response live — no more waiting for a spinner and then seeing the whole answer appear at once.

### How it works

```
User types a message
        ↓
Sky-blue bubble appears on the right, instantly
        ↓
"🤖 Thinking …" bubble appears on the left (animated dots)
        ↓
POST /api/v1/estimate/stream (Server-Sent Events)
        ↓
Model streams Markdown directly (not JSON) — each token is
already human-readable, so it can be shown the instant it arrives
        ↓
"Thinking" bubble is replaced, live, by the growing response text
        ↓
On completion: API metrics (tokens · cost) appended, sidebar updated
```

**Why Markdown instead of JSON for streaming?** The `/api/v1/estimate` endpoint (used for programmatic/API access) still returns structured JSON. But `/api/v1/estimate/stream` uses a different system prompt (`build_system_prompt(output_format="markdown")` in `routers/estimations.py`) that asks the model to respond directly in readable Markdown. This means the raw tokens streamed from Claude can be displayed to the user immediately, with no buffering or reformatting — true token-by-token real-time output, not a replay after the fact.

### Features

- **Real chat bubbles** — user messages on the right (sky blue), assistant replies on the left (light gray), like WhatsApp/iMessage
- **True real-time streaming** — placeholder + delta pattern re-renders the bubble on every token as it's generated
- **"Thinking" indicator** — animated dots shown the instant you hit send, replaced the moment the first token streams in
- **Auto-clearing input** — `st.chat_input()` clears itself after every message, so you can keep chatting
- **Sidebar dashboard** (in this order): New Chat button → Conversation Stats → Session Metrics (calls, tokens, cost, last call info) → How to Use → CAG reference examples used as context
- **XSS-safe rendering** — all message content is HTML-escaped before being converted from the small Markdown subset we use (bold, bullets, line breaks) into HTML, since bubbles are rendered with `unsafe_allow_html=True`

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
  "estimation": {
    "project_name": "string",
    "meeting_summary": "string",
    "tasks": [
      {
        "task_id": 1,
        "name": "string",
        "description": "string",
        "estimated_hours": 24,
        "estimated_cost_usd": 960,
        "complexity": "High",
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
  "model": "claude-haiku-4-5-20251001",
  "provider": "anthropic",
  "tokens_used": {
    "input_tokens": 3577,
    "output_tokens": 2295,
    "total_tokens": 5872
  },
  "cost_breakdown": {
    "input_cost_usd": 0.000011,
    "output_cost_usd": 0.000034,
    "total_cost_usd": 0.000045
  }
}
```

## Verification Pipeline

To ensure project estimates are accurate and reliable, implement this verification pipeline:

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

### Running Verification Tests

```bash
# Run all verification tests
pytest tests/ -v

# Run specific verification test
pytest tests/test_verification.py::test_estimation_schema -v

# Run with coverage
pytest tests/ --cov=cag_estimate
```

### Example Test

```python
def test_estimation_response_validation():
    """Verify estimation response structure and calculations."""
    response = estimate_project(
        transcription="your meeting transcription...",
        hourly_rate=40
    )
    
    # Schema validation
    assert "estimation" in response
    assert "model" in response
    assert "tokens_used" in response
    
    # Business logic validation
    estimation = response["estimation"]
    total_hours = sum(task["estimated_hours"] for task in estimation["tasks"])
    assert total_hours == estimation["summary"]["total_hours"]
    
    # Reasonableness check
    assert 8 <= estimation["summary"]["total_hours"] <= 500
    assert 30 <= estimation["summary"]["hourly_rate"] <= 150
    
    print("✓ All validations passed")
```

## Project Structure

```
cag-estimate/
├── src/
│   ├── cag_estimate/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app setup
│   │   ├── config.py              # Configuration (Pydantic BaseSettings)
│   │   ├── routers/
│   │   │   └── estimations.py    # Estimation endpoints (JSON + SSE stream)
│   │   ├── services/
│   │   │   └── llm_service.py    # LLM integration
│   │   └── context/
│   │       └── examples.py        # Reference estimation examples
│   └── ui/
│       └── streamlit_app.py       # Streamlit chat UI (real-time streaming)
├── tests/
│   ├── test_verification.py       # Verification pipeline tests
│   └── test_endpoints.py          # API endpoint tests
├── run_streamlit.sh                # Launches the chat UI
├── .env                           # Environment variables (local)
├── .env.example                   # Environment template
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required) | - | `sk-ant-...` |
| `LLM_PROVIDER` | LLM provider to use | `anthropic` | `anthropic` |
| `LLM_MODEL` | Claude model to use | `claude-haiku-4-5-20251001` | `claude-opus-4-1-20250805` |
| `APP_ENV` | Application environment | `development` | `production` |
| `LOG_LEVEL` | Logging level | `DEBUG` | `INFO`, `WARNING` |

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
# Install dev dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=cag_estimate
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
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
