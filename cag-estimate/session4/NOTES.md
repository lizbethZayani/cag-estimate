# Session 4 notes

What was added to close the Session 4 gap against the reference repo (`ai-engineering@session_4_live`, `estimator/`), and where this project deliberately differs. See the README for how each piece works.

## Added

| Area | Where |
|------|-------|
| Versioned prompt templates (Jinja2) | `src/cag_estimate/prompts/` |
| Structured extraction with validators and retries | `schemas/estimation.py`, `LLMWrapper.complete_structured()` |
| `EstimationService` pipeline and thin router | `services/estimation.py`, `routers/estimations.py`, `dependencies.py` |
| Input and output guardrails | `guardrails/` |
| Semantic cache on Redis Stack | `cache/semantic.py`, `docker-compose.yml` |
| Conversation UI with streaming and structured modes | `src/ui/` |
| structlog level from `LOG_LEVEL` | `logging_config.py`, `main.py` |
| Offline tests (shared fixture in `tests/conftest.py`, pipeline-order test) | `tests/` |

## Deliberate deviations

- **Domain:** this project estimates hours, tasks and an hourly rate (`ProjectEstimation`), not the reference's EUR/phases schema.
- **One cache threshold:** a single `SEMANTIC_CACHE_THRESHOLD` (default `0.90`) instead of per-case thresholds; `SEMANTIC_CACHE_LOG_ONLY` is there to calibrate it.
- **Router fallback kept:** structured calls go through the existing LiteLLM Router (via Instructor), so the primary-to-fallback model behaviour still applies.
- **Input guardrail before the cache:** a rejected input is never served from either cache and never reaches the LLM.
- **Prompt-independent cache keys:** keys use transcription, hourly rate, prompt version and model, not the rendered prompt, so wording edits within a version keep entries; a new prompt version is a new key and a new semantic bucket.
- **Streaming stays Markdown:** the stream endpoint has the input guardrail but no structured validation, no output guardrail and no cache, to keep the token-by-token UI. A rejected input there is one SSE `error` event on HTTP 200 (with `reason`) rather than a 400, because the UI calls `raise_for_status()`.
- **Hours validation:** exact sum equality between task hours and `total_hours` (the old test-only validator allowed 15%); the complexity ranges are Simple 4-24, Medium 10-64, High 16-112.
- **`.env.example` not updated:** `EMBEDDING_MODEL` and `SEMANTIC_CACHE_*` are documented in the README and `DOCKER.md` instead; `docker-compose.yml` passes them through to the `api` container.

## Rails UI port (`estimator-web/`)

The Rails 8 UI from the reference repo (`ai-engineering@session_4_live`, `estimator-web/`) was ported to the repo root as `estimator-web/` and adapted to this project's API. See [`../../estimator-web/README.md`](../../estimator-web/README.md).

- **Contract:** the request is `{transcription, hourly_rate}` and the result is `ProjectEstimation` with tasks in hours and USD (not the reference's EUR/phases/weeks schema). Models were renamed and reshaped accordingly (`Task`, `EstimationSummary`, `EstimationResult`, `EstimationResponse`).
- **400 body:** the guardrail `reason` and `message` are top-level in our API, not nested under `detail` as in the reference; the client reads them from the top level. 422 carries a `detail` list and 502 a generic message.
- **Neutral branding:** "CAG Estimate" naming; the reference's LIDR logo and favicon were not copied.
- **English copy:** all UI text is in English.
- **Docker-only runtime:** `estimator-web` and a `postgres` service were added to `docker-compose.yml` (Ruby 3.4.4 image; the host only has an older system Ruby). The UI reaches the API at `http://api:8000`.
- **Streamlit kept:** the Streamlit chat and the streaming endpoint are unchanged; the Rails UI uses only the structured endpoint.

## Known gaps

- The semantic cache was verified only against a faked RediSearch index; real Redis Stack behaviour is checked manually (see `VERIFICATION.md`).
- The Streamlit UI was tested with Streamlit's `AppTest` and patched HTTP, not against a live backend.
- The backend is single-shot: the chat history is not sent to the model as context.
- `tests/verify_api.py` still reads the previous response shape and no longer works against the current API.
