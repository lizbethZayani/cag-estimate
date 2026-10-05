# Estimator Web (Rails UI)

A Rails web UI for CAG Estimate. It sends a meeting transcription and an hourly rate to the Python API (`POST /api/v1/estimate`), shows the structured estimate (project summary, task table, totals) and keeps a history of successful estimates in Postgres.

Stack: Rails 8.0.5, Ruby 3.4.4, PostgreSQL, Hotwire (Turbo + Stimulus), importmap, Propshaft, Tailwind CSS 4 (standalone, via `tailwindcss-rails`), Faraday, Minitest + WebMock.

Everything runs in Docker. No Ruby is needed on the host.

## Architecture

```
Browser -> EstimationsController -> EstimatorAi::Client -> POST /api/v1/estimate (Python API)
                  |
                  +-> Estimation (ActiveRecord, `estimations` table in Postgres)
```

| Piece | Role |
|-------|------|
| `EstimationsController` | `new`, `create`, `show`, `index`. Validates the form, calls the client, persists the successful response, maps errors to UI states. |
| `EstimatorAi::Client` (`app/services/estimator_ai/client.rb`) | Faraday client. Posts the payload and raises typed errors per HTTP status. |
| `EstimationRequest` | Form model (ActiveModel). `transcription` 20..80000 characters, `hourly_rate` integer 30..150 (default 40). |
| `Estimation` | ActiveRecord history row: transcription, hourly rate, raw `response_payload` (jsonb), `prompt_version`, `cached`. |
| `EstimationResponse` | Wraps the stored or received payload: `result`, `prompt_version`, `cached`. |
| `EstimationResult` | `project_name`, `meeting_summary`, `tasks`, `summary`. |
| `Task` | `task_id`, `name`, `description`, `estimated_hours`, `estimated_cost_usd`, `complexity`, `includes`. |
| `EstimationSummary` | `total_hours`, `total_cost_usd`, `team_size`, `estimated_duration_weeks`, `hourly_rate`, `assumptions`. |

Routes (`config/routes.rb`): `root` is `estimations#new`; `resources :estimations` (`index`, `new`, `create`, `show`); `GET /up` is the Rails health check.

## API contract it consumes

Request:

```json
{ "transcription": "...", "hourly_rate": 40 }
```

Responses and how the UI handles them:

| API response | Client behavior | UI state |
|--------------|-----------------|----------|
| `200` `{ "result": { project_name, meeting_summary, tasks[], summary }, "prompt_version": "v1", "cached": false }` | Returns the parsed body | The estimate is saved to `estimations` and the user is redirected to its show page. |
| `400` top-level `{ "reason", "message" }` with `reason` in `moderation`, `prompt_injection`, `pii` (not nested under `detail`) | `GuardrailViolation` carrying the API `message` | Form re-rendered with HTTP 422 and the API message as a flash alert. |
| `400` with any other body | `InvalidRequest` ("The API rejected the request") | Same as above. |
| `422` `{ "detail": [ { "loc", "msg", ... } ] }` | `InvalidRequest`, message built from `detail` | Form re-rendered with HTTP 422 and the message as a flash alert. |
| `502` | `ServerError` | Form re-rendered with HTTP 503 and the flash "AI service unavailable. Please try again in a moment." |
| Other status, connection failure or timeout | `ServerError` / `Faraday::ConnectionFailed` / `Faraday::TimeoutError` | Same 503 flash. The detail is only written to the Rails log. |

Local form validation failures (transcription length, hourly rate range) also re-render the form with HTTP 422, without calling the API. Only successful estimates are persisted.

The UI uses the structured endpoint only. It does not use `/api/v1/estimate/stream`.

## Environment variables

| Variable | Used by | Default |
|----------|---------|---------|
| `ESTIMATOR_API_BASE_URL` | `config/initializers/estimator_ai.rb` | `http://localhost:8000` (compose sets `http://api:8000`) |
| `ESTIMATOR_AI_TIMEOUT` | `config/initializers/estimator_ai.rb` (seconds) | `180` |
| `DATABASE_HOST` | `config/database.yml` | unset (compose sets `postgres`) |
| `DATABASE_PORT` | `config/database.yml` | `5432` |
| `DATABASE_USER` | `config/database.yml` | unset (compose sets `postgres`) |
| `DATABASE_PASSWORD` | `config/database.yml` | unset (compose sets `postgres`) |

`.env.example` lists these with the Docker values. The databases are `estimator_web_development` and `estimator_web_test`.

## Run

From the `cag-estimate/` directory (the one with `docker-compose.yml`). The API needs `ANTHROPIC_API_KEY` in `cag-estimate/.env` (see `DOCKER.md`).

```bash
docker compose up -d --build estimator-web postgres api redis
```

- Rails UI: http://localhost:3000
- Python API: http://localhost:8000
- Postgres is not published to the host.

The `estimator-web` container runs `bin/rails db:prepare && bin/dev` (Rails server plus the Tailwind watcher from `Procfile.dev`). Source is bind-mounted from `estimator-web/`, so edits are picked up without a rebuild.

## Tests and lint

```bash
docker compose exec estimator-web bin/rails test
docker compose exec estimator-web bin/rubocop
docker compose exec estimator-web bin/brakeman
```

If the container is not running, use `docker compose run --rm estimator-web bin/rails test`. HTTP calls to the API are stubbed with WebMock; no real API or LLM is needed.

## Stimulus controllers

| Controller | File | What it does |
|------------|------|--------------|
| `transcription-upload` | `app/javascript/controllers/transcription_upload_controller.js` | Reads a `.txt` file in the browser (FileReader) and puts its text in the textarea so it can be edited before submitting. Rejects files over 200000 bytes with an inline message. No multipart upload. |
| `form-loading` | `app/javascript/controllers/form_loading_controller.js` | On submit, disables the button, shows a spinner and a status panel with an elapsed timer (mm:ss) and a rolling phase message (changes at 0, 2, 8, 30 and 90 seconds). |

## Project layout

```
estimator-web/
├── app/
│   ├── controllers/estimations_controller.rb
│   ├── models/                  # estimation, estimation_request, estimation_response,
│   │                            # estimation_result, estimation_summary, task
│   ├── services/estimator_ai/client.rb
│   ├── views/estimations/       # new, _form, show, index
│   └── javascript/controllers/  # Stimulus controllers
├── config/
│   ├── database.yml
│   ├── importmap.rb
│   ├── routes.rb
│   └── initializers/estimator_ai.rb
├── db/                          # migration and schema for `estimations`
├── test/                        # controllers, models, services (Minitest + WebMock)
├── Dockerfile                   # ruby:3.4.4-slim
├── Procfile.dev                 # web + Tailwind watcher
└── .env.example
```

## Known notes and limitations

- Rails development mode does not reload initializers. After changing anything in `config/initializers/*` or an environment variable, restart the container (`docker compose restart estimator-web`).
- Host Ruby is not required; the container is the only supported runtime.
- The Gemfile still lists production-oriented gems from the reference project (Kamal, Thruster, Solid Cache/Queue/Cable). Production deployment is out of scope.
- History lives in the Postgres `estimations` table (volume `estimator-web-pg-data`). Only successful estimates are stored; rejected or failed requests are not.
- The history page shows the 20 most recent estimations.
- The Streamlit chat UI in the Python project is still available and independent of this app.
