# Session 8: manual validation guide (persistence and semantic search)

Use this to verify the pgvector persistence and `/search` yourself, step by step, with the test data and the results you should see. The expected values were observed during the session 8 build with a real OpenAI key. The whole guide costs well under one cent. It supersedes V3 to V6 of `session7/MANUAL_VALIDATION.md`.

## Before you start

- Work from the **project directory**: `cag-estimate/cag-estimate/` (the folder that contains `docker-compose.yml`, `pyproject.toml` and `.env`).
- `.env` needs `OPENAI_API_KEY` (with credit) and `ANTHROPIC_API_KEY` (the app's settings require both). `DATABASE_URL` is optional: the compose file sets it for the `api` container.
- Start the stack: `docker compose up -d pgvector redis api`. After a dependency change use `docker compose up -d --build --no-deps api`.
- Do not paste `#` comments after a command in zsh: it passes them to the program as arguments.

## Test data

**Sample file:** `data/budgets_sample.json`, a bare JSON array of 15 budgets with 52 components in total (see `session7/MANUAL_VALIDATION.md`). The first budget is `BUD-2024-001` (4 components, finance, 2024).

**Text pairs for `compare.py`:**

| Pair | Text 1 | Text 2 | Cosine similarity |
|---|---|---|---|
| A (close) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | 0.5957 |
| B (unrelated) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | 0.1920 |
| C (generic) | Backend services | API development | 0.5407 |

## Checklist

### V1. Offline tests and lint

```bash
PYTHONPATH=src uv run pytest -q -m "not integration"
uv run ruff check .
```

Expected: 0 failures and `All checks passed!`. The full suite (offline plus live) is 921 passed (see V8).

### V2. pgvector is healthy

```bash
docker compose ps pgvector
docker compose exec pgvector psql -U estimator -d estimator -c "SELECT version()"
docker compose exec pgvector psql -U estimator -d estimator -c "\dx"
```

Expected: `pgvector` is `healthy`; the version line starts with `PostgreSQL 16.15`. `\dx` lists `vector` (0.8.7) after V3 has run the migration (before it, only `plpgsql`).

### V3. Migration and schema

```bash
docker compose run --rm --no-deps api alembic upgrade head
docker compose run --rm --no-deps api alembic current
docker compose exec pgvector psql -U estimator -d estimator -c "\d documents"
docker compose exec pgvector psql -U estimator -d estimator -c "\d chunks"
docker compose exec pgvector psql -U estimator -d estimator -c "\di"
docker compose exec pgvector psql -U estimator -d estimator -c "select count(*) from pg_indexes where indexdef ilike '%hnsw%' or indexdef ilike '%ivfflat%'"
```

Expected:

- `alembic current` shows `0001 (head)`.
- `documents`: `id` bigint, `source_path` text, `document_type` varchar(50), `ingested_at` timestamptz default `now()`, `metadata` jsonb default `'{}'`.
- `chunks`: `id`, `document_id` (FK to `documents` with `ON DELETE CASCADE`), `chunk_type`, `content`, `embedding vector(1536)` (nullable), `metadata` jsonb, `created_at`.
- `\di` lists `ix_documents_source_path`, `ix_chunks_document_id`, `ix_chunks_chunk_type` and `ix_chunks_metadata_gin` (type `gin`).
- The last query prints `0`: there is NO vector index (no hnsw, no ivfflat).

Round trip (this erases the data of the two tables):

```bash
docker compose run --rm --no-deps api alembic downgrade base
docker compose run --rm --no-deps api alembic upgrade head
```

Expected: after the downgrade `\d documents` says the relation does not exist while `\dx` still lists `vector`; after the upgrade the schema is back.

### V4. Ingest one budget

Build the body from the first budget of the sample file and send it:

```bash
python3 -c 'import json; print(json.dumps({"source_path": "manual::BUD-2024-001", "document_type": "historical_budget", "content": json.load(open("data/budgets_sample.json"))[0]}))' > /tmp/ingest_one.json
curl -s -X POST http://localhost:8000/embeddings/ingest -H "Content-Type: application/json" -d @/tmp/ingest_one.json
```

Expected `200` with `chunks_created: 4` and `embedding_dimension: 1536` (`document_id` depends on your data; `ingestion_time_ms` was between about 300 and 5,600 ms).

Repeat the same curl. Expected `409` with the literal body `{"detail":"Document already ingested","document_id":N}` where `N` is the first `document_id`.

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/embeddings/ingest -H "Content-Type: application/json" -d '{}'
```

Expected: `422`.

Check the rows:

```bash
docker compose exec pgvector psql -U estimator -d estimator -c "select id, source_path, metadata from documents where source_path like 'manual::%'"
docker compose exec pgvector psql -U estimator -d estimator -c "select count(*), count(embedding), vector_dims(min(embedding)) from chunks c join documents d on d.id = c.document_id where d.source_path like 'manual::%'"
```

Expected: one document whose metadata is `{"year": 2024, "budget_id": "BUD-2024-001", "client_sector": "finance"}`; 4 chunks, 4 non-null embeddings, 1536 dimensions.

Clean up (the cascade removes the chunks):

```bash
docker compose exec pgvector psql -U estimator -d estimator -c "delete from documents where source_path like 'manual::%'"
```

Expected: `DELETE 1`, and the chunk count query above then returns 0 rows of data.

### V5. Ingest the full sample and run the examples

```bash
docker compose exec api python scripts/query_examples.py
```

Expected on an empty database: first line `Ingested 15, already present 0`, then 5 queries with 5 results each. On a re-run: `Ingested 0, already present 15` (idempotent via the `409`). Exit code `0`.

```bash
docker compose exec pgvector psql -U estimator -d estimator -c "select (select count(*) from documents) as documents, (select count(*) from chunks) as chunks, (select count(*) from chunks where embedding is null) as null_embeddings"
```

Expected: `15 | 52 | 0`.

Compare your output with `output_examples.txt` at the project root. Distances vary slightly between runs; tolerance 0.01. Reference values from that file:

| Query | Best distance | What to see |
|---|---|---|
| OAuth 2.0 authentication with JWT tokens | 0.5133 | top 4 hits are the OAuth mobile banking project |
| payment processing with Stripe | 0.5848 | real-time payments gateway first |
| restaurant reservations (out of domain) | 0.7190 | booking-like components (grocery slot booking, patient appointments) |
| high-complexity backend components | 0.5304 | see the file |
| real-time dashboard | 0.4987 | see the file |

### V6. Search through curl

```bash
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "OAuth 2.0 authentication with JWT tokens", "k": 3}'
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "restaurant reservations", "k": 3}'
```

Expected: `200` with `query`, `k: 3`, `search_time_ms` (about 170 to 230 ms) and 3 results ordered by ascending `distance`. Each result has `chunk_id`, `document_id`, `chunk_type`, `content`, `distance` and `metadata`. The OAuth query has a best distance of about 0.51; "restaurant reservations" about 0.72, clearly larger.

Invalid inputs (no OpenAI call, each prints `422`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "x", "k": 0}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "x", "k": 51}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "", "k": 3}'
```

On an empty corpus (after `alembic downgrade base` plus `upgrade head`) a valid search returns `200` with `"results": []`.

### V7. Error paths without an OpenAI call

The 422 cases in V4 and V6 need no OpenAI call. The other paths are covered by the automated tests and are not easy to trigger by hand: `502` on search when the embeddings API fails, `500` (generic body) on ingest when it fails, and `503` when the database or the key is not configured.

### V8. Transactions and live integration tests

The rollback behaviour (an embeddings failure leaves no orphan document or chunks), the cascade delete, the migration/model match (`alembic check`), the cosine ordering and the NULL-embedding rule are covered by the live tests in `tests/integration/`: `test_pgvector_store.py`, `test_rag_ingest_service.py` and `test_semantic_retriever.py`.

One-time setup of the test database, then run them:

```bash
docker compose exec pgvector psql -U estimator -d estimator -c "CREATE DATABASE estimator_test"
PYTHONPATH=src uv run pytest -q -m integration
PYTHONPATH=src uv run pytest -q
```

Expected: 16 integration tests pass (they skip if the database is unreachable); the full run is `921 passed`. A higher number is fine if tests were added; there must be 0 failures.

### V9. `compare.py` still works

```bash
PYTHONPATH=src uv run python scripts/compare.py --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" --text-b "Authorization service using JSON Web Tokens for a banking application"
```

Expected last line: `Cosine similarity: 0.5957` (pair A). Pairs B and C give `0.1920` and `0.5407`. Tolerance 0.005.

### V10. Deliverable checklist (from the brief)

- [ ] Branch is `session-08`: `git branch --show-current`.
- [ ] Migration file exists: `alembic/versions/0001_initial_schema.py`.
- [ ] Tables `documents` and `chunks` exist (V3).
- [ ] GIN index on `chunks.metadata` exists and there is no vector index (V3).
- [ ] `POST /embeddings/ingest` persists one budget and answers `409` on a duplicate `source_path` (V4).
- [ ] `POST /search` returns the top k by cosine distance (V6).
- [ ] `scripts/query_examples.py` exists and runs (V5).
- [ ] `output_examples.txt` is committed with the real output.
- [ ] README has the "Persistence and semantic search" section with the four justifications (two tables, JSONB metadata, cosine distance, no vector index yet).
- [ ] `git status` shows nothing uncommitted for the deliverable files (the untracked notes in `odd/`, `.atl/` and the practice files are not part of it).

### V11. Submit

Push the branch `session-08`, check the URL opens (public repo, or give the reviewer access), and email the full branch URL to george@lidr.co.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `429 ... You have no credits remaining` (`insufficient_quota`) | The OpenAI account has no credit. Ingest returns `500` and search returns `502`. Add credit. |
| `address already in use` on port 5433 | Another process uses host port 5433. Stop it or remap the `pgvector` port in a compose override. |
| `alembic` cannot find `DATABASE_URL` | Run it through compose (`docker compose run --rm --no-deps api alembic ...`) so the variable is set, or export `DATABASE_URL` when running on the host (host URL: `postgresql+asyncpg://estimator:estimator@localhost:5433/estimator`). |
| `api` not healthy for a few seconds right after a rebuild | Normal start-up. Wait and retry. |
| `ModuleNotFoundError: cag_estimate` | Plain `uv run` can lose the editable install on macOS. Use `PYTHONPATH=src uv run ...`. |
| `docker compose` says no configuration file | Run it from `cag-estimate/cag-estimate/`. |
| `503` from `/search` or ingest | The database or `OPENAI_API_KEY` is not configured, or `pgvector` is down: check `docker compose ps`. |
| Live integration tests are skipped | The `estimator_test` database does not exist or the container is down (V8). |
