# Session 7 pre-exercise: manual validation guide

Use this to verify the embedding pipeline yourself, step by step, with the test data and the results you should see. Every "expected" value below was observed on 2026-10-18 with a real OpenAI key. The whole guide costs well under one cent.

## Before you start

- Work from the **project directory**: `cag-estimate/cag-estimate/` (the folder that contains `docker-compose.yml`, `pyproject.toml` and `.env`).
- `.env` needs `OPENAI_API_KEY` (with credit) and `ANTHROPIC_API_KEY` (the app's settings require both).
- The stack must be running: `docker compose up -d api` (and `redis` if it is not up). After any dependency change use `docker compose up -d --build --no-deps api`.
- Do not paste `#` comments after a command in zsh: it passes them to the program as arguments.

## Test data

**Sample file:** `data/budgets_sample.json`, a bare JSON array.

| Fact | Value |
|---|---|
| Budgets / components | 15 / 52 |
| Components per budget | 001:4, 002:4, 003:4, 004:3, 005:4, 006:4, 007:3, 008:3, 009:4, 010:3, 011:3, 012:3, 013:3, 014:4, 015:3 |
| Sectors | finance 4, ecommerce 4, healthcare 4, industrial 3 |
| Complexity (components) | high 22, medium 18, low 12 |
| First chunk id | `BUD-2024-001::AUTH-001` |

**Text pairs for `compare.py`:**

| Pair | Text 1 | Text 2 |
|---|---|---|
| A (close) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application |
| B (unrelated) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime |
| C (generic) | Backend services | API development |

## Checklist

### V1. Offline tests and lint

```bash
PYTHONPATH=src uv run pytest -q
uv run ruff check .
```

Expected: `802 passed` (a higher number is fine if tests were added; there must be 0 failures) and `All checks passed!`.

### V2. Stack is up and the endpoint is visible

```bash
curl -s localhost:8000/health
curl -s localhost:8000/openapi.json | python3 -c "import sys,json;print([p for p in json.load(sys.stdin)['paths'] if 'embeddings' in p])"
```

Expected: `{"status":"ok","service":"CAG Estimate API"}` and `['/embeddings/ingest']`. Then open http://localhost:8000/docs: there is an **embeddings** tag with `POST /embeddings/ingest`. In Swagger you can use "Try it out" with a body shaped `{"budgets": [ ...the array from the file... ]}`.

### V3. Ingest the official sample

```bash
python3 -c 'import json; print(json.dumps({"budgets": json.load(open("data/budgets_sample.json"))}))' > /tmp/ingest_body.json
curl -s -X POST http://localhost:8000/embeddings/ingest -H "Content-Type: application/json" -d @/tmp/ingest_body.json > /tmp/ingest_out.json
python3 - <<'EOF'
import json
out = json.load(open("/tmp/ingest_out.json"))
chunks = out["chunks"]
print("stats:", out["stats"])
print("chunks:", len(chunks), "| unique ids:", len({c["chunk_id"] for c in chunks}), "| dims:", {len(c["embedding"]) for c in chunks})
print("first:", chunks[0]["chunk_id"], chunks[0]["token_count"], sorted(chunks[0]["metadata"]))
print("tokens min/max:", min(c["token_count"] for c in chunks), max(c["token_count"] for c in chunks))
EOF
```

Expected output:

```
stats: {'total_budgets': 15, 'total_chunks': 52, 'total_tokens': 4126, 'estimated_cost_usd': 8.252e-05}
chunks: 52 | unique ids: 52 | dims: {1536}
first: BUD-2024-001::AUTH-001 106 ['budget_id', 'client_sector', 'complexity', 'component_id', 'estimated_hours', 'main_technology', 'year']
tokens min/max: 66 106
```

Tolerance: `total_tokens` can differ by a few tokens if your `tiktoken` version differs (then the cost changes in the 7th decimal). Chunk count, ids and dimensions must match exactly. The first call after a restart takes a few extra seconds because `tiktoken` downloads its encoding file.

### V4. Chunk text format

Print the text of the first chunk: `python3 -c "import json;print(json.load(open('/tmp/ingest_out.json'))['chunks'][0]['text'])"`. Expected (exactly this, including the blank line):

```
[Project: Mobile banking API with OAuth 2.0 authentication and PSD2 compliance]
[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]

Component: OAuth 2.0 authentication backend
Description: Implementation of OAuth 2.0 flows (authorization code, refresh token) with JWT-based session management, multi-tenant token isolation, and rate limiting per client.
Tech stack: ruby_on_rails, postgresql, redis
Complexity: high
Estimated hours: 120
```

Metadata of that chunk (not embedded): `budget_id` BUD-2024-001, `component_id` AUTH-001, `client_sector` finance, `main_technology` ruby_on_rails, `year` 2024, `complexity` high, `estimated_hours` 120.

### V5. Cost arithmetic

`4126 tokens / 1,000,000 * $0.02 = $0.00008252`. The `estimated_cost_usd` from V3 must equal that (the price constant is in `embedder.py` and changes over time).

### V6. Endpoint error paths (no OpenAI call)

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/embeddings/ingest -H "Content-Type: application/json" -d '{"budgets":[]}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/embeddings/ingest -H "Content-Type: application/json" -d '{}'
```

Expected: `422` and `422`. For the empty list the body message is `Value error, send at least one budget to ingest`. A budget with an unknown sector (for example `gaming`) is also `422`; allowed sectors are finance, ecommerce, healthcare, industrial, and complexity is low, medium or high. An OpenAI failure returns a generic `500` (`The embeddings service failed. Please try again later.`) and never the OpenAI error text; that one is only easy to trigger without credit, and is covered by the automated tests.

### V7. `compare.py`: the three pairs

Outside the container:

```bash
PYTHONPATH=src uv run python scripts/compare.py --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" --text-b "Authorization service using JSON Web Tokens for a banking application"
PYTHONPATH=src uv run python scripts/compare.py --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" --text-b "Database migration from MySQL to PostgreSQL with zero downtime"
PYTHONPATH=src uv run python scripts/compare.py --text-a "Backend services" --text-b "API development"
```

Expected last line of each: `Cosine similarity: 0.5957` (A), `0.1920` (B), `0.5407` (C). Tolerance: within 0.005. The output has exactly three lines (`Text A: ...`, `Text B: ...`, `Cosine similarity: ...`).

Inside the container (run from the same directory), pair A should give the same `0.5957`:

```bash
docker compose exec api python scripts/compare.py --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" --text-b "Authorization service using JSON Web Tokens for a banking application"
```

Reading the results: B is far below A (a gap of about 0.40), so unrelated topics separate. A is just under the rough 0.6 expectation, and C (two generic phrases) scores almost as high as A: that is why the chunker prepends a context header. Identical texts should give `1.0000` (7 of 8 runs did; one run returned 0.9947 because OpenAI's API occasionally varies slightly; just rerun).

### V8. `compare.py` exit codes

| Command | Expected |
|---|---|
| valid call | exit `0`, three lines |
| `OPENAI_API_KEY=sk-invalid PYTHONPATH=src uv run python scripts/compare.py --text-a a --text-b b` | exit `1`, `error: embedding failed: AuthenticationError: Error code: 401 ...` |
| `OPENAI_API_KEY= PYTHONPATH=src uv run python scripts/compare.py --text-a a --text-b b` | exit `2`, `error: OPENAI_API_KEY is not configured` |
| `--text-a " " --text-b b` | exit `2`, `error: texts must not be empty or blank` |
| only `--text-a` | exit `2`, argparse usage message |

Check an exit code with `echo $?` right after the command.

### V9. Docker mounts

```bash
docker compose exec -T api ls /app/scripts /app/data
```

Expected: `compare.py` (plus a `__pycache__` folder) under `/app/scripts` and `budgets_sample.json` under `/app/data`.

### V10. Deliverable checklist (from the brief)

- [ ] Branch is `session-07/pre-exercise`: `git branch --show-current`.
- [ ] Module with the five files: `ls src/cag_estimate/embedding_pipeline/` shows `__init__.py`, `chunker.py`, `embedder.py`, `schemas.py`, `router.py` (plus `service.py` and `SANITY_CHECK.md`).
- [ ] `scripts/compare.py` exists and is executable: `ls -l scripts/compare.py`.
- [ ] `POST /embeddings/ingest` is registered and visible in `/docs` (V2).
- [ ] `SANITY_CHECK.md` has the three values (V7) and a short comment.
- [ ] README has the "Embedding pipeline" section with ingest, `compare.py` inside and outside the container.
- [ ] `tiktoken` is declared: `rg tiktoken pyproject.toml`.
- [ ] `git status` shows nothing uncommitted for the deliverable files (the untracked notes in `odd/`, `.atl/` and the practice files are not part of it).

### V11. Submit

Push the branch, check the URL opens (public repo, or give the reviewer access), and email the full branch URL to george@lidr.co at the latest two days before the live session.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `429 ... You have no credits remaining` (`credit_balance_exhausted`) | The OpenAI account has no credit. It fails immediately by design (no retries for this error). Add credit. |
| `ModuleNotFoundError: cag_estimate` | Plain `uv run` can lose the editable install on macOS. Use `PYTHONPATH=src uv run ...`. |
| `docker compose` says no configuration file | Run it from `cag-estimate/cag-estimate/`. |
| Endpoint missing in `/docs` or the container crashes after a dependency change | Rebuild only the API image: `docker compose up -d --build --no-deps api`. |
| First ingest is slow (about 3-4 s) | `tiktoken` downloads its encoding file on first use; later calls are fast. |
| `compare.py` fails with a pydantic validation error | `.env` lacks `ANTHROPIC_API_KEY` (the settings require it). |
