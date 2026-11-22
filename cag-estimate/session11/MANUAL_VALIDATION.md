# Session 11: manual validation guide (cited RAG estimates and RAGAS)

Use this to reproduce the session 11 work and measure the RAGAS baseline yourself. Steps run in order. The offline demo costs nothing; the endpoint and the evaluation call OpenAI.

## Before you start

- Work from the **project directory**: `cag-estimate/cag-estimate/` (the folder with `docker-compose.yml`, `pyproject.toml` and `.env`).
- `.env` needs `OPENAI_API_KEY` (with credit) and `ANTHROPIC_API_KEY` (the settings require both).
- Do not paste `#` comments after a command in zsh.

## 1. Prerequisites

- The `pgvector` container is running with the corpus from sessions 8 to 10: **15 documents / 52 chunks** (see step 3 of the session 10 guide to check or re-ingest).
- Install the evaluation dependencies (the `evals` group, not part of the Docker image):

```bash
uv sync --group evals
```

The first import of freshly installed packages can be slow on a nearly full disk (macOS may have to fetch evicted files); wait it out rather than interrupting.

## 2. Run the offline demo

No network, database or key needed:

```bash
PYTHONPATH=src uv run python scripts/demo_verify_citations.py; echo exit=$?
```

Expected: a table with three lines, one `grounded`, one `dangling` (it cites the invented `BUD-2099-999::FAKE-001`) and one `insufficient`; a summary line `lines=3 grounded=1 dangling=1 insufficient=1`; the sentence `DANGLING CITATION DETECTED`; `exit=0`. Exit `1` would mean the verifier missed the planted id.

## 3. Optional: call the endpoint

The image bakes the code, so rebuild the `api` before calling the new endpoint:

```bash
docker compose up -d --build pgvector redis api
docker compose ps
```

```bash
curl -s -X POST http://localhost:8000/api/v1/estimate/rag -H "Content-Type: application/json" -H "X-Request-ID: demo-1" -d '{"query": "Online store with product catalog, cart and card checkout"}' | jq '{prompt_version, retrieved_chunk_ids, citation_report}'
docker compose logs --tail=50 api
```

Expected: `200` with `estimate`, `citation_report`, `prompt_version` and `retrieved_chunk_ids`; a `citation_report` log line with `request_id=demo-1`. Add `-i` to see the echoed `X-Request-ID` header. Every cited `chunk_id` in the estimate must appear in `retrieved_chunk_ids`.

## 4. Run the RAGAS evaluation

From the project directory, against the running pgvector (on the host, not in the container):

```bash
DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator PYTHONPATH=src uv run --group evals python scripts/eval_ragas.py
```

- It generates a cited estimate for each of the 5 golden queries through the real service, then RAGAS scores them with the judge `gpt-4o-mini` and `text-embedding-3-small`.
- It writes `session11/ragas_samples.json` (answers, contexts, ground truths, citation reports) and `session11/ragas_raw.json` (raw scores and averages), then prints a RAGAS table and a citation table.
- **Cost:** small. 5 generations plus the judge calls on `gpt-4o-mini`.
- Options: `--limit N` (first N queries), `--judge-model`, `--embedding-model`, `--golden`, `--samples`, `--out`.
- `--score-only` re-scores the saved `--samples` without generating again (use it after a judge failure or to change the judge model).
- Exit codes: `0` ok, `1` at least one query was rejected (scored `n/a`), `2` bad usage, input or configuration.

## 5. Record the results

Paste the two printed tables into `session11/RESULTS.md` (sections "Results" and "Citation verification report") and fill in the last section.

## If something fails

| Symptom | Cause and fix |
|---|---|
| `eval_ragas.py` exits `2` with "the RAG estimation service is not configured" | `DATABASE_URL` or `OPENAI_API_KEY` is missing. Set both (see step 4) and keep `pgvector` running. |
| Exit `2` with "OPENAI_API_KEY is not set" | The judge needs the key: add it to `.env`. |
| Exit `1`, a query row shows `rejected` and `n/a` | The service rejected that query (dangling citations twice, invalid output or a generation error). Check the citation table and the logs, then re-run. |
| A single metric prints `n/a` | That metric failed (a `ragas_metric_failed` log line names it and the error class). Fix the cause and run again with `--score-only`. |
| `502` `citation_verification_failed` from the endpoint | The model cited ids that were not retrieved, twice. Retry; the log has `citations_rejected` with the ids. |
| `503` `rag_not_configured` | `OPENAI_API_KEY` or `DATABASE_URL` is missing in the container: check `docker compose config`. |
| `ModuleNotFoundError: cag_estimate` or `ragas` | Use `PYTHONPATH=src uv run --group evals ...` and run `uv sync --group evals`. |
| Very slow first run | Cold imports on a nearly full disk, or the first OpenAI calls. Wait; do not interrupt. |

## Automated tests

Offline suite (no OpenAI calls):

```bash
PYTHONPATH=src uv run pytest -q -m "not integration"
```

The integration tests need the `estimator_test` database (create it once as in the session 8 guide):

```bash
PYTHONPATH=src uv run pytest -q -m integration
```
