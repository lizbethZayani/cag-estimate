# Session 10: manual validation guide (hybrid search and reranking)

Use this to reproduce the session 10 setup and measure the four retrieval configurations yourself. Steps run in order. Search requests call the OpenAI embeddings API (a fraction of a cent for the whole guide); the reranker runs locally.

## Before you start

- Work from the **project directory**: `cag-estimate/cag-estimate/` (the folder with `docker-compose.yml`, `pyproject.toml` and `.env`).
- `.env` needs `OPENAI_API_KEY` (with credit) and `ANTHROPIC_API_KEY` (the settings require both).
- Do not paste `#` comments after a command in zsh.
- The configurations: A vector, B hybrid, C vector + rerank, D hybrid + rerank.

## 1. Rebuild and start the stack

The image now includes `sentence-transformers` (torch), so rebuild it. The build is slow and the image is much heavier.

```bash
docker compose up -d --build pgvector redis api
docker compose ps
```

Expected: `pgvector`, `redis` and `api` are `healthy` (the `api` may need a few seconds).

## 2. Apply the migration

```bash
docker compose run --rm --no-deps api alembic upgrade head
docker compose run --rm --no-deps api alembic current
docker compose exec pgvector psql -U estimator -d estimator -c "\d chunks"
docker compose exec pgvector psql -U estimator -d estimator -c "select count(*) as chunks, count(content_tsv) as with_tsv from chunks"
```

Expected: `alembic current` shows `0002 (head)`; `\d chunks` lists `content_tsv` (`tsvector`, generated always as `to_tsvector('english', content)` stored) and the index `ix_chunks_content_tsv` (gin); `chunks` and `with_tsv` are equal (every existing chunk is populated, no re-ingest needed).

## 3. Confirm the corpus

If the database is empty, ingest first: `docker compose exec api python scripts/query_examples.py`.

```bash
docker compose exec pgvector psql -U estimator -d estimator -c "select (select count(*) from documents) as documents, (select count(*) from chunks) as chunks, (select count(*) from chunks where embedding is null) as null_embeddings"
```

Expected: `15 | 52 | 0`.

## 4. Verify the reranker

```bash
docker compose exec api python scripts/verify_reranker.py
```

The first run downloads the model (about 450 MB) into the `hf-cache` volume. Expected: three lines (`model=...`, `relevant=...`, `irrelevant=...`) with `relevant` higher than `irrelevant`, exit code `0`. Check it with `echo $?` right after. Exit `1` means the model failed to load or score; exit `2` means the ranking is wrong.

## 5. Smoke test the four configurations

Run each and check `search_mode` and `reranked` in the response (use `| jq '{search_mode, reranked, search_time_ms}'` if you have `jq`):

```bash
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "online store with card checkout", "k": 5, "search_mode": "vector", "rerank": false}'
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "online store with card checkout", "k": 5, "search_mode": "hybrid", "rerank": false}'
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "online store with card checkout", "k": 5, "search_mode": "vector", "rerank": true}'
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "online store with card checkout", "k": 5, "search_mode": "hybrid", "rerank": true}'
```

| Config | Expected `search_mode` | Expected `reranked` | Expected `score` |
|---|---|---|---|
| A | `vector` | `false` | `null` |
| B | `hybrid` | `false` | RRF score |
| C | `vector` | `true` | rerank score |
| D | `hybrid` | `true` | rerank score |

If C or D answer `reranked: false`, the reranker failed soft: look at `docker compose logs --tail=100 api` for `rerank_failed` and fix it before measuring. The first reranked request is slow (model load).

## 6. Run the measurement

From the project directory, against `http://localhost:8000`:

```bash
PYTHONPATH=src uv run python scripts/eval_retrieval.py --output session10/eval_raw.json
```

- It runs the 5 golden queries (`src/cag_estimate/evals/retrieval_golden_set.json`) under A, B, C and D.
- For every query and configuration it sends one **warm-up** request that is discarded (it absorbs the one-off model load), then `--runs` timed requests (default 3). Other options: `--k` (default 5), `--base-url` (or env `API_BASE_URL`), `--golden`.
- It **aborts** with exit code `1` if the API answers a different `search_mode` or `reranked` than requested, or any request fails (so a silently failed reranker cannot produce a misleading table). Exit `2` is bad usage or an unreadable golden set.
- `--output` writes every raw measurement as JSON.
- It prints `queries=... runs=... k=...`, a comparison table and a per-query precision table.

## 7. Record the results

Paste the printed tables into `session10/RESULTS.md` (sections "Results" and "Per-query results"), then fill in the conclusions.

## If something fails

| Symptom | Cause and fix |
|---|---|
| `verify_reranker.py` exits `1`, or the model download fails | No network from the container, or the `hf-cache` volume is not writable. Check `docker compose logs api` and retry; the model is about 450 MB. |
| `502` from `/search` | The embeddings API failed (no credit or key). See the session 8 troubleshooting table. |
| `503` from `/search` | Database or `OPENAI_API_KEY` not configured, or `pgvector` is down: `docker compose ps`. |
| `eval_retrieval.py` aborts: "asked for ... reranked=True but the API applied ... reranked=False" | The reranker failed soft. Fix step 4 first, then rerun. |
| Hybrid search errors or `content_tsv` does not exist | The migration was not applied: run step 2. |
| `ModuleNotFoundError: cag_estimate` on the host | Use `PYTHONPATH=src uv run ...`. |
| `verify_reranker.py` fails with `Permission denied: /home/appuser/.cache/huggingface/hub` | The `hf-cache` volume was created root-owned (before the Dockerfile created the directory for `appuser`). Rebuild the image, then recreate the empty volume: `docker compose rm -sf api && docker volume rm cag-estimate-hf-cache && docker compose up -d api`. |
| `redis` fails with `Bind for 0.0.0.0:6379 failed: port is already allocated` | Another container (for example `cag-estimate-redis-test`) already owns the host port. `/search` does not need Redis: start only the API with `docker compose up -d --no-deps api`. |

## Automated tests

The integration tests run against the `estimator_test` database (create it once as in the session 8 guide, V8):

```bash
PYTHONPATH=src uv run pytest -q -m integration
```
