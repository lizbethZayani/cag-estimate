# Sanity check: do the embeddings discriminate?

Run date: 2026-10-18. Model: `text-embedding-3-small`, default dimension (1536). Tool: `scripts/compare.py` (cosine similarity computed by hand), run outside the container with `PYTHONPATH=src uv run python scripts/compare.py` and, for pair A, also inside the container with `docker compose exec api python scripts/compare.py`. Both runs gave the same value.

## Results

| Pair | Text 1 | Text 2 | Cosine similarity | Expectation |
|---|---|---|---|---|
| A (close) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | **0.5957** | high (roughly > 0.6) |
| B (unrelated) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | **0.1920** | low (roughly < 0.4) |
| C (generic) | Backend services | API development | **0.5407** | no fixed expectation |

## Commentary

- Pair B behaves as expected: 0.1920 is far below 0.4, so unrelated topics are clearly separated.
- Pair A lands just under the roughly-0.6 expectation (0.5957). The two texts describe the same thing with different words (OAuth/JWT vs "JSON Web Tokens", fintech vs banking), so the thresholds are an orientation, not a hard line. What matters is the gap: A scores about 0.40 higher than B.
- Pair C is the surprising one. Two very short generic phrases score 0.5407, almost as high as the genuinely close pair A. With so little text there is no discriminating signal, so the model mostly captures "software backend work". This is a good argument for the context header the chunker prepends (project summary, client sector, year, main technology): a chunk that only said "API development" would match almost anything.
- Overall the pipeline works end to end and the embeddings separate close from distant texts, which is the minimum this sanity check is meant to show. It is not a formal quality evaluation.

## Sample ingest (same run)

The session 7 run was a historical in-memory result (52 chunks, 4,126 tokens, $0.0000825). Since session 8 the same corpus is persisted (15 documents, 52 chunks) and queried through `/search`; see `output_examples.txt` at the project root.
