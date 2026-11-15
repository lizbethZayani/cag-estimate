# Session 11: results (cited RAG estimates and RAGAS baseline)

TEMPLATE: no numbers yet. Paste the tables printed by `scripts/eval_ragas.py` (see `MANUAL_VALIDATION.md`) and fill in the last section. Raw outputs: `session11/ragas_samples.json` and `session11/ragas_raw.json`.

## Setup

Corpus: 15 historical budgets, 52 chunks (`data/budgets_sample.json`). Golden set: 5 English queries with their relevant budgets and a reference total derived from real component hours (`src/cag_estimate/evals/retrieval_golden_set.json`).

| Query | Relevant budgets | Reference total (hours) |
|---|---|---|
| q1-ecommerce-store | BUD-2024-005, BUD-2024-006 | 620 |
| q2-fintech-compliance | BUD-2024-001, BUD-2024-002 | 420 |
| q3-remote-care | BUD-2024-009, BUD-2024-010 | 830 |
| q4-industrial-data | BUD-2024-013, BUD-2024-015 | 840 |
| q5-logistics | BUD-2024-007, BUD-2024-014 | 530 |

| Item | Value |
|---|---|
| Generator | OpenAI Responses API, model `GENERATION_MODEL` (default `gpt-4o-mini`), `GENERATION_MAX_OUTPUT_TOKENS` 4000 |
| Retrieval | hybrid, no rerank, `RAG_TOP_K` 10 |
| Judge | `gpt-4o-mini` |
| Embeddings (answer relevancy) | `text-embedding-3-small` |

## Method

- **Answer.** The cited estimate is rendered as plain text in hours (one bullet per line with its evidence, the computed total, assumptions, confidence); an `insufficient` estimate is rendered as its explanation.
- **Ground truth.** The reference of each query is derived from the real historical component hours of its relevant budgets, one bullet per component plus the total.
- **Faithfulness:** share of the answer's claims that the retrieved contexts support.
- **Answer relevancy:** how well the answer addresses the question (similarity of questions regenerated from the answer).
- **Context precision:** whether the relevant retrieved chunks are ranked first, judged against the reference.
- **Context recall:** share of the reference that the retrieved contexts cover.

## Deviations from the brief

- Thin RAG service (retrieve, generate, verify), because this repo has no rewrite, expand or route stages.
- Generation uses the OpenAI Responses API with strict structured output, as the brief asks.
- `hours` is nullable: it is null exactly when the line is not grounded.
- Chunk ids use the `BUDGET::COMPONENT` form (a chunk is a budget component).
- ragas 0.4.4: the modern `ragas.metrics.collections` interface instead of `evaluate()`, and `ragas.embeddings.OpenAIEmbeddings` because `embedding_factory` is deprecated.

## Results

PENDING: paste the table printed by scripts/eval_ragas.py

| Query | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| | | | | |

## Citation verification report

PENDING: paste the table printed by scripts/eval_ragas.py

| Query | Status | Lines | Grounded | Dangling | Insufficient |
|---|---|---|---|---|---|
| | | | | | |

## What bothers me most about the numbers

PENDING: 2 to 3 sentences. Does faithfulness drop with coarse citation? Is context recall weak on any query?
