# Session 11: results (cited RAG estimates and RAGAS baseline)

The tables below are printed by `scripts/eval_ragas.py` (see `MANUAL_VALIDATION.md`) from a real run on 2026-11-18: 5 queries, one generation each (every query accepted on the first attempt), scored by `gpt-4o-mini`. Raw outputs: `session11/ragas_samples.json` (answers, contexts and citation reports) and `session11/ragas_raw.json` (scores).

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

| Query | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| q1-ecommerce-store | 0.39 | 0.10 | 1.00 | 1.00 |
| q2-fintech-compliance | 0.31 | 0.45 | 1.00 | 0.80 |
| q3-remote-care | 0.24 | 0.48 | 0.96 | 1.00 |
| q4-industrial-data | 0.76 | 0.26 | 1.00 | 0.83 |
| q5-logistics | 0.50 | 0.43 | 1.00 | 1.00 |
| **Average** | 0.44 | 0.34 | 0.99 | 0.93 |

Generated total against the reference total:

| Query | Generated (hours) | Reference (hours) |
|---|---|---|
| q1-ecommerce-store | 730 | 620 |
| q2-fintech-compliance | 420 | 420 |
| q3-remote-care | 830 | 830 |
| q4-industrial-data | 1290 | 840 |
| q5-logistics | 530 | 530 |

## Citation verification report

Every estimate was accepted on the first attempt: all 27 lines are grounded, none cites a chunk that was not retrieved, and none was marked as insufficient data.

| Query | Status | Lines | Grounded | Dangling | Insufficient |
|---|---|---|---|---|---|
| q1-ecommerce-store | ok | 6 | 6 | 0 | 0 |
| q2-fintech-compliance | ok | 4 | 4 | 0 | 0 |
| q3-remote-care | ok | 5 | 5 | 0 | 0 |
| q4-industrial-data | ok | 8 | 8 | 0 | 0 |
| q5-logistics | ok | 4 | 4 | 0 | 0 |
| **Total** | | 27 | 27 | 0 | 0 |

The verifier also catches an invented id: `scripts/demo_verify_citations.py` plants `BUD-2099-999::FAKE-001` and reports it as dangling (exit 0).

## What bothers me most about the numbers

Faithfulness is the weakest metric (average 0.44) even though the citation check passed on 27 of 27 lines and every cited figure is quoted verbatim: knowing that a cited id exists says little about whether every statement of the answer is supported. The lowest scores (q3 0.24, q2 0.31) are on the two queries where the generator reproduced the reference total exactly, so the unsupported statements are probably the narrative around the figures (assumptions, "Confidence: high", the computed total); I did not check this statement by statement. Context recall is not weak because of retrieval: every component of every reference was among the 10 retrieved chunks (checked), so the 0.80 on q2 and 0.83 on q4 most likely come from reference statements that are not component facts, such as the total line. The second worry is over-scoping: q1 (730 h against 620 h, an unrequested "personalized recommendations" line) and q4 (1290 h against 840 h, 8 lines against 5) add lines that are grounded and cited but were not asked for, and they have the two lowest answer relevancy scores (0.10 and 0.26); citation verification cannot catch that. There is no coarse-citation baseline here, so this run cannot show whether line-level citation changed faithfulness.

Caveats: 5 queries and one generation each, so the numbers are indicative. The judge is noisy: scoring the same samples twice (before and after the scorer fix) moved q2 faithfulness from 0.56 to 0.31 and q2 context recall from 1.00 to 0.80, so differences under about 0.25 should not be read as real.

## Measurement fixes found in the first live run

The first scoring pass returned `n/a` for `answer_relevancy` on every query and for `faithfulness` on q3 and q4. Two causes, both in the evaluation wiring and fixed with regression tests: ragas 0.4.4 awaits `aembed_text` in `AnswerRelevancy`, which rejects a synchronous OpenAI client (the embeddings client is now async), and the default judge output limit of 1024 tokens truncated the structured output (`IncompleteOutputException`; the limit is now 4096). The samples were then re-scored without regenerating them (`--score-only`).
