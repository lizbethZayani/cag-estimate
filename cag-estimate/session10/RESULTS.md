# Session 10: results (hybrid search and reranking)

The numbers below are pasted from a real run of `scripts/eval_retrieval.py` (see `MANUAL_VALIDATION.md`) on 2026-11-14: 5 queries, 3 timed runs after 1 discarded warm-up, `k=5`, reranker on CPU inside the `api` container. Raw measurements: `session10/eval_raw.json`.

## Setup

Corpus: 15 historical budgets, 52 chunks (`data/budgets_sample.json`). Golden set: 5 English queries, 2 relevant budgets each (`src/cag_estimate/evals/retrieval_golden_set.json`).

| Query | Topic | Relevant budgets |
|---|---|---|
| q1-ecommerce-store | Online store: catalog, cart and card checkout, order refunds | BUD-2024-005, BUD-2024-006 |
| q2-fintech-compliance | Fintech: OAuth, open banking, regulatory reporting, e-signature | BUD-2024-001, BUD-2024-002 |
| q3-remote-care | Remote care: wearables, clinical dashboard, video, HL7/FHIR | BUD-2024-009, BUD-2024-010 |
| q4-industrial-data | Sensor pipeline, load forecasting, predictive maintenance | BUD-2024-013, BUD-2024-015 |
| q5-logistics | Last-mile logistics: routes, slot booking, inventory, docks | BUD-2024-007, BUD-2024-014 |

Configurations:

| Config | Search | Reranking |
|---|---|---|
| A | vector | off |
| B | hybrid (vector + lexical, RRF) | off |
| C | vector | on |
| D | hybrid (vector + lexical, RRF) | on |

## Method

- **precision@5**: share of the 5 returned chunks whose parent budget (`metadata.budget_id`) is in the annotated relevant set; fewer than 5 results count as misses. Caveat: results are component chunks, so one relevant budget can fill several of the 5 slots, and with only 15 budgets and 5 queries the numbers are indicative, not statistically significant.
- **Latency**: client wall-clock of `POST /search`, median over `--runs` timed requests after one discarded warm-up per query and configuration; the table also shows the server-side `search_time_ms` median.

## Deviations from the brief

- English corpus and golden set, so the full-text search uses the `english` configuration instead of Spanish.
- This repo has no rewrite/generate pipeline, so the scope is the retrieval layer (`POST /search`).
- The lexical branch uses an OR tsquery over the sanitized query tokens ranked by `ts_rank_cd`, not `plainto_tsquery`, which ANDs every term and would match nothing for a long project description.

## Results

| Config | Search | Reranking | Precision@5 (mean) | Latency median (ms) | Server search_time median (ms) |
|---|---|---|---|---|---|
| A | vector | off | 0.92 | 219.1 | 215.0 |
| B | hybrid | off | 1.00 | 219.5 | 216.0 |
| C | vector | on | 0.92 | 3835.2 | 3831.0 |
| D | hybrid | on | 0.92 | 3833.7 | 3828.0 |

## Per-query results

| Query | A | B | C | D |
|---|---|---|---|---|
| q1-ecommerce-store | 0.60 | 1.00 | 0.60 | 0.60 |
| q2-fintech-compliance | 1.00 | 1.00 | 1.00 | 1.00 |
| q3-remote-care | 1.00 | 1.00 | 1.00 | 1.00 |
| q4-industrial-data | 1.00 | 1.00 | 1.00 | 1.00 |
| q5-logistics | 1.00 | 1.00 | 1.00 | 1.00 |

Only q1 separates the configurations. Top 5 for q1 (relevant budgets: 005 and 006):

| Config | Top 5 chunks (budget/component) | Relevant |
|---|---|---|
| A | 005/CART-002, 005/CATALOG-001, **007/INV-003**, 005/STORE-004, **007/SLOT-001** | 3 of 5 |
| B | 005/CART-002, 005/CATALOG-001, 005/STORE-004, 005/RECO-003, 006/ORDER-002 | 5 of 5 |
| C and D | 005/CATALOG-001, 005/CART-002, 005/RECO-003, **008/RESALE-003**, **007/SLOT-001** | 3 of 5 |

## Conclusions

**Which configuration?** B, hybrid without reranking. It has the best precision@5 (1.00 against 0.92 for the other three) at the same latency as plain vector search (about 220 ms). On q1 the keyword branch is what helps: terms such as "order management" bring `006/ORDER-002` and `005/RECO-003` into the top 5, while vector search drifts to the grocery-delivery budget (007).

**Does reranking pay off here?** No. It multiplies latency by about 17 (3.8 s against 0.22 s on CPU) and gains nothing: C equals A (0.92) and D is worse than B (0.92 against 1.00), because the cross-encoder pushed `008/RESALE-003` and `007/SLOT-001` above the better 005 and 006 chunks on q1. C and D return identical results because the corpus has only 52 chunks, so the top-50 recall pool is almost the whole corpus: "recall then rerank" degenerates into reranking everything, and the search mode stops mattering. Its cost is also more than latency: the `api` image grew from 1.93 GB to 17.9 GB (torch), the model is about 450 MB, and the first reranked request took 13 s while the model loaded.

**Caveats.** The golden set is small: 5 queries, and 4 of them score 1.00 under every configuration, so the whole difference comes from q1. Precision is measured on chunks with 2 relevant budgets per query, and the golden set is mine, so these numbers are indicative, not significant. I would revisit reranking with a corpus of thousands of chunks (where a top-50 pool is selective), a harder golden set, and a lighter or GPU-backed model, before paying its cost.
