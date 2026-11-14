# Session 10: results (hybrid search and reranking)

Template: the numbers are pasted from the real run of `scripts/eval_retrieval.py` (see `MANUAL_VALIDATION.md`).

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

PENDING: paste the table printed by scripts/eval_retrieval.py

| Config | Search | Reranking | Precision@5 (mean) | Latency median (ms) | Server search_time median (ms) |
|---|---|---|---|---|---|
| A | vector | off | | | |
| B | hybrid | off | | | |
| C | vector | on | | | |
| D | hybrid | on | | | |

## Per-query results

PENDING: paste the table printed by scripts/eval_retrieval.py

| Query | A | B | C | D |
|---|---|---|---|---|
| q1-ecommerce-store | | | | |
| q2-fintech-compliance | | | | |
| q3-remote-care | | | | |
| q4-industrial-data | | | | |
| q5-logistics | | | | |

## Conclusions

PENDING: answer with the real numbers once the run is done.

1. Which configuration would you use in the project, and why?
2. Does the relevance gain from reranking justify its latency in this use case?

Remember that the cost of reranking is not only latency: it also adds the torch weight to the Docker image and a model download of about 450 MB.
