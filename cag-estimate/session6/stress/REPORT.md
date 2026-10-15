# CAG stress test — report

Baseline for Session 6. Model: `claude-haiku-4-5-20251001` (single provider). Nothing was tuned: `SESSION_MAX_TURNS=6`, prompts and validators are untouched.
Data: `results.csv`, 68 rows (54 ok, 14 errors). Run 2026-10-14/15 against a local server on the `session_6` branch.

| Setting | Value |
|---|---|
| Scenarios | `growing`, `pivot`, `contradiction`, 20 turns each, **1 repeat** |
| Attachment sweep | 5 / 20 / 50 / 100 KB PDF, `growing`, first 2 turns, 1 repeat |
| Budgets (pass/fail metrics) | latency <= 8000 ms, cost <= $0.02 per turn |
| Visible spend | $4.61 (sum of `cost_usd` over ok rows) |

**How to read the numbers**
- `turn_index` on ok rows is the number of *successful* turns so far (history depth). A failed turn does not advance it. The "turn" axis below is therefore history depth.
- `latency_ms`, `tokens_*` and `cost_usd` are the **server-side sum over every LLM call in the turn**. That includes Instructor re-prompts after validator failures. `tokens_out` reaches 13,220 at depth 15 while one call is capped at 8,192, so a single turn there made more than one call. The growth below is *context growth plus retries*; the CSV cannot separate the two.
- Cache hit rate is 0% for both caches: the session path uses neither (`cache_hit_kind` is `none` on all 54 ok rows).
- Only 1 repeat per cell, so P95 over 2-17 samples is indicative, not statistical.

## 1. Summary

| Scenario | KB | n | ok | err | P50 s | P95 s | Total $ | Exact / semantic cache | Fact recall |
|---|--:|--:|--:|--:|--:|--:|--:|---|--:|
| growing | 0 | 20 | 17 | 3 | 71.6 | 106.8 | 1.528 | 0% / 0% | 100% |
| pivot | 0 | 20 | 15 | 5 | 104.8 | 170.3 | 1.564 | 0% / 0% | 100% |
| contradiction | 0 | 20 | 16 | 4 | 51.8 | 183.8 | 1.385 | 0% / 0% | 100% |
| growing | 5 | 2 | 2 | 0 | 22.1 | 23.6 | 0.038 | 0% / 0% | 100% |
| growing | 20 | 2 | 2 | 0 | 22.4 | 23.7 | 0.041 | 0% / 0% | 100% |
| growing | 50 | 2 | 2 | 0 | 25.1 | 30.7 | 0.055 | 0% / 0% | 100% |
| growing | 100 | 2 | 0 | 2 | n/a | n/a | 0 | n/a | n/a |

Percentiles and recall are over ok turns only. Budget metrics: **latency budget passed on 0 of 54** ok turns (even depth 1 takes about 19 s); **cost budget passed on 7 of 54**, all at depth 1-3.

## 2. Curves

**Latency vs tokens_in** (0 KB, ok turns)

| tokens_in | n | mean tokens_in | mean latency s | mean cost $ |
|---|--:|--:|--:|--:|
| < 10k | 6 | 6,700 | 21.9 | 0.018 |
| 10-30k | 12 | 20,166 | 40.8 | 0.041 |
| 30-60k | 9 | 43,463 | 79.8 | 0.080 |
| 60-100k | 16 | 80,532 | 110.6 | 0.132 |
| > 100k | 5 | 128,168 | 183.1 | 0.208 |

**Cumulative cost vs history depth** (USD, ok turns only)

| Depth | growing | pivot | contradiction |
|--:|--:|--:|--:|
| 1 | 0.016 | 0.016 | 0.017 |
| 3 | 0.066 | 0.059 | 0.093 |
| 6 | 0.247 | 0.293 | 0.215 |
| 10 | 0.698 | 0.820 | 0.415 |
| 15 | 1.297 | 1.564 | 1.348 |

Per-turn cost goes from $0.016 at depth 1 to $0.168 at depth 15 (10.4x); latency from 19 s to 143 s (7.4x); tokens_in from 5.2k to 101.5k (19.5x).

**Recall vs depth, and what memory held**

| Depth | Fact recall (project name "Nimbus") | summary_chars | anchors | messages in window |
|--:|--:|--:|--:|--:|
| 1 | 100% | 0 | 0 | 2 |
| 6 | 100% | 0 | 0 | 12 |
| 7 | 100% | 151 | 0 | 12 |
| 10 | 100% | 310 | 0 | 12 |
| 15 | 100% | 540 | 0 | 12 |

**Attachments:** tokens_in rises about 2k tokens per 15 KB of text (5 KB: 5,879 -> 50 KB: 11,709 on turn 1) and cost stays near $0.02-0.03. A **100 KB PDF is rejected with HTTP 413** on both turns: the service raises an error above 80,000 attachment characters, it does not truncate.

## 3. Interpretation

**Where the CAG starts to break.** Memory is not what breaks first: the turn-1 project name was recalled on 54 of 54 ok turns, because the heuristic metadata extractor keeps it in `project_metadata` regardless of the window. What breaks is cost, latency and reliability, from the first turns. The 8 s latency budget is never met (about 19 s at depth 1) and the $0.02 cost budget is only met at depth 1-3. After the window fills (depth 6) each turn costs 4.6x the first one, by depth 15 it is 10.4x, and the cumulative cost of one 15-turn session is $1.30-$1.56. Reliability degrades with it: 1 of 18 turns failed in turns 1-6 (6%), 6 of 18 in turns 7-12 (33%) and 5 of 24 in turns 13-20 (21%). The failures are HTTP 502 where Instructor ran out of its 4 attempts because the estimate broke a validator (for example `total_hours` of 522-524 against a 500 maximum), plus one client-side read timeout (duration not recorded). Failing hard is visible; the quiet part is that the retries that precede each failure inflate tokens and latency first.

**What would justify RAG, and the limits of this data.** The dominant dimension is cost and latency growing with history depth and with retries, not recall. A realistic trigger for RAG is a project that needs more than the 80,000 characters the CAG accepts (the 100 KB PDF is refused outright) or a session that must stay under a latency SLA: at 100k+ tokens per turn the mean latency is 183 s. Caveats before drawing conclusions: one repeat per cell; the fact tracker only follows the turn-1 project name, so the pivot (React -> Flutter) and the contradiction (30k -> 80k EUR) were exercised but not measured; anchors stayed at 0 because the heuristic detector never fired on these transcripts; and the summarizer holds 540 chars by depth 15, so any loss of detail there is not captured by an exact-substring metric. The visible spend ($4.61) excludes failed retries, which are billed but log no cost row, so real spend is higher.

## 4. Design notes

- **Where the metrics live:** `LatencyBudgetMetric`, `CostBudgetMetric` and `MemoryDriftMetric` are in `src/cag_estimate/evals/stress/metrics.py`, not `evals/metrics.py`, because they consume the turn observation and the session snapshot, not a final estimate. They reuse `MetricResult`.
- **How the observation reaches the runner:** `turn_observed` is logged and also returned as `observation` in the `/estimate` response; the snapshot (`GET /sessions/{id}`) exposes summary and anchor text. This keeps the runner HTTP-only, with no log parsing.
- **Reproduce:** `python -m cag_estimate.evals.stress.run --base-url http://localhost:8000 --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 --repeats 3 --turns 20 --output session6/stress/results.csv` (PDFs are generated on demand). A first full attempt failed on exhausted API credits and is not included.
