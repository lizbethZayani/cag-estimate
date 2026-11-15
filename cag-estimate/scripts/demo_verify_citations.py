#!/usr/bin/env python3
"""Offline demo of citation verification: a planted invented id is caught as dangling.

It builds a small estimate for budget BUD-2024-005 (retrieved chunk ids come from
``data/budgets_sample.json``) with three lines: one grounded on a real chunk, one grounded on
the invented id ``BUD-2099-999::FAKE-001`` and one not grounded. No network, database or API key:

    PYTHONPATH=src uv run python scripts/demo_verify_citations.py

Exit codes: 0 the planted dangling citation was detected (the demo's purpose), 1 it was NOT
detected (the verifier is broken).
"""

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from cag_estimate.schemas.rag_estimation import EstimateLineItem, RagEstimate, SourceReference
from cag_estimate.services.citations import CitationReport, verify_citations

EXIT_OK = 0
EXIT_NOT_DETECTED = 1

BUDGETS_PATH = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"
BUDGET_ID = "BUD-2024-005"
REAL_CHUNK_ID = f"{BUDGET_ID}::CATALOG-001"  # "Product catalog API", 150 h in the sample data
PLANTED_ID = "BUD-2099-999::FAKE-001"

Verifier = Callable[[RagEstimate, set[str]], CitationReport]


def load_chunk_ids(path: Path = BUDGETS_PATH, budget_id: str = BUDGET_ID) -> set[str]:
    """Chunk ids ``<budget_id>::<component_id>`` of one budget, as the retriever builds them."""
    budgets = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{budget['budget_id']}::{component['component_id']}"
        for budget in budgets
        if budget["budget_id"] == budget_id
        for component in budget["components"]
    }


def _source(chunk_id: str, evidence: str) -> SourceReference:
    return SourceReference(
        chunk_id=chunk_id, document_id=chunk_id.split("::")[0], evidence=evidence
    )


def build_estimate(real_chunk_id: str) -> RagEstimate:
    """Three lines: grounded on a real chunk, grounded on the planted id, not grounded."""
    return RagEstimate(
        lines=[
            EstimateLineItem(
                component="Product catalog API",
                hours=150,
                rationale="Matches the catalog component of the historical store.",
                grounded=True,
                sources=[_source(real_chunk_id, "Product catalog API")],
            ),
            EstimateLineItem(
                component="Loyalty program",
                hours=80,
                rationale="Supported by a source the retriever never returned.",
                grounded=True,
                sources=[_source(PLANTED_ID, "Loyalty program, 80 hours")],
            ),
            EstimateLineItem(
                component="Gift cards",
                hours=None,
                rationale="No historical budget includes gift cards.",
                grounded=False,
                sources=[],
            ),
        ],
        assumptions=[],
        confidence="low",
        insufficient_context_explanation=None,
    )


def render_report(report: CitationReport) -> str:
    """Markdown table of the per-line outcome plus a summary line."""
    rows = [
        f"| {line.component} | {line.status} | {', '.join(line.cited_chunk_ids) or '-'} "
        f"| {', '.join(line.dangling_chunk_ids) or '-'} |"
        for line in report.lines
    ]
    summary = (
        f"lines={report.total_lines} grounded={report.grounded_lines} "
        f"dangling={report.dangling_lines} insufficient={report.insufficient_lines}"
    )
    return "\n".join(
        ["| Line | Status | Cited ids | Dangling ids |", "|---|---|---|---|", *rows, "", summary]
    )


def main(
    argv: Sequence[str] | None = None,
    verify: Verifier = verify_citations,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    retrieved = load_chunk_ids()
    estimate = build_estimate(REAL_CHUNK_ID)
    report = verify(estimate, retrieved)
    out.write(render_report(report) + "\n")
    if report.has_dangling and PLANTED_ID in report.dangling_chunk_ids:
        out.write(f"DANGLING CITATION DETECTED: {PLANTED_ID}\n")
        return EXIT_OK
    err.write(f"error: the verifier did not detect the planted id {PLANTED_ID}\n")
    return EXIT_NOT_DETECTED


if __name__ == "__main__":
    sys.exit(main())
