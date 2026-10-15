"""Deterministic generator of the attachment PDFs used by the stress test.

``attach_{5,20,50,100}kb.pdf`` carry about that many thousand characters of
extractable text (the app measures attachments in characters). Words come from a
fixed vocabulary and a seeded RNG and the PDF dates are pinned, so the same bytes
are produced on every run. The output is git-ignored and rebuilt on demand::

    uv run python -m cag_estimate.evals.stress.fixtures.build_pdfs
"""

import random
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

ATTACHMENT_SIZES_KB = (5, 20, 50, 100)
DEFAULT_OUTPUT_DIR = Path("session6/stress/fixtures")
_SEED = 20260610
_FIXED_DATE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_VOCABULARY = [
    "portal",
    "customer",
    "order",
    "status",
    "ticket",
    "invoice",
    "agent",
    "dashboard",
    "report",
    "integration",
    "requirement",
    "module",
    "release",
    "backup",
    "staging",
    "pilot",
    "account",
    "role",
    "email",
    "search",
    "audit",
    "mobile",
    "browser",
    "database",
    "service",
    "endpoint",
    "review",
    "schedule",
    "budget",
    "milestone",
    "feature",
    "delivery",
    "support",
    "workflow",
    "notification",
    "permission",
    "template",
    "deployment",
    "monitoring",
]


def _body_text(chars: int) -> str:
    rng = random.Random(_SEED)
    words: list[str] = []
    length = 0
    while length < chars:
        word = rng.choice(_VOCABULARY)
        words.append(word)
        length += len(word) + 1
    return " ".join(words)[:chars]


def attachment_path(output_dir: Path, kb: int) -> Path:
    return output_dir / f"attach_{kb}kb.pdf"


def build_attachment_pdf(kb: int, path: Path) -> Path:
    """Write a PDF of about ``kb`` x 1000 extractable characters to ``path``."""
    pdf = FPDF()
    pdf.creation_date = _FIXED_DATE
    pdf.set_compression(False)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, _body_text(kb * 1000))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(pdf.output()))
    return path


def build_all(output_dir: Path, sizes: tuple[int, ...] = ATTACHMENT_SIZES_KB) -> list[Path]:
    return [build_attachment_pdf(kb, attachment_path(output_dir, kb)) for kb in sizes]


def ensure_fixtures(
    output_dir: Path, sizes: tuple[int, ...] = ATTACHMENT_SIZES_KB
) -> dict[int, Path]:
    """Paths by size in KB, building only the files that do not exist yet."""
    paths: dict[int, Path] = {}
    for kb in sizes:
        path = attachment_path(output_dir, kb)
        if not path.exists():
            build_attachment_pdf(kb, path)
        paths[kb] = path
    return paths


if __name__ == "__main__":
    for built in build_all(DEFAULT_OUTPUT_DIR):
        print(built)
