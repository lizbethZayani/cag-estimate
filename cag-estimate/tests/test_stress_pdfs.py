"""Offline tests for the deterministic attachment PDF generator."""

from pathlib import Path

import pytest
from pypdf import PdfReader

from cag_estimate.evals.stress.fixtures.build_pdfs import (
    ATTACHMENT_SIZES_KB,
    build_all,
    build_attachment_pdf,
    ensure_fixtures,
)


def _extracted_chars(path: Path) -> int:
    return sum(len(page.extract_text()) for page in PdfReader(path).pages)


@pytest.mark.parametrize("kb", [5, 20])
def test_extracted_text_length_matches_the_target_size(tmp_path: Path, kb: int) -> None:
    path = build_attachment_pdf(kb, tmp_path / f"attach_{kb}kb.pdf")

    assert _extracted_chars(path) == pytest.approx(kb * 1000, rel=0.15)


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = build_attachment_pdf(5, tmp_path / "a.pdf").read_bytes()
    second = build_attachment_pdf(5, tmp_path / "b.pdf").read_bytes()

    assert first == second


def test_build_all_writes_one_file_per_size(tmp_path: Path) -> None:
    paths = build_all(tmp_path, sizes=(5, 20))

    assert [path.name for path in paths] == ["attach_5kb.pdf", "attach_20kb.pdf"]
    assert all(path.exists() for path in paths)
    assert ATTACHMENT_SIZES_KB == (5, 20, 50, 100)


def test_ensure_fixtures_builds_only_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "attach_5kb.pdf"
    existing.write_bytes(b"keep")

    paths = ensure_fixtures(tmp_path, sizes=(5, 20))

    assert existing.read_bytes() == b"keep"
    assert paths[20].exists()
    assert set(paths) == {5, 20}
