#!/usr/bin/env python3
"""Check that the configured cross-encoder loads and ranks a relevant document higher.

Run it once after installing the dependencies (the first run downloads the model):

    uv run python scripts/verify_reranker.py

The model comes from ``RERANK_MODEL`` (see ``Settings``).

Exit codes: 0 the relevant document scored higher, 1 the model failed to load or score,
2 the ranking is wrong.
"""

import sys
from collections.abc import Sequence
from typing import TextIO

from cag_estimate.config import get_settings
from cag_estimate.embedding_pipeline.reranker import CrossEncoderReranker

EXIT_OK = 0
EXIT_MODEL_FAILED = 1
EXIT_WRONG_RANKING = 2

QUERY = "online store with product catalog and card checkout"
RELEVANT = (
    "E-commerce platform with a product catalog, shopping cart and a card payment "
    "checkout for an online store."
)
IRRELEVANT = (
    "Hospital system to schedule patient appointments and manage doctor availability "
    "in clinic wards."
)


def check_ranking(relevant_score: float, irrelevant_score: float) -> int:
    """Exit code: ``EXIT_OK`` only when the relevant document strictly outscores the other."""
    return EXIT_OK if relevant_score > irrelevant_score else EXIT_WRONG_RANKING


def main(
    argv: Sequence[str] | None = None,
    reranker: CrossEncoderReranker | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    model = reranker or CrossEncoderReranker(get_settings().rerank_model)
    try:
        relevant, irrelevant = model.score(QUERY, [RELEVANT, IRRELEVANT])
    except Exception as exc:  # noqa: BLE001 - any load or inference failure is exit 1
        err.write(f"error: reranker model failed ({type(exc).__name__})\n")
        return EXIT_MODEL_FAILED
    out.write(f"model={model.model_name}\nrelevant={relevant:.4f}\nirrelevant={irrelevant:.4f}\n")
    code = check_ranking(relevant, irrelevant)
    if code != EXIT_OK:
        err.write("error: the ranking is wrong, the irrelevant document scored higher\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
