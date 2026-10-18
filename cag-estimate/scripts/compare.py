#!/usr/bin/env python3
"""Compare two texts: embed both and print their cosine similarity.

Run it from the project directory (a complete ``.env`` is needed, as for the API).

Locally (``uv run`` can hide the editable install, so point it at ``src``):

    PYTHONPATH=src uv run python scripts/compare.py --text-a "..." --text-b "..."

In the container:

    docker compose exec api python scripts/compare.py --text-a "..." --text-b "..."

Exit codes: 0 success, 1 the embedding request failed, 2 bad usage or no OpenAI key.
"""

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from openai import OpenAI

from cag_estimate.config import get_settings
from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder, cosine_similarity

EXIT_OK = 0
EXIT_EMBEDDING_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cosine similarity between two embedded texts.")
    parser.add_argument("--text-a", required=True, help="first text")
    parser.add_argument("--text-b", required=True, help="second text")
    return parser


def build_embedder() -> OpenAIEmbedder | None:
    """Real embedder from the settings, or None when no OpenAI key is configured."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    client = OpenAI(api_key=settings.openai_api_key, max_retries=0)
    return OpenAIEmbedder(client, model=settings.embedding_model)


def main(
    argv: Sequence[str] | None = None,
    embedder: OpenAIEmbedder | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    if not args.text_a.strip() or not args.text_b.strip():
        err.write("error: texts must not be empty or blank\n")
        return EXIT_USAGE
    embedder = embedder or build_embedder()
    if embedder is None:
        err.write("error: OPENAI_API_KEY is not configured\n")
        return EXIT_USAGE
    try:
        similarity = cosine_similarity(
            embedder.embed_one(args.text_a), embedder.embed_one(args.text_b)
        )
    except Exception as exc:  # noqa: BLE001 - a CLI reports any failure and exits 1
        err.write(f"error: embedding failed: {type(exc).__name__}: {exc}\n")
        return EXIT_EMBEDDING_FAILED
    out.write(f"Text A: {args.text_a}\nText B: {args.text_b}\n")
    out.write(f"Cosine similarity: {similarity:.4f}\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
