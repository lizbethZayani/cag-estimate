#!/usr/bin/env python3
"""Session 9 pre-work: manual trace of a transcript through the current system.

Client only. It adds no behaviour to the service. Two steps:

1. Embed the text with OpenAI (no endpoint of the service returns a raw vector).
2. POST the same text to ``/search`` and print the raw JSON.

Run it from the project directory (``cag-estimate/cag-estimate``), with the stack up:

    PYTHONPATH=src uv run python ../examples/trace_s09.py ../examples/transcripts/02_ambiguous.txt

Use ``--text`` instead of a file to try a short hand-written query.
"""

import argparse
import json
import math
import os
import sys

import httpx
from openai import OpenAI

from cag_estimate.config import get_settings

MODEL = "text-embedding-3-small"
HEAD_TAIL = 5


def embed(text: str) -> list[float]:
    api_key = get_settings().openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("error: OPENAI_API_KEY is not configured")
    response = OpenAI(api_key=api_key, max_retries=0).embeddings.create(model=MODEL, input=text)
    return response.data[0].embedding


def print_embedding(vector: list[float]) -> None:
    norm = math.sqrt(sum(x * x for x in vector))
    print("== STEP 1: embedding ==")
    print(f"model={MODEL} dimensions={len(vector)} l2_norm={norm:.6f}")
    print(f"first {HEAD_TAIL}: {[round(x, 6) for x in vector[:HEAD_TAIL]]}")
    print(f"last  {HEAD_TAIL}: {[round(x, 6) for x in vector[-HEAD_TAIL:]]}")


def search(text: str, k: int, base_url: str) -> dict:
    response = httpx.post(f"{base_url}/search", json={"query": text, "k": k}, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", help="transcript file")
    parser.add_argument("--text", help="short query instead of a file")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()
    if bool(args.path) == bool(args.text):
        parser.error("give either a transcript path or --text")
    text = args.text or open(args.path, encoding="utf-8").read()
    print(f"input: {len(text)} characters, {len(text.split())} words")
    print_embedding(embed(text))
    print("== STEP 2: POST /search ==")
    print(json.dumps(search(text, args.k, args.base_url), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
