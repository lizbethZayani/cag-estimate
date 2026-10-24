#!/usr/bin/env python3
"""Ingest the sample budgets through the API, then run five semantic search queries.

Run it against a running API (the pgvector database must be migrated):

    uv run python scripts/query_examples.py

The API address comes from ``API_BASE_URL`` (default ``http://localhost:8000``).
Re-runs are safe: budgets that are already ingested answer 409 and are only counted.

Exit codes: 0 success, 1 the API failed or is unreachable, 2 bad usage or input file.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any, TextIO

import httpx

EXIT_OK = 0
EXIT_API_FAILED = 1
EXIT_USAGE = 2

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_DATA = "data/budgets_sample.json"
DOCUMENT_TYPE = "historical_budget"
SNIPPET_CHARS = 120
TIMEOUT_SECONDS = 120.0
QUERIES = (
    "OAuth 2.0 authentication with JWT tokens",
    "payment processing with Stripe",
    "restaurant reservations",
    "high-complexity backend components",
    "real-time dashboard",
)


class InputError(Exception):
    """The data file is missing or is not a JSON array of budgets."""


class ApiError(Exception):
    """The API answered with an unexpected status or could not be reached."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest sample budgets and run search queries.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="JSON array of budgets")
    parser.add_argument("--k", type=int, default=5, help="results per query")
    return parser


def load_budgets(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            budgets = json.load(handle)
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read {path}: {type(exc).__name__}") from exc
    if not isinstance(budgets, list) or not all(
        isinstance(b, dict) and isinstance(b.get("budget_id"), str) for b in budgets
    ):
        raise InputError(f"{path} must be a JSON array of budgets with a budget_id")
    return budgets


def post(client: httpx.Client, path: str, body: dict[str, Any]) -> httpx.Response:
    try:
        return client.post(path, json=body)
    except httpx.HTTPError as exc:
        raise ApiError(f"request to {path} failed: {type(exc).__name__}") from exc


def ingest(client: httpx.Client, budgets: list[dict[str, Any]], data_path: str) -> tuple[int, int]:
    """POST each budget; return (ingested, already present)."""
    ingested = present = 0
    for budget in budgets:
        body = {
            "source_path": f"{data_path}::{budget['budget_id']}",
            "document_type": DOCUMENT_TYPE,
            "content": budget,
        }
        status = post(client, "/embeddings/ingest", body).status_code
        if status == 200:
            ingested += 1
        elif status == 409:
            present += 1
        else:
            raise ApiError(f"ingest of {budget['budget_id']} failed with status {status}")
    return ingested, present


def format_result(rank: int, hit: dict[str, Any]) -> str:
    snippet = " ".join(str(hit["content"]).split())[:SNIPPET_CHARS]
    return (
        f"  {rank}. chunk_id={hit['chunk_id']} distance={hit['distance']:.4f} "
        f"type={hit['chunk_type']} | {snippet}"
    )


def run_query(client: httpx.Client, number: int, query: str, k: int, out: TextIO) -> None:
    response = post(client, "/search", {"query": query, "k": k})
    if response.status_code != 200:
        raise ApiError(f"search for query {number} failed with status {response.status_code}")
    payload = response.json()
    out.write(f"Query {number}: {query}\n")
    lines = [format_result(rank, hit) for rank, hit in enumerate(payload["results"], start=1)]
    out.write("\n".join(lines or ["  (no results)"]) + "\n")
    out.write(f"  search_time_ms={payload['search_time_ms']}\n\n")


def main(
    argv: Sequence[str] | None = None,
    client: httpx.Client | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        budgets = load_budgets(args.data)
    except InputError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_USAGE
    base_url = os.environ.get("API_BASE_URL", DEFAULT_BASE_URL)
    http = client or httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS)
    try:
        ingested, present = ingest(http, budgets, args.data)
        out.write(f"Ingested {ingested}, already present {present}\n\n")
        for number, query in enumerate(QUERIES, start=1):
            run_query(http, number, query, args.k, out)
    except ApiError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_API_FAILED
    finally:
        if client is None:
            http.close()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
