#!/usr/bin/env python3
"""Measure precision@k and latency of the four retrieval configurations against a live API.

Run it against a running API whose database already holds the ingested sample budgets:

    uv run python scripts/eval_retrieval.py --runs 3

Configurations: A vector, B hybrid, C vector + rerank, D hybrid + rerank. For every query and
configuration one warm-up request is discarded (it absorbs the one-off reranker model load),
then ``--runs`` timed requests are made. The run aborts when the API does not apply the
requested strategy (for example a reranker that failed soft), because the table would lie.

The API address comes from ``--base-url`` or ``API_BASE_URL`` (default ``http://localhost:8000``).

Exit codes: 0 success, 1 the API failed or answered a different strategy, 2 bad usage or input.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx
from pydantic import ValidationError

from cag_estimate.evals.retrieval_eval import (
    CONFIGS,
    DEFAULT_K,
    GOLDEN_SET_PATH,
    GoldenQuery,
    QueryMeasurement,
    RetrievalConfig,
    load_golden_set,
    precision_at_k,
    render_markdown_table,
)

EXIT_OK = 0
EXIT_API_FAILED = 1
EXIT_USAGE = 2

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RUNS = 3
TIMEOUT_SECONDS = 120.0


class InputError(Exception):
    """The golden set is unreadable or an option is out of range."""


class ApiError(Exception):
    """The API failed, was unreachable or did not apply the requested strategy."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrieval configurations A-D.")
    parser.add_argument(
        "--base-url", default=os.environ.get("API_BASE_URL", DEFAULT_BASE_URL), help="API address"
    )
    parser.add_argument("--golden", default=str(GOLDEN_SET_PATH), help="golden set JSON")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="timed runs per query")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="results per query")
    parser.add_argument("--output", default=None, help="write every raw measurement as JSON")
    return parser


def validate_options(args: argparse.Namespace) -> None:
    if args.runs < 1 or args.k < 1:
        raise InputError("--runs and --k must be at least 1")


def read_golden(path: str) -> list[GoldenQuery]:
    try:
        return load_golden_set(Path(path)).queries
    except (OSError, ValueError, ValidationError) as exc:
        raise InputError(f"cannot load the golden set {path}: {type(exc).__name__}") from exc


def search(
    client: httpx.Client, config: RetrievalConfig, query: str, k: int
) -> tuple[dict[str, Any], float]:
    """One ``POST /search``; return the payload and the client wall-clock latency in ms."""
    body = {"query": query, "k": k, "search_mode": config.search_mode, "rerank": config.rerank}
    started = time.perf_counter()
    try:
        response = client.post("/search", json=body)
    except httpx.HTTPError as exc:
        raise ApiError(f"request to /search failed: {type(exc).__name__}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise ApiError(f"search failed with status {response.status_code}")
    payload = response.json()
    if payload["search_mode"] != config.search_mode or payload["reranked"] != config.rerank:
        raise ApiError(
            f"config {config.name} asked for search_mode={config.search_mode} "
            f"reranked={config.rerank} but the API applied search_mode={payload['search_mode']} "
            f"reranked={payload['reranked']}"
        )
    return payload, elapsed_ms


def measure(
    client: httpx.Client, config: RetrievalConfig, golden: GoldenQuery, k: int, runs: int
) -> QueryMeasurement:
    search(client, config, golden.query, k)  # warm-up, discarded
    timed = [search(client, config, golden.query, k) for _ in range(runs)]
    first_payload = timed[0][0]
    returned = [hit["metadata"]["budget_id"] for hit in first_payload["results"]]
    return QueryMeasurement(
        config=config.name,
        query_id=golden.id,
        precision=precision_at_k(returned, golden.relevant_budget_ids, k),
        latencies_ms=tuple(elapsed for _, elapsed in timed),
        server_times_ms=tuple(payload["search_time_ms"] for payload, _ in timed),
    )


def write_raw(path: str, measurements: Sequence[QueryMeasurement], k: int, runs: int) -> None:
    raw = {
        "k": k,
        "runs": runs,
        "measurements": [
            {
                "config": m.config,
                "query_id": m.query_id,
                "precision": m.precision,
                "latencies_ms": list(m.latencies_ms),
                "server_times_ms": list(m.server_times_ms),
            }
            for m in measurements
        ],
    }
    Path(path).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


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
        validate_options(args)
        queries = read_golden(args.golden)
    except InputError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_USAGE
    http = client or httpx.Client(base_url=args.base_url, timeout=TIMEOUT_SECONDS)
    try:
        measurements = [
            measure(http, config, golden, args.k, args.runs)
            for config in CONFIGS
            for golden in queries
        ]
    except ApiError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_API_FAILED
    finally:
        if client is None:
            http.close()
    out.write(f"queries={len(queries)} runs={args.runs} k={args.k}\n\n")
    out.write(render_markdown_table(measurements) + "\n")
    if args.output:
        write_raw(args.output, measurements, args.k, args.runs)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
