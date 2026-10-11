"""CLI: ``python -m cag_estimate.evals.run``.

Exit codes: 0 every case passed every metric; 1 some case failed or errored;
2 usage or connectivity problem detected before running (nothing was estimated).
"""

import argparse
import json
import sys
from pathlib import Path

import requests

from cag_estimate.config import Settings
from cag_estimate.evals.dataset import GoldenCase, load_dataset
from cag_estimate.evals.metrics import METRICS, NOT_APPLICABLE
from cag_estimate.evals.runner import (
    DEFAULT_TIMEOUT_SECONDS,
    CaseReport,
    EvalReport,
    HttpClient,
    Mode,
    evaluate_cases,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
DEFAULT_BASE_URL = "http://localhost:8000"
HEALTH_TIMEOUT_SECONDS = 10.0
ID_WIDTH = 34
ACB_ITERATIONS: int = Settings.model_fields["acb_max_iterations"].default


def write(text: str = "") -> None:
    """The single output channel of the CLI."""
    sys.stdout.write(f"{text}\n")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cag_estimate.evals.run",
        description="Run the golden cases against a RUNNING API (real LLM calls).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--mode", choices=["standard", "acb"], default="standard")
    parser.add_argument("--limit", type=_positive_int, help="run only the first N cases")
    parser.add_argument("--case", action="append", default=[], help="case id (repeatable)")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="list the plan, send nothing")
    return parser


def select_cases(cases: list[GoldenCase], ids: list[str], limit: int | None) -> list[GoldenCase]:
    """Filter by id (unknown ids raise ``ValueError``), then keep the first ``limit``."""
    unknown = sorted(set(ids) - {case.id for case in cases})
    if unknown:
        raise ValueError(f"unknown case id(s): {', '.join(unknown)}")
    chosen = [case for case in cases if not ids or case.id in ids]
    return chosen[:limit]


def _estimate_calls(cases: list[GoldenCase]) -> int:
    return sum(len(case.turns) for case in cases)


def _acb_note(calls: int) -> str:
    extra = ACB_ITERATIONS * calls
    return f"up to {extra} drafts and {extra} critic reviews in total ({ACB_ITERATIONS} per turn)"


def _plan(cases: list[GoldenCase], mode: Mode, base_url: str) -> str:
    calls = _estimate_calls(cases)
    text = f"{len(cases)} cases, {calls} estimate calls"
    if mode == "acb":
        text += f" (acb: up to {ACB_ITERATIONS} x (draft + review) per turn)"
    return f"{text} against {base_url}"


def write_dry_run(cases: list[GoldenCase], mode: Mode, base_url: str) -> None:
    write(f"Dry run, nothing is sent: {_plan(cases, mode, base_url)}")
    for case in cases:
        write(f"  {case.id:<{ID_WIDTH}} {len(case.turns)} turn(s)  {case.description}")
    if mode == "acb":
        write(f"Reviewed mode adds {_acb_note(_estimate_calls(cases))}.")


def _check_health(client: HttpClient, base_url: str) -> str | None:
    """Return a problem description, or None when ``/health`` answers 200."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        status = client.get(url, timeout=HEALTH_TIMEOUT_SECONDS).status_code
    except Exception as exc:  # noqa: BLE001 - any failure means the API is unreachable
        return f"cannot reach {url}: {type(exc).__name__}"
    return None if status == 200 else f"cannot use {url}: HTTP {status}"


def _verdict(result_name: str, report: CaseReport) -> str:
    result = next(item for item in report.results if item.name == result_name)
    if result.details == NOT_APPLICABLE:
        return "-"
    return "PASS" if result.passed else "FAIL"


def _header() -> str:
    columns = " ".join(f"{metric.name:<{len(metric.name)}}" for metric in METRICS)
    return f"{'case':<{ID_WIDTH}} {'latency':>10}  {columns}"


def _row(report: CaseReport) -> str:
    prefix = f"{report.case_id:<{ID_WIDTH}} {report.latency_ms:>7.0f} ms  "
    if report.error is not None:
        return f"{prefix}ERROR: {report.error}"
    columns = " ".join(f"{_verdict(m.name, report):<{len(m.name)}}" for m in METRICS)
    return prefix + columns


def write_case(report: CaseReport) -> None:
    write(_row(report))
    if report.error is None:
        for result in report.results:
            if not result.passed:
                write(f"    - {result.name}: {result.details}")


def write_summary(report: EvalReport) -> None:
    write()
    write(f"{report.passed_cases}/{len(report.cases)} cases passed all metrics")
    for name, count in report.metric_pass_counts.items():
        write(f"  {name:<22} {count}/{len(report.cases)}")
    write(f"average latency per estimate call: {report.average_turn_latency_ms:.0f} ms")


def _write_json(report: EvalReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None, client: HttpClient | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cases = select_cases(load_dataset(), args.case, args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        write_dry_run(cases, args.mode, args.base_url)
        return EXIT_OK
    http: HttpClient = client if client is not None else requests.Session()
    problem = _check_health(http, args.base_url)
    if problem:
        write(problem)
        return EXIT_USAGE
    write(f"{_plan(cases, args.mode, args.base_url)}: this calls the real LLM")
    write(_header())
    report = evaluate_cases(cases, http, args.base_url, args.mode, args.timeout, write_case)
    write_summary(report)
    if args.output:
        _write_json(report, args.output)
    return EXIT_OK if report.passed_cases == len(cases) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
