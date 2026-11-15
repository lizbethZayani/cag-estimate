"""Tests for the retrieval evaluation logic: precision@k, aggregation, tables, golden set."""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from cag_estimate.evals.retrieval_eval import (
    CONFIGS,
    GOLDEN_SET_PATH,
    GoldenQuery,
    GoldenSet,
    QueryMeasurement,
    load_golden_set,
    mean_precision,
    median_latency_ms,
    precision_at_k,
    render_markdown_table,
    require_ground_truth,
)

BUDGETS_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"


def _query(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q1",
        "query": "text",
        "relevant_budget_ids": ["BUD-2024-001"],
        "rationale": "why",
    }
    return {**base, **overrides}


def test_precision_counts_relevant_share_of_first_k() -> None:
    returned = ["A", "B", "C", "D", "E"]

    assert precision_at_k(returned, {"A", "C"}) == pytest.approx(0.4)


def test_precision_all_relevant_is_one() -> None:
    assert precision_at_k(["A", "A", "B", "B", "A"], {"A", "B"}) == 1.0


def test_precision_none_relevant_is_zero() -> None:
    assert precision_at_k(["X", "Y", "Z", "X", "Y"], {"A"}) == 0.0


def test_precision_missing_slots_count_as_not_relevant() -> None:
    assert precision_at_k(["A", "B"], {"A", "B"}, k=5) == pytest.approx(0.4)


def test_precision_empty_results_is_zero() -> None:
    assert precision_at_k([], {"A"}) == 0.0


def test_precision_duplicate_budgets_fill_several_slots() -> None:
    assert precision_at_k(["A", "A", "A", "B", "C"], {"A"}) == pytest.approx(0.6)


def test_precision_only_looks_at_the_first_k() -> None:
    assert precision_at_k(["X", "X", "A", "A", "A", "A"], {"A"}, k=3) == pytest.approx(1 / 3)


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejects_k_below_one(k: int) -> None:
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=k)


def test_median_latency_odd_and_even() -> None:
    assert median_latency_ms([30.0, 10.0, 20.0]) == 20.0
    assert median_latency_ms([10.0, 20.0, 30.0, 40.0]) == 25.0


def test_median_latency_of_nothing_raises() -> None:
    with pytest.raises(ValueError):
        median_latency_ms([])


def test_configs_are_exactly_a_to_d() -> None:
    assert [(c.name, c.search_mode, c.rerank) for c in CONFIGS] == [
        ("A", "vector", False),
        ("B", "hybrid", False),
        ("C", "vector", True),
        ("D", "hybrid", True),
    ]


def _measurement(config: str, query_id: str, precision: float) -> QueryMeasurement:
    return QueryMeasurement(
        config=config,
        query_id=query_id,
        precision=precision,
        latencies_ms=(10.0, 30.0),
        server_times_ms=(4, 8),
    )


def _all_measurements() -> list[QueryMeasurement]:
    return [
        _measurement(config.name, query_id, precision)
        for config, precision in zip(CONFIGS, (0.2, 0.4, 0.6, 0.8), strict=True)
        for query_id in ("q1", "q2")
    ]


def test_mean_precision_averages_one_config_over_queries() -> None:
    measurements = [_measurement("A", "q1", 0.2), _measurement("A", "q2", 0.6)]
    measurements.append(_measurement("B", "q1", 1.0))

    assert mean_precision(measurements, "A") == pytest.approx(0.4)


def test_comparison_table_has_headers_and_one_row_per_config() -> None:
    table = render_markdown_table(_all_measurements())

    assert (
        "| Config | Search | Reranking | Precision@5 (mean) | Latency median (ms) "
        "| Server search_time median (ms) |"
    ) in table
    rows = [
        line for line in table.splitlines() if line.startswith(("| A ", "| B ", "| C ", "| D "))
    ]
    assert len(rows) == 4
    assert "| A | vector | off | 0.20 | 20.0 | 6.0 |" in table
    assert "| D | hybrid | on | 0.80 | 20.0 | 6.0 |" in table


def test_per_query_table_has_a_row_per_query_and_a_column_per_config() -> None:
    table = render_markdown_table(_all_measurements())

    assert "| Query | A | B | C | D |" in table
    assert "| q1 | 0.20 | 0.40 | 0.60 | 0.80 |" in table
    assert "| q2 | 0.20 | 0.40 | 0.60 | 0.80 |" in table


def test_golden_query_accepts_a_valid_record() -> None:
    golden = GoldenQuery.model_validate(
        _query(relevant_budget_ids=["BUD-2024-005", "BUD-2024-006"])
    )

    assert golden.relevant_budget_ids == ["BUD-2024-005", "BUD-2024-006"]


@pytest.mark.parametrize("bad_id", ["BUD-24-001", "bud-2024-001", "BUD-2024-1", "X"])
def test_golden_query_rejects_a_bad_budget_id_format(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        GoldenQuery.model_validate(_query(relevant_budget_ids=[bad_id]))


def test_golden_query_rejects_an_empty_relevant_list() -> None:
    with pytest.raises(ValidationError):
        GoldenQuery.model_validate(_query(relevant_budget_ids=[]))


def test_golden_query_rejects_duplicate_relevant_ids() -> None:
    with pytest.raises(ValidationError):
        GoldenQuery.model_validate(_query(relevant_budget_ids=["BUD-2024-001", "BUD-2024-001"]))


def test_load_golden_set_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"version": 1, "queries": [_query(), _query()]}))

    with pytest.raises(ValidationError):
        load_golden_set(path)


def test_load_golden_set_reads_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"version": 1, "queries": [_query()]}))

    golden = load_golden_set(path)

    assert golden.version == 1
    assert [q.id for q in golden.queries] == ["q1"]


def test_packaged_golden_set_has_five_queries_with_existing_budgets() -> None:
    golden = load_golden_set(GOLDEN_SET_PATH)
    known = {b["budget_id"] for b in json.loads(BUDGETS_SAMPLE.read_text(encoding="utf-8"))}

    assert len(golden.queries) == 5
    for query in golden.queries:
        assert set(query.relevant_budget_ids) <= known, query.id


def test_golden_query_accepts_a_record_without_ground_truth() -> None:
    assert GoldenQuery.model_validate(_query()).ground_truth is None


def test_golden_query_accepts_a_record_with_ground_truth() -> None:
    golden = GoldenQuery.model_validate(_query(ground_truth="Total: 10 hours."))

    assert golden.ground_truth == "Total: 10 hours."


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_golden_query_rejects_a_blank_ground_truth(blank: str) -> None:
    with pytest.raises(ValidationError):
        GoldenQuery.model_validate(_query(ground_truth=blank))


def test_load_golden_set_reads_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps({"version": 1, "queries": [_query(ground_truth="ref"), _query(id="q2")]})
    )

    golden = load_golden_set(path)

    assert [q.ground_truth for q in golden.queries] == ["ref", None]


def test_require_ground_truth_returns_all_queries_when_complete() -> None:
    golden = GoldenSet.model_validate(
        {
            "version": 1,
            "queries": [_query(ground_truth="a"), _query(id="q2", ground_truth="b")],
        }
    )

    assert [q.id for q in require_ground_truth(golden)] == ["q1", "q2"]


def test_require_ground_truth_names_the_missing_query_ids() -> None:
    golden = GoldenSet.model_validate(
        {
            "version": 1,
            "queries": [
                _query(ground_truth="a"),
                _query(id="q2"),
                _query(id="q3"),
            ],
        }
    )

    with pytest.raises(ValueError, match="q2.*q3"):
        require_ground_truth(golden)


def test_packaged_golden_set_has_ground_truth_on_every_query() -> None:
    golden = load_golden_set(GOLDEN_SET_PATH)

    assert len(require_ground_truth(golden)) == 5
    for query in golden.queries:
        assert query.ground_truth is not None and query.ground_truth.strip(), query.id


_BULLET = re.compile(
    r"^- .+: (?P<hours>\d+) hours \((?P<budget>BUD-\d{4}-\d{3})::(?P<component>[A-Z0-9-]+)\)$"
)
_TOTAL = re.compile(r"^Total: (?P<total>\d+) hours\.$")


def test_packaged_ground_truths_match_the_historical_budgets() -> None:
    golden = load_golden_set(GOLDEN_SET_PATH)
    hours_by_component = {
        (b["budget_id"], c["component_id"]): c["estimated_hours"]
        for b in json.loads(BUDGETS_SAMPLE.read_text(encoding="utf-8"))
        for c in b["components"]
    }

    for query in golden.queries:
        assert query.ground_truth is not None
        lines = query.ground_truth.split("\n")
        bullets = [m for line in lines if (m := _BULLET.match(line))]
        total = _TOTAL.match(lines[-1])
        assert bullets, query.id
        assert total is not None, query.id
        cited_budgets = set()
        for bullet in bullets:
            key = (bullet["budget"], bullet["component"])
            assert key in hours_by_component, (query.id, key)
            assert int(bullet["hours"]) == hours_by_component[key], (query.id, key)
            cited_budgets.add(bullet["budget"])
        assert int(total["total"]) == sum(int(b["hours"]) for b in bullets), query.id
        assert set(query.relevant_budget_ids) <= cited_budgets, query.id
