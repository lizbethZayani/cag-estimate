"""Offline tests for the eval runner: scripted client, report, and an end-to-end run on the app."""

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from cag_estimate.dependencies import get_session_estimation_service, get_session_store
from cag_estimate.evals.dataset import load_dataset
from cag_estimate.evals.metrics import CaseOutcome, TurnResult
from cag_estimate.evals.runner import build_report, evaluate_cases, run_case
from cag_estimate.main import app
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import SessionStore
from tests.acb_fakes import ACBWrapper, accept
from tests.eval_fakes import (
    SESSION_ID,
    FakeResponse,
    ScriptedClient,
    make_case,
    ok_reply,
)
from tests.test_sessions_endpoints import FakeWrapper

BASE = "http://api.test"
MEMORY_TURNS = [
    "We need a small bookstore web shop with a catalog and checkout.",
    "Add a customer reviews feature for every book page of the shop.",
]


def test_each_turn_goes_to_one_session_in_order_and_the_session_is_deleted() -> None:
    client = ScriptedClient([ok_reply(turns=1), ok_reply(turns=2)])
    case = make_case(turns=MEMORY_TURNS, check_memory=True)

    outcome = run_case(case, client, BASE)

    assert outcome.error is None and [t.body["turns"] for t in outcome.turns] == [1, 2]
    assert client.urls("POST") == [
        f"{BASE}/api/v1/sessions",
        f"{BASE}/api/v1/sessions/{SESSION_ID}/estimate",
        f"{BASE}/api/v1/sessions/{SESSION_ID}/estimate",
    ]
    assert [call.data["transcript"] for call in client.calls[1:3]] == MEMORY_TURNS
    assert client.urls("DELETE") == [f"{BASE}/api/v1/sessions/{SESSION_ID}"]


def test_a_trailing_slash_in_the_base_url_is_ignored() -> None:
    client = ScriptedClient([ok_reply()])
    run_case(make_case(), client, BASE + "/")
    assert client.urls("POST")[0] == f"{BASE}/api/v1/sessions"


def test_hourly_rate_is_sent_and_tier_only_when_not_auto() -> None:
    client = ScriptedClient([ok_reply(), ok_reply()])
    run_case(make_case(hourly_rate=75), client, BASE)
    run_case(make_case(hourly_rate=75, tier="pm"), client, BASE)

    auto_form, forced_form = [c.data for c in client.calls if c.url.endswith("/estimate")]
    assert auto_form == {"transcript": make_case().turns[0], "hourly_rate": "75"}
    assert forced_form["tier"] == "pm" and forced_form["hourly_rate"] == "75"


def test_acb_mode_uses_the_reviewed_endpoint() -> None:
    client = ScriptedClient([ok_reply()])
    run_case(make_case(), client, BASE, mode="acb")
    assert client.urls("POST")[1].endswith(f"/sessions/{SESSION_ID}/estimate-acb")


def test_a_transport_error_becomes_an_errored_case_and_still_deletes_the_session() -> None:
    client = ScriptedClient([ConnectionError("refused")])

    outcome = run_case(make_case(), client, BASE)

    assert outcome.error is not None and "ConnectionError" in outcome.error
    assert client.urls("DELETE") == [f"{BASE}/api/v1/sessions/{SESSION_ID}"]


def test_the_turns_done_before_a_transport_error_are_kept() -> None:
    client = ScriptedClient([ok_reply(), TimeoutError("slow")])
    outcome = run_case(make_case(turns=MEMORY_TURNS), client, BASE)
    assert len(outcome.turns) == 1 and "TimeoutError" in (outcome.error or "")


def test_a_failed_session_creation_is_an_errored_case_without_delete() -> None:
    client = ScriptedClient(create_status=503)

    outcome = run_case(make_case(), client, BASE)

    assert outcome.error is not None and "503" in outcome.error
    assert client.urls("DELETE") == []


def test_a_failing_delete_does_not_change_the_outcome() -> None:
    client = ScriptedClient([ok_reply()], delete_error=ConnectionError("gone"))
    outcome = run_case(make_case(), client, BASE)
    assert outcome.error is None and len(outcome.turns) == 1


def test_the_run_stops_at_the_first_non_200_turn() -> None:
    client = ScriptedClient([FakeResponse(502, {"detail": "boom"})])
    outcome = run_case(make_case(turns=MEMORY_TURNS), client, BASE)
    assert [t.status_code for t in outcome.turns] == [502] and outcome.error is None


def test_a_non_json_body_is_recorded_as_an_empty_body() -> None:
    client = ScriptedClient([FakeResponse(502, None)])
    outcome = run_case(make_case(), client, BASE)
    assert outcome.turns[0].body == {}


def test_latency_is_recorded_per_turn() -> None:
    outcome = run_case(make_case(), ScriptedClient([ok_reply()]), BASE)
    assert outcome.turns[0].latency_ms >= 0.0


# --- report ------------------------------------------------------------------


def _outcomes() -> list[CaseOutcome]:
    good = run_case(make_case(id="good-case"), ScriptedClient([ok_reply()]), BASE)
    bad = CaseOutcome(case=make_case(id="bad-case"), error="ConnectionError: refused")
    slow = CaseOutcome(
        case=make_case(id="slow-case"),
        turns=[TurnResult(200, ok_reply().body, 100.0), TurnResult(200, ok_reply().body, 300.0)],
    )
    return [good, bad, slow]


def test_report_totals_and_serialization() -> None:
    report = build_report(_outcomes(), mode="standard", base_url=BASE)
    data = report.to_dict()

    assert data["mode"] == "standard" and data["base_url"] == BASE
    assert data["total_cases"] == 3 and data["passed_cases"] == 1 and data["failed_cases"] == 2
    assert data["metric_pass_counts"]["schema_consistency"] == 1
    assert set(data["metric_pass_counts"]) == {
        "schema_consistency",
        "range_bounds",
        "content_recall",
        "guardrail_behavior",
        "tier_expectation",
        "memory_consistency",
    }
    assert [c["id"] for c in data["cases"]] == ["good-case", "bad-case", "slow-case"]
    assert data["cases"][1]["error"] == "ConnectionError: refused"
    assert data["cases"][2]["latency_ms"] == 400.0
    assert data["cases"][0]["metrics"][0]["name"] == "schema_consistency"
    assert data["average_turn_latency_ms"] == pytest.approx(
        (data["cases"][0]["latency_ms"] + 400) / 3
    )


def test_an_empty_report_has_zero_latency_and_no_failures() -> None:
    data = build_report([], mode="acb", base_url=BASE).to_dict()
    assert data["total_cases"] == 0 and data["average_turn_latency_ms"] == 0.0


def test_evaluate_cases_reports_each_case_as_it_finishes() -> None:
    client = ScriptedClient([ok_reply(), ok_reply()])
    cases = [make_case(id="first-case"), make_case(id="second-case")]
    seen: list[str] = []

    report = evaluate_cases(
        cases, client, BASE, "standard", on_case=lambda c: seen.append(c.case_id)
    )

    assert seen == ["first-case", "second-case"] and report.passed_cases == 2


# --- end to end on the real FastAPI app --------------------------------------


@pytest.fixture
def api() -> Iterator[TestClient]:
    store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_session_estimation_service] = lambda: SessionEstimationService(
        FakeWrapper()
    )
    yield TestClient(app, base_url="http://testserver")
    app.dependency_overrides.clear()


def _dataset_case(case_id: str):
    return next(case for case in load_dataset() if case.id == case_id)


def test_end_to_end_guardrail_case_passes_through_the_real_guardrail(api: TestClient) -> None:
    outcome = run_case(_dataset_case("guardrail-prompt-injection"), api, "http://testserver")
    assert outcome.turns[0].status_code == 400
    report = build_report([outcome], mode="standard", base_url="http://testserver")
    assert report.passed_cases == 1


def test_end_to_end_pii_case_passes_through_the_real_guardrail(api: TestClient) -> None:
    outcome = run_case(_dataset_case("guardrail-pii-email"), api, "http://testserver")
    assert build_report([outcome], mode="standard", base_url="x").passed_cases == 1


def test_end_to_end_single_turn_case(api: TestClient) -> None:
    case = make_case(expected_task_count_range=[3, 5], expected_hours_range=[20, 40])
    outcome = run_case(case, api, "http://testserver")
    assert outcome.error is None and outcome.turns[0].status_code == 200
    assert build_report([outcome], mode="standard", base_url="x").passed_cases == 1


def test_end_to_end_memory_case(api: TestClient) -> None:
    case = make_case(turns=MEMORY_TURNS, check_memory=True)
    outcome = run_case(case, api, "http://testserver")
    assert [t.body["turns"] for t in outcome.turns] == [1, 2]
    assert build_report([outcome], mode="standard", base_url="x").passed_cases == 1


def test_end_to_end_acb_mode(api: TestClient) -> None:
    app.dependency_overrides[get_session_estimation_service] = lambda: SessionEstimationService(
        ACBWrapper([accept()] * 2), max_iterations=2
    )
    outcome = run_case(make_case(), api, "http://testserver", mode="acb")
    assert outcome.turns[0].status_code == 200 and "acb" in outcome.turns[0].body


def test_end_to_end_sessions_are_deleted(api: TestClient) -> None:
    run_case(make_case(), api, "http://testserver")
    store = app.dependency_overrides[get_session_store]()
    assert len(store) == 0
