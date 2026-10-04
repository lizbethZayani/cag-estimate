"""Offline tests for the Streamlit UI helpers (no server, no real HTTP)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import estimate_client
import pytest
import requests
import view_models
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "src" / "ui" / "streamlit_app.py"

RESULT = {
    "project_name": "Shop",
    "meeting_summary": "An online shop.",
    "tasks": [
        {"task_id": 1, "name": "Auth", "description": "d", "estimated_hours": 16,
         "estimated_cost_usd": 640, "complexity": "Medium", "includes": []},
        {"task_id": 2, "name": "Cart", "description": "d", "estimated_hours": 24,
         "estimated_cost_usd": 1200, "complexity": "High", "includes": []},
    ],
    "summary": {"total_hours": 40, "total_cost_usd": 1840, "team_size": "2 developers",
                "estimated_duration_weeks": 3, "hourly_rate": 40, "assumptions": ["a"]},
}


def _response(status: int = 200, body: dict | None = None, lines: list[str] | None = None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body or {}
    response.iter_lines.return_value = lines or []
    response.__enter__.return_value = response
    return response


# ---------- view_models ----------

@pytest.mark.parametrize("reason", ["moderation", "prompt_injection", "pii"])
def test_known_reasons_have_distinct_friendly_messages(reason):
    message = view_models.guardrail_message(reason)
    assert message != view_models.GENERIC_ERROR
    assert reason not in message


def test_unknown_or_missing_reason_falls_back_to_generic():
    assert view_models.guardrail_message("weird") == view_models.GENERIC_ERROR
    assert view_models.guardrail_message(None) == view_models.GENERIC_ERROR


def test_task_rows_have_name_complexity_hours_cost():
    rows = view_models.task_rows(RESULT)
    assert rows[0] == {"Task": "Auth", "Complexity": "Medium", "Hours": 16, "Cost": "$640"}
    assert rows[1]["Cost"] == "$1,200"


def test_totals_are_formatted():
    totals = view_models.totals(RESULT)
    assert totals == {
        "Total hours": "40",
        "Total cost": "$1,840",
        "Hourly rate": "$40/hour",
        "Duration": "3 weeks",
        "Team": "2 developers",
    }


def test_parse_sse_line_token_done_error():
    assert view_models.parse_sse_line('data: {"type": "token", "content": "hi"}') == {
        "type": "token", "content": "hi"}
    assert view_models.parse_sse_line('data: {"type": "done"}') == {"type": "done"}
    assert view_models.parse_sse_line('data: {"type": "error", "message": "m"}')["type"] == "error"


@pytest.mark.parametrize("line", ["", ": keepalive", "data: {not json", "data: [1]"])
def test_parse_sse_line_ignores_noise(line):
    assert view_models.parse_sse_line(line) is None


def test_event_error_message_uses_reason_and_never_raw_text():
    pii = view_models.event_error_message({"type": "error", "reason": "pii", "message": "x"})
    assert pii == view_models.guardrail_message("pii")
    raw = view_models.event_error_message({"type": "error", "message": "Traceback boom"})
    assert raw == view_models.GENERIC_ERROR


def test_status_error_message_maps_400_reason_and_502():
    assert view_models.status_error_message(400, {"reason": "moderation"}) == (
        view_models.guardrail_message("moderation"))
    assert view_models.status_error_message(502, {"detail": "secret"}) == view_models.GENERIC_ERROR
    assert view_models.status_error_message(400, None) == view_models.GENERIC_ERROR


# ---------- estimate_client ----------

def test_api_base_url_reads_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://api:9000")
    assert estimate_client.api_base_url() == "http://api:9000"
    monkeypatch.delenv("API_BASE_URL")
    assert estimate_client.api_base_url() == "http://localhost:8000"


def test_fetch_structured_success():
    body = {"result": RESULT, "prompt_version": "v1", "cached": True}
    with patch("estimate_client.requests.post", return_value=_response(200, body)) as post:
        answer = estimate_client.fetch_structured("desc", 40)
    assert answer.result == RESULT and answer.cached is True and answer.error is None
    assert post.call_args.kwargs["json"] == {"transcription": "desc", "hourly_rate": 40}


def test_fetch_structured_200_without_result_is_generic_error():
    with patch("estimate_client.requests.post", return_value=_response(200, {"cached": True})):
        answer = estimate_client.fetch_structured("d", 40)
    assert answer.result is None and answer.error == view_models.GENERIC_ERROR


def test_fetch_structured_guardrail_400():
    with patch("estimate_client.requests.post", return_value=_response(400, {"reason": "pii"})):
        answer = estimate_client.fetch_structured("desc", 40)
    assert answer.result is None
    assert answer.error == view_models.guardrail_message("pii")


def test_fetch_structured_502_and_connection_error_are_generic():
    with patch("estimate_client.requests.post", return_value=_response(502, {"detail": "x"})):
        assert estimate_client.fetch_structured("d", 40).error == view_models.GENERIC_ERROR
    with patch("estimate_client.requests.post", side_effect=requests.ConnectionError("boom")):
        error = estimate_client.fetch_structured("d", 40).error
    assert error is not None and "boom" not in error


def test_stream_events_yields_parsed_events_and_maps_error_reason():
    lines = [
        'data: {"type": "token", "content": "a"}',
        "",
        'data: {"type": "error", "reason": "prompt_injection", "message": "x"}',
    ]
    with patch("estimate_client.requests.post", return_value=_response(200, lines=lines)):
        events = list(estimate_client.stream_events("d", 40))
    assert events[0] == {"type": "token", "content": "a"}
    assert events[1]["message"] == view_models.guardrail_message("prompt_injection")


def test_stream_events_http_failure_yields_generic_error():
    with patch("estimate_client.requests.post", return_value=_response(500)):
        events = list(estimate_client.stream_events("d", 40))
    assert events == [{"type": "error", "message": view_models.GENERIC_ERROR}]


# ---------- streamlit smoke ----------

def test_structured_mode_renders_table_and_cache_badge():
    answer = estimate_client.StructuredAnswer(result=RESULT, cached=True, error=None)
    with patch("estimate_client.check_api_health", return_value=True), \
            patch("estimate_client.fetch_structured", return_value=answer):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Structured (JSON)").run()
        app.chat_input[0].set_value("Build a shop").run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert any("cache" in c.value.lower() for c in [*app.caption, *app.success, *app.info])
    assert any(m["role"] == "assistant" for m in app.session_state["messages"])
