"""Offline tests for the Streamlit UI helpers (no server, no real HTTP)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import estimate_client
import pytest
import requests
import view_models
from streamlit.testing.v1 import AppTest

REAL_CREATE_SESSION = estimate_client.create_session  # the autouse fixture patches the module attr
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


META = {
    "project_name": "Bookstore",
    "assumed_team_size": 3,
    "mentioned_technologies": ["React", "AWS"],
    "agreed_scope": "MVP shop",
}
SESSION_BODY = {
    "result": RESULT, "prompt_version": "v1", "cached": False, "session_id": "abcdef12-3456",
    "project_metadata": META, "turns": 1, "attachments": ["a.txt"],
}
FILES = [("a.txt", b"hello", "text/plain")]


def _response(status: int = 200, body: dict | None = None, lines: list[str] | None = None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body or {}
    response.iter_lines.return_value = lines or []
    response.__enter__.return_value = response
    return response


@pytest.fixture(autouse=True)
def _no_real_session_calls():
    """Keep every test offline: the default mode creates a session on load."""
    start = estimate_client.SessionStart(session_id="sess-1234-5678")
    with patch("estimate_client.create_session", return_value=start) as create:
        yield create


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


# ---------- view_models: sessions ----------

def test_session_status_messages_are_distinct_and_never_raw():
    codes = [404, 413, 415, 422, 502]
    messages = {code: view_models.session_status_error_message(code, {"detail": "secret"})
                for code in codes}
    assert len(set(messages.values())) == len(codes)
    assert all("secret" not in m for m in messages.values())
    assert messages[404] == view_models.SESSION_EXPIRED_MESSAGE
    assert "expired" in messages[404]


def test_session_status_message_400_uses_guardrail_reason():
    assert view_models.session_status_error_message(400, {"reason": "pii"}) == (
        view_models.guardrail_message("pii"))
    assert view_models.session_status_error_message(500, None) == view_models.GENERIC_ERROR


def test_metadata_rows_show_values_and_placeholders():
    rows = dict(view_models.metadata_rows(META))
    assert rows == {
        "Project": "Bookstore", "Team size": "3",
        "Technologies": "React, AWS", "Scope": "MVP shop",
    }
    empty = dict(view_models.metadata_rows(None))
    assert set(empty.values()) == {view_models.EMPTY_VALUE}
    assert dict(view_models.metadata_rows({"assumed_team_size": None}))["Team size"] == (
        view_models.EMPTY_VALUE)


def test_short_session_id_and_turn_caption_and_user_turn_text():
    assert view_models.short_session_id("abcdef12-3456") == "abcdef12"
    assert view_models.short_session_id(None) == view_models.EMPTY_VALUE
    assert "Turn 3" in view_models.turn_caption(3)
    assert "window" in view_models.turn_caption(3)
    assert view_models.user_turn_text("Hi", []) == "Hi"
    assert view_models.user_turn_text("Hi", ["a.pdf", "b.txt"]) == "Hi\n\n📎 a.pdf, b.txt"


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


# ---------- estimate_client: sessions ----------

def test_create_session_success():
    with patch("estimate_client.requests.post",
               return_value=_response(201, {"session_id": "s-1"})) as post:
        start = REAL_CREATE_SESSION()
    assert start.session_id == "s-1" and start.error is None
    assert post.call_args.args[0].endswith("/api/v1/sessions")


@pytest.mark.parametrize("response", [_response(502, {"detail": "x"}), _response(201, {})])
def test_create_session_failure_is_generic(response):
    with patch("estimate_client.requests.post", return_value=response):
        start = REAL_CREATE_SESSION()
    assert start.session_id is None and start.error == view_models.GENERIC_ERROR


def test_create_session_connection_error():
    with patch("estimate_client.requests.post", side_effect=requests.ConnectionError("boom")):
        start = REAL_CREATE_SESSION()
    assert start.session_id is None and "boom" not in (start.error or "")


def test_get_session_returns_metadata_and_turns():
    body = {"session_id": "s", "project_metadata": META, "turns": 2}
    with patch("estimate_client.requests.get", return_value=_response(200, body)) as get:
        snapshot = estimate_client.get_session("s")
    assert snapshot.project_metadata == META and snapshot.turns == 2 and snapshot.error is None
    assert get.call_args.args[0].endswith("/api/v1/sessions/s")


def test_get_session_404_is_expired():
    with patch("estimate_client.requests.get", return_value=_response(404, {"detail": "x"})):
        snapshot = estimate_client.get_session("s")
    assert snapshot.expired is True and snapshot.error == view_models.SESSION_EXPIRED_MESSAGE


def test_get_session_connection_error():
    with patch("estimate_client.requests.get", side_effect=requests.ConnectionError("boom")):
        snapshot = estimate_client.get_session("s")
    assert snapshot.error == view_models.GENERIC_ERROR and snapshot.expired is False


def test_send_session_estimate_success_builds_multipart_payload():
    with patch("estimate_client.requests.post", return_value=_response(200, SESSION_BODY)) as post:
        answer = estimate_client.send_session_estimate("abc", "build it", 55, FILES)
    assert answer.result == RESULT and answer.error is None and answer.expired is False
    assert answer.project_metadata == META and answer.turns == 1
    assert post.call_args.args[0].endswith("/api/v1/sessions/abc/estimate")
    assert post.call_args.kwargs["data"] == {"transcript": "build it", "hourly_rate": 55}
    assert post.call_args.kwargs["files"] == [("attachments", ("a.txt", b"hello", "text/plain"))]


def test_send_session_estimate_without_files_sends_no_attachments():
    with patch("estimate_client.requests.post", return_value=_response(200, SESSION_BODY)) as post:
        estimate_client.send_session_estimate("abc", "t", 40, [])
    assert post.call_args.kwargs["files"] == []


@pytest.mark.parametrize("status", [413, 415, 422, 502])
def test_send_session_estimate_error_statuses_use_safe_messages(status):
    with patch("estimate_client.requests.post",
               return_value=_response(status, {"detail": "raw server text"})):
        answer = estimate_client.send_session_estimate("abc", "t", 40, [])
    assert answer.result is None and answer.expired is False
    assert answer.error == view_models.session_status_error_message(status, None)


def test_send_session_estimate_404_marks_session_expired():
    with patch("estimate_client.requests.post", return_value=_response(404, {"detail": "x"})):
        answer = estimate_client.send_session_estimate("abc", "t", 40, [])
    assert answer.expired is True and answer.error == view_models.SESSION_EXPIRED_MESSAGE


def test_send_session_estimate_400_guardrail_reason():
    with patch("estimate_client.requests.post", return_value=_response(400, {"reason": "pii"})):
        answer = estimate_client.send_session_estimate("abc", "t", 40, [])
    assert answer.error == view_models.guardrail_message("pii")


def test_send_session_estimate_connection_error_and_missing_result():
    with patch("estimate_client.requests.post", side_effect=requests.ConnectionError("boom")):
        answer = estimate_client.send_session_estimate("abc", "t", 40, [])
    assert answer.result is None and "boom" not in (answer.error or "")
    with patch("estimate_client.requests.post", return_value=_response(200, {"turns": 1})):
        answer = estimate_client.send_session_estimate("abc", "t", 40, [])
    assert answer.result is None and answer.error == view_models.GENERIC_ERROR


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


# ---------- streamlit: conversation mode ----------

def _session_answer(**overrides):
    fields = {"result": RESULT, "project_metadata": META, "turns": 1}
    return estimate_client.SessionAnswer(**{**fields, **overrides})


def _send_button(app):
    return next(b for b in app.button if b.label == "Send")


def _new_conversation_button(app):
    return next(b for b in app.sidebar.button if "New conversation" in b.label)


def _conversation_app():
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_conversation_is_default_mode_and_creates_session_on_load(_no_real_session_calls):
    app = _conversation_app()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Conversation (memory)"
    assert app.session_state["session_id"] == "sess-1234-5678"
    _no_real_session_calls.assert_called_once()


def test_conversation_send_shows_table_and_updates_metadata_panel():
    with patch("estimate_client.send_session_estimate", return_value=_session_answer()) as send:
        app = _conversation_app()
        app.text_area[0].set_value("Bookstore with React").run()
        _send_button(app).click().run()
    assert not app.exception
    assert send.call_args.args[:3] == ("sess-1234-5678", "Bookstore with React", 40)
    assert len(app.dataframe) == 1
    assert app.session_state["turns"] == 1
    assert app.session_state["project_metadata"]["project_name"] == "Bookstore"
    panel = " ".join(m.value for m in [*app.sidebar.markdown, *app.sidebar.caption])
    assert "Bookstore" in panel and "React, AWS" in panel and "Session sess" in panel and "1 turn" in panel


def test_empty_message_is_not_sent():
    with patch("estimate_client.send_session_estimate") as send:
        app = _conversation_app()
        _send_button(app).click().run()
    send.assert_not_called()
    assert not app.exception


def test_new_conversation_resets_messages_metadata_and_creates_session(_no_real_session_calls):
    with patch("estimate_client.send_session_estimate", return_value=_session_answer()):
        app = _conversation_app()
        app.text_area[0].set_value("Bookstore").run()
        _send_button(app).click().run()
        _new_conversation_button(app).click().run()
    assert _no_real_session_calls.call_count == 2
    assert app.session_state["conversation_messages"] == []
    assert app.session_state["project_metadata"] is None
    assert app.session_state["turns"] == 0
    assert len(app.dataframe) == 0


def test_expired_session_creates_a_fresh_one_without_resending(_no_real_session_calls):
    expired = _session_answer(
        result=None, project_metadata=None, turns=0, expired=True,
        error=view_models.SESSION_EXPIRED_MESSAGE)
    with patch("estimate_client.send_session_estimate", return_value=expired) as send:
        app = _conversation_app()
        app.text_area[0].set_value("Bookstore").run()
        _send_button(app).click().run()
    assert send.call_count == 1
    assert _no_real_session_calls.call_count == 2
    assert any(view_models.SESSION_EXPIRED_MESSAGE in e.value for e in app.error)
    assert len(app.dataframe) == 0
    assert not any(m.get("kind") == "structured" for m in app.session_state["conversation_messages"])


def test_failed_turn_adds_no_assistant_answer():
    failed = _session_answer(result=None, project_metadata=None, turns=0, error="Too big.")
    with patch("estimate_client.send_session_estimate", return_value=failed):
        app = _conversation_app()
        app.text_area[0].set_value("Bookstore").run()
        _send_button(app).click().run()
    assert any("Too big." in e.value for e in app.error)
    assert app.session_state["project_metadata"] is None


def test_session_creation_failure_keeps_the_ui_usable(_no_real_session_calls):
    _no_real_session_calls.return_value = estimate_client.SessionStart(
        session_id=None, error=view_models.GENERIC_ERROR)
    app = _conversation_app()
    assert not app.exception
    assert any(view_models.GENERIC_ERROR in e.value for e in app.error)
    assert len(app.text_area) == 1
