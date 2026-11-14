"""Tests for request_id resolution and the correlation middleware."""

import re
from collections.abc import Iterator

import pytest
import structlog
from fastapi import Request
from fastapi.testclient import TestClient
from structlog.contextvars import bind_contextvars, clear_contextvars

from cag_estimate.main import app
from cag_estimate.request_context import (
    REQUEST_ID_HEADER,
    current_request_id,
    resolve_request_id,
)

HEX_ID = re.compile(r"^[0-9a-f]{32}$")
PROBE_PATH = "/_probe"


@pytest.fixture(autouse=True)
def _clean_state():
    clear_contextvars()
    yield
    clear_contextvars()
    structlog.reset_defaults()


# --- resolve_request_id ------------------------------------------------------


@pytest.mark.parametrize("incoming", ["abc", "req-1.2_3", "A" * 64, "a"])
def test_valid_incoming_id_is_accepted(incoming):
    assert resolve_request_id(incoming) == incoming


@pytest.mark.parametrize(
    "incoming", [None, "", "A" * 65, "has space", "semi;colon", "new\nline", "café"]
)
def test_missing_or_invalid_incoming_id_is_replaced(incoming):
    assert HEX_ID.match(resolve_request_id(incoming))


def test_generated_ids_are_unique():
    assert resolve_request_id(None) != resolve_request_id(None)


def test_current_request_id_reads_the_bound_variable():
    assert current_request_id() is None

    bind_contextvars(request_id="abc")

    assert current_request_id() == "abc"


# --- structlog contextvars ---------------------------------------------------


def _capture_events() -> list[dict]:
    """Configure structlog with the default context merging and a capturing sink."""
    events: list[dict] = []

    def capture(_logger, _method, event_dict):
        events.append(dict(event_dict))
        raise structlog.DropEvent

    structlog.configure(processors=[structlog.contextvars.merge_contextvars, capture])
    return events


def test_bound_contextvars_appear_in_log_events():
    events = _capture_events()
    bind_contextvars(request_id="abc")

    structlog.get_logger().info("something_happened")

    assert events[0]["request_id"] == "abc"


def test_default_structlog_processors_merge_contextvars():
    structlog.reset_defaults()

    assert structlog.contextvars.merge_contextvars in structlog.get_config()["processors"]


# --- middleware --------------------------------------------------------------


@pytest.fixture
def client() -> Iterator[TestClient]:
    seen: dict[str, str | None] = {}

    @app.get(PROBE_PATH)
    async def probe(request: Request) -> dict:
        seen["contextvar"] = current_request_id()
        seen["state"] = request.state.request_id
        structlog.get_logger().info("inside_route")
        return {}

    with TestClient(app) as test_client:
        test_client.seen = seen  # type: ignore[attr-defined]
        yield test_client
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") != PROBE_PATH]


def test_valid_incoming_id_is_echoed(client):
    response = client.get("/health", headers={REQUEST_ID_HEADER: "client-id-1"})

    assert response.headers[REQUEST_ID_HEADER] == "client-id-1"


def test_missing_id_generates_a_hex_id(client):
    response = client.get("/health")

    assert HEX_ID.match(response.headers[REQUEST_ID_HEADER])


@pytest.mark.parametrize("bad", ["A" * 65, "bad id!"])
def test_invalid_and_oversized_ids_are_replaced(client, bad):
    response = client.get("/health", headers={REQUEST_ID_HEADER: bad})

    assert HEX_ID.match(response.headers[REQUEST_ID_HEADER])


def test_id_is_bound_and_stored_in_state_during_the_request(client):
    client.get(PROBE_PATH, headers={REQUEST_ID_HEADER: "inside-1"})

    assert client.seen == {"contextvar": "inside-1", "state": "inside-1"}


def test_id_is_cleared_after_the_request(client):
    client.get(PROBE_PATH, headers={REQUEST_ID_HEADER: "inside-1"})

    assert current_request_id() is None


def test_sequential_requests_do_not_share_ids(client):
    first = client.get("/health")
    second = client.get("/health")

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


def test_events_logged_in_a_route_carry_the_request_id(client):
    events = _capture_events()

    client.get(PROBE_PATH, headers={REQUEST_ID_HEADER: "log-1"})

    inside = [e for e in events if e["event"] == "inside_route"]
    assert [e["request_id"] for e in inside] == ["log-1"]
