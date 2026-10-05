"""HTTP client for the estimation API. Errors are mapped to safe UI messages."""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests
import view_models

DEFAULT_API_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class StructuredAnswer:
    """Outcome of a structured call: either a result dict or a safe error message."""

    result: dict[str, Any] | None
    cached: bool = False
    error: str | None = None


@dataclass(frozen=True)
class SessionStart:
    """Outcome of creating a session: an id or a safe error message."""

    session_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SessionSnapshot:
    """Memory state of a session; ``expired`` is set when the API no longer knows it."""

    project_metadata: dict[str, Any] | None = None
    turns: int = 0
    error: str | None = None
    expired: bool = False


@dataclass(frozen=True)
class SessionAnswer(SessionSnapshot):
    """Outcome of a session estimate: result plus fresh memory, or a safe error."""

    result: dict[str, Any] | None = None


UploadFile = tuple[str, bytes, str]


def api_base_url() -> str:
    """Single place where the API base URL is read."""
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)


def check_api_health() -> bool:
    try:
        return requests.get(f"{api_base_url()}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def _json_body(response: requests.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def fetch_structured(transcription: str, hourly_rate: int) -> StructuredAnswer:
    """Call ``POST /api/v1/estimate`` and return the result or a safe error."""
    payload = {"transcription": transcription, "hourly_rate": hourly_rate}
    try:
        response = requests.post(
            f"{api_base_url()}/api/v1/estimate", json=payload, timeout=TIMEOUT_SECONDS
        )
    except requests.RequestException:
        return StructuredAnswer(result=None, error=view_models.GENERIC_ERROR)
    body = _json_body(response)
    if response.status_code != 200 or body is None:
        return StructuredAnswer(
            result=None, error=view_models.status_error_message(response.status_code, body)
        )
    if not isinstance(body.get("result"), dict):
        return StructuredAnswer(result=None, error=view_models.GENERIC_ERROR)
    return StructuredAnswer(result=body["result"], cached=bool(body.get("cached")))


def _sessions_url() -> str:
    return f"{api_base_url()}/api/v1/sessions"


def _failure(response: requests.Response) -> dict[str, Any]:
    """Safe ``error``/``expired`` fields for a non-success session response."""
    status = response.status_code
    return {
        "error": view_models.session_status_error_message(status, _json_body(response)),
        "expired": status == 404,
    }


def create_session() -> SessionStart:
    """Call ``POST /api/v1/sessions`` and return the new id or a safe error."""
    try:
        response = requests.post(_sessions_url(), timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return SessionStart(error=view_models.GENERIC_ERROR)
    body = _json_body(response) or {}
    if response.status_code != 201 or not body.get("session_id"):
        return SessionStart(error=view_models.GENERIC_ERROR)
    return SessionStart(session_id=body["session_id"])


def get_session(session_id: str) -> SessionSnapshot:
    """Call ``GET /api/v1/sessions/{id}`` for the current project memory."""
    try:
        response = requests.get(f"{_sessions_url()}/{session_id}", timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return SessionSnapshot(error=view_models.GENERIC_ERROR)
    body = _json_body(response)
    if response.status_code != 200 or body is None:
        return SessionSnapshot(**_failure(response))
    return SessionSnapshot(project_metadata=body.get("project_metadata"), turns=body.get("turns", 0))


def send_session_estimate(
    session_id: str, transcript: str, hourly_rate: int, files: list[UploadFile]
) -> SessionAnswer:
    """Call ``POST /api/v1/sessions/{id}/estimate`` (multipart) and return the answer or error."""
    multipart = [("attachments", (name, content, mime)) for name, content, mime in files]
    try:
        response = requests.post(
            f"{_sessions_url()}/{session_id}/estimate",
            data={"transcript": transcript, "hourly_rate": hourly_rate},
            files=multipart,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return SessionAnswer(error=view_models.GENERIC_ERROR)
    body = _json_body(response)
    if response.status_code != 200 or body is None:
        return SessionAnswer(**_failure(response))
    if not isinstance(body.get("result"), dict):
        return SessionAnswer(error=view_models.GENERIC_ERROR)
    return SessionAnswer(
        result=body["result"],
        project_metadata=body.get("project_metadata"),
        turns=body.get("turns", 0),
    )


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") == "error":
        return {"type": "error", "message": view_models.event_error_message(event)}
    return event


def stream_events(transcription: str, hourly_rate: int) -> Iterator[dict[str, Any]]:
    """Yield SSE events from ``POST /api/v1/estimate/stream``; errors carry safe text."""
    payload = {"transcription": transcription, "hourly_rate": hourly_rate}
    try:
        with requests.post(
            f"{api_base_url()}/api/v1/estimate/stream",
            json=payload,
            stream=True,
            timeout=TIMEOUT_SECONDS,
        ) as response:
            if response.status_code != 200:
                yield {"type": "error", "message": view_models.GENERIC_ERROR}
                return
            for line in response.iter_lines(decode_unicode=True):
                event = view_models.parse_sse_line(line)
                if event is not None:
                    yield _safe_event(event)
    except requests.RequestException:
        yield {"type": "error", "message": view_models.GENERIC_ERROR}
