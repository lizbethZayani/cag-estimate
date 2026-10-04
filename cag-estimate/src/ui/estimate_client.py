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
