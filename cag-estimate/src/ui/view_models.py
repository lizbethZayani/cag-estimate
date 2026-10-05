"""Pure presentation logic for the Streamlit UI (no Streamlit, no HTTP)."""

import json
from typing import Any

GENERIC_ERROR = "The estimation service is temporarily unavailable. Please try again."

GUARDRAIL_MESSAGES = {
    "moderation": "Your description was flagged as inappropriate content. Please rephrase it.",
    "prompt_injection": (
        "Your message looks like instructions to the assistant. "
        "Please describe the project only."
    ),
    "pii": "Your description seems to contain personal data. Please remove it and try again.",
}

SESSION_EXPIRED_MESSAGE = "The session expired (the API restarted). Starting a new conversation."
SESSION_STATUS_MESSAGES = {
    404: SESSION_EXPIRED_MESSAGE,
    413: "The attachments are too large or too many. Use up to 5 files of 5 MB each.",
    415: "One of the attachments has an unsupported type. Use pdf, txt, md, csv or json.",
    422: "The message or an attachment could not be read. Check them and try again.",
}
EMPTY_VALUE = "—"


def guardrail_message(reason: str | None) -> str:
    """Friendly message for a guardrail reason; generic for unknown reasons."""
    return GUARDRAIL_MESSAGES.get(reason or "", GENERIC_ERROR)


def event_error_message(event: dict[str, Any]) -> str:
    """Safe text for an SSE ``error`` event. The raw server message is never shown."""
    return guardrail_message(event.get("reason"))


def status_error_message(status_code: int, body: dict[str, Any] | None) -> str:
    """Safe text for a non-2xx response: guardrail message on 400, generic otherwise."""
    if status_code == 400 and body:
        return guardrail_message(body.get("reason"))
    return GENERIC_ERROR


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one SSE ``data:`` line into an event dict; None for anything else."""
    if not line.startswith("data: "):
        return None
    try:
        event = json.loads(line[len("data: "):])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def task_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Table rows (name, complexity, hours, cost) from a ProjectEstimation dict."""
    return [
        {
            "Task": task["name"],
            "Complexity": task["complexity"],
            "Hours": task["estimated_hours"],
            "Cost": f"${task['estimated_cost_usd']:,}",
        }
        for task in result["tasks"]
    ]


def totals(result: dict[str, Any]) -> dict[str, str]:
    """Formatted totals and hourly rate from a ProjectEstimation dict."""
    summary = result["summary"]
    return {
        "Total hours": f"{summary['total_hours']:,}",
        "Total cost": f"${summary['total_cost_usd']:,}",
        "Hourly rate": f"${summary['hourly_rate']}/hour",
        "Duration": f"{summary['estimated_duration_weeks']} weeks",
        "Team": summary["team_size"],
    }


def session_status_error_message(status_code: int, body: dict[str, Any] | None) -> str:
    """Safe text for a non-2xx session response; guardrail reason on 400."""
    if status_code == 400:
        return status_error_message(status_code, body)
    return SESSION_STATUS_MESSAGES.get(status_code, GENERIC_ERROR)


def metadata_rows(metadata: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Label/value rows for the project memory panel; placeholders for missing values."""
    data = metadata or {}
    team_size = data.get("assumed_team_size")
    return [
        ("Project", data.get("project_name") or EMPTY_VALUE),
        ("Team size", str(team_size) if team_size is not None else EMPTY_VALUE),
        ("Technologies", ", ".join(data.get("mentioned_technologies") or []) or EMPTY_VALUE),
        ("Scope", data.get("agreed_scope") or EMPTY_VALUE),
    ]


def short_session_id(session_id: str | None) -> str:
    """First segment of the session UUID, for display."""
    return session_id.split("-")[0] if session_id else EMPTY_VALUE


def turn_caption(turns: int) -> str:
    """Caption under an answer: turn number and what the model remembers."""
    return f"Turn {turns} · the model sees the project memory plus a sliding window of recent turns"


def user_turn_text(text: str, file_names: list[str]) -> str:
    """User bubble text with the attached file names appended."""
    if not file_names:
        return text
    return f"{text}\n\n📎 {', '.join(file_names)}"
