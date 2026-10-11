"""Shared offline helpers for the eval tests: bodies, cases and a scripted HTTP client."""

from typing import Any, NamedTuple

from cag_estimate.evals.dataset import GoldenCase

TURN = "We need a small bookstore web shop with a catalog and checkout."
SESSION_ID = "session-1"


def make_case(**overrides: Any) -> GoldenCase:
    return GoldenCase.model_validate(
        {"id": "case-one", "description": "d", "turns": [TURN], **overrides}
    )


def make_body(turns: int = 1, hours: int = 30, name: str = "Bookstore", **extra: Any) -> dict:
    """A response body in the shape of ``SessionEstimationResponse`` (3 tasks, 40 USD/h)."""
    tasks = [
        {
            "task_id": n,
            "name": f"Catalog task {n}",
            "description": "Build the Django catalog",
            "estimated_hours": hours // 3,
            "estimated_cost_usd": hours // 3 * 40,
            "complexity": "Medium",
            "includes": ["code"],
        }
        for n in range(1, 4)
    ]
    body = {
        "result": {
            "project_name": name,
            "meeting_summary": "A web shop for books",
            "tasks": tasks,
            "summary": {
                "total_hours": hours // 3 * 3,
                "total_cost_usd": hours // 3 * 3 * 40,
                "hourly_rate": 40,
            },
        },
        "prompt_version": "v2",
        "turns": turns,
        "resolved_tier": "default",
        "project_metadata": {"mentioned_technologies": ["Stripe"]},
    }
    body.update(extra)
    return body


class FakeResponse:
    """The only response shape the runner relies on: ``status_code`` and ``json()``."""

    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self.body = body

    def json(self) -> Any:
        if self.body is None:
            raise ValueError("no JSON body")
        return self.body


class Call(NamedTuple):
    method: str
    url: str
    data: dict[str, str] | None
    timeout: float | None


class ScriptedClient:
    """An ``HttpClient`` that records calls and replays scripted estimate replies in order."""

    def __init__(
        self,
        replies: list[FakeResponse | Exception] | None = None,
        create_status: int = 201,
        health_status: int = 200,
        delete_error: Exception | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.create_status = create_status
        self.health_status = health_status
        self.delete_error = delete_error
        self.calls: list[Call] = []

    def post(self, url: str, data: Any = None, files: Any = None, timeout: Any = None) -> Any:
        self.calls.append(Call("POST", url, data, timeout))
        if url.endswith("/api/v1/sessions"):
            return FakeResponse(self.create_status, {"session_id": SESSION_ID})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url: str, timeout: Any = None) -> Any:
        self.calls.append(Call("GET", url, None, timeout))
        return FakeResponse(self.health_status, {"status": "ok"})

    def delete(self, url: str, timeout: Any = None) -> Any:
        self.calls.append(Call("DELETE", url, None, timeout))
        if self.delete_error is not None:
            raise self.delete_error
        return FakeResponse(204)

    def urls(self, method: str) -> list[str]:
        return [call.url for call in self.calls if call.method == method]


def ok_reply(turns: int = 1, **kwargs: Any) -> FakeResponse:
    return FakeResponse(200, make_body(turns=turns, **kwargs))


def rejection_reply(reason: str = "prompt_injection") -> FakeResponse:
    return FakeResponse(400, {"reason": reason, "message": "The input was rejected."})
