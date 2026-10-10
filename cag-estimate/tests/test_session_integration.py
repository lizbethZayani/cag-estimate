"""Step 7 integration tests: memory, attachments and sliding window over HTTP.

Everything is real (SessionStore, SessionEstimationService, prompt rendering,
metadata extractor, attachment extraction) except the LLM wrapper, which is a
deterministic fake that records each call and derives its answer from its inputs.
"""

import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from cag_estimate.dependencies import get_session_estimation_service, get_session_store
from cag_estimate.main import app
from cag_estimate.prompts import render_system_prompt
from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import SessionStore
from tests.conftest import build_estimation
from tests.pdf_helpers import build_pdf

MAX_TURNS = 6
HOURS_PER_TASK = 10
ATTACHMENT_SEPARATOR = "--- attachment: spec.pdf ---"


class RecordingWrapper:
    """Fake LLM: records every call and answers from what it was given.

    Like a real model it takes the project name from the transcript when stated
    and otherwise from the ``<project_metadata>`` block of the system prompt.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[ProjectEstimation, dict[str, Any]]:
        self.calls.append({**kwargs, "history": list(kwargs["history"])})
        return self._answer(kwargs["system_prompt"], kwargs["user_message"]), {}

    @staticmethod
    def _answer(system_prompt: str, user_message: str) -> ProjectEstimation:
        name = _search(r"Project (\w+)", user_message) or _search(
            r"- Project name: (.+)", system_prompt
        )
        team = _search(r"team of (\d+)", user_message) or "2"
        task_count = 3 + int("Invoice export" in user_message)
        tasks = [
            {
                "task_id": number,
                "name": f"Task {number}",
                "description": "Build it",
                "estimated_hours": HOURS_PER_TASK,
                "estimated_cost_usd": HOURS_PER_TASK * 40,
                "complexity": "Medium",
                "includes": ["code"],
            }
            for number in range(1, task_count + 1)
        ]
        hours = HOURS_PER_TASK * task_count
        return build_estimation(
            project_name=name or "Unnamed",
            tasks=tasks,
            summary={
                "total_hours": hours,
                "total_cost_usd": hours * 40,
                "team_size": f"{team} developers",
                "estimated_duration_weeks": 2,
                "hourly_rate": 40,
            },
        )


def _search(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


@pytest.fixture
def wrapper() -> RecordingWrapper:
    return RecordingWrapper()


@pytest.fixture
def store() -> SessionStore:
    return SessionStore(system_prompt_provider=render_system_prompt, max_turns=MAX_TURNS)


@pytest.fixture(autouse=True)
def overrides(wrapper: RecordingWrapper, store: SessionStore) -> Iterator[None]:
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_session_estimation_service] = lambda: SessionEstimationService(
        wrapper
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def _new_session(client: httpx.AsyncClient) -> str:
    return (await client.post("/api/v1/sessions")).json()["session_id"]


async def _estimate(
    client: httpx.AsyncClient,
    session_id: str,
    transcript: str,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
) -> dict[str, Any]:
    response = await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": transcript, "hourly_rate": "40"},
        files=files,
    )
    assert response.status_code == 200
    return response.json()


async def test_two_chained_requests_update_project_metadata(
    client: httpx.AsyncClient, wrapper: RecordingWrapper
) -> None:
    session_id = await _new_session(client)

    first = await _estimate(
        client, session_id, "Project Falcon: a team of 3 builds a Django app with PostgreSQL."
    )
    second = await _estimate(client, session_id, "Please add Stripe payments and a Redis queue.")

    first_metadata = first["project_metadata"]
    assert first_metadata["project_name"] == "Falcon"
    assert first_metadata["assumed_team_size"] == 3
    assert {"Django", "PostgreSQL"} <= set(first_metadata["mentioned_technologies"])

    second_metadata = second["project_metadata"]
    assert second_metadata["project_name"] == "Falcon"
    assert {"Django", "PostgreSQL", "Stripe", "Redis"} <= set(
        second_metadata["mentioned_technologies"]
    )
    assert "<project_metadata>" not in wrapper.calls[0]["system_prompt"]
    second_system = wrapper.calls[1]["system_prompt"]
    assert "<project_metadata>" in second_system
    assert "- Project name: Falcon" in second_system
    assert "- Team size: 3" in second_system
    assert "Django" in second_system


async def test_pdf_attachment_influences_the_estimate(
    client: httpx.AsyncClient, wrapper: RecordingWrapper
) -> None:
    # Qualitative test of the plumbing: the fake LLM reacts to the attachment
    # text it receives, so this proves the PDF text reaches the prompt and that
    # the response reflects it, not that a real model would estimate more.
    transcript = "Project Falcon needs a customer portal."
    pdf = build_pdf("Invoice export required")
    session_without = await _new_session(client)
    session_with = await _new_session(client)

    plain = await _estimate(client, session_without, transcript)
    attached = await _estimate(
        client,
        session_with,
        transcript,
        files=[("attachments", ("spec.pdf", pdf, "application/pdf"))],
    )

    assert attached["result"]["summary"]["total_hours"] > plain["result"]["summary"]["total_hours"]
    assert len(attached["result"]["tasks"]) == len(plain["result"]["tasks"]) + 1
    assert attached["attachments"] == ["spec.pdf"]
    assert plain["attachments"] == []
    plain_prompt, attached_prompt = (call["user_message"] for call in wrapper.calls)
    assert ATTACHMENT_SEPARATOR not in plain_prompt
    assert ATTACHMENT_SEPARATOR in attached_prompt
    assert "Invoice export required" in attached_prompt


async def test_history_never_exceeds_max_turns(
    client: httpx.AsyncClient, wrapper: RecordingWrapper
) -> None:
    session_id = await _new_session(client)

    for turn in range(1, 9):
        body = await _estimate(client, session_id, f"Project Falcon, update number {turn}.")
        assert body["turns"] == min(turn, MAX_TURNS)

    assert len(wrapper.calls) == 8
    for number, call in enumerate(wrapper.calls, start=1):
        history = call["history"]
        assert len(history) == 2 * min(number - 1, MAX_TURNS)
        assert [message["role"] for message in history] == ["user", "assistant"] * (
            len(history) // 2
        )
        assert ("- Project name: Falcon" in call["system_prompt"]) == (number > 1)
    last_history = "\n".join(message["content"] for message in wrapper.calls[-1]["history"])
    assert "update number 1." not in last_history
    assert "update number 2." in last_history
    assert "update number 7." in last_history
