"""Tests for the Responses API generator, with a scripted fake client (no network)."""

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from cag_estimate.schemas.rag_estimation import RagEstimate
from cag_estimate.services.rag_errors import RagGenerationError, RagOutputInvalidError
from cag_estimate.services.rag_generation import RagGenerator
from tests.rag_fakes import QUERY, make_estimate

CONTEXT = '<sources>\n<source id="BUD-1::CART-001" document_id="BUD-1">Cart</source>\n</sources>'
MODEL = "gpt-test"
MAX_TOKENS = 1234
RAW_SECRET = "sk-very-secret raw provider text"


class FakeResponses:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(output_parsed=outcome)


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.responses = FakeResponses(outcomes)


def _generator(*outcomes: Any) -> tuple[RagGenerator, FakeResponses]:
    client = FakeClient(list(outcomes))
    return RagGenerator(client, MODEL, MAX_TOKENS), client.responses


def _validation_error() -> ValidationError:
    try:
        RagEstimate.model_validate(
            {
                "lines": [],
                "assumptions": [],
                "confidence": "high",
                "insufficient_context_explanation": None,
            }
        )
    except ValidationError as exc:
        return exc
    raise AssertionError("the payload should be invalid")


async def test_generate_returns_the_parsed_estimate() -> None:
    estimate = make_estimate("BUD-1::CART-001")
    generator, _ = _generator(estimate)

    assert await generator.generate(QUERY, CONTEXT) is estimate


async def test_generate_sends_the_model_format_and_both_messages() -> None:
    generator, responses = _generator(make_estimate("BUD-1::CART-001"))

    await generator.generate(QUERY, CONTEXT)

    (call,) = responses.calls
    assert call["text_format"] is RagEstimate
    assert call["model"] == MODEL
    assert call["max_output_tokens"] == MAX_TOKENS
    system, user = call["input"]
    assert system["role"] == "system"
    assert "HOURS" in system["content"]
    assert user["role"] == "user"
    assert QUERY in user["content"]
    assert 'id="BUD-1::CART-001"' in user["content"]


async def test_feedback_reaches_the_user_message() -> None:
    generator, responses = _generator(make_estimate("BUD-1::CART-001"))

    await generator.generate(QUERY, CONTEXT, feedback="cite only ids in the context")

    assert "cite only ids in the context" in responses.calls[0]["input"][1]["content"]


async def test_no_parsed_output_is_a_generation_error() -> None:
    generator, _ = _generator(None)

    with pytest.raises(RagGenerationError):
        await generator.generate(QUERY, CONTEXT)


async def test_a_validation_error_becomes_an_output_invalid_error_with_the_rule() -> None:
    generator, _ = _generator(_validation_error())

    with pytest.raises(RagOutputInvalidError) as info:
        await generator.generate(QUERY, CONTEXT)

    assert "insufficient" in str(info.value)
    assert isinstance(info.value.__cause__, ValidationError)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(RAW_SECRET),
        openai.APIConnectionError(message=RAW_SECRET, request=httpx.Request("POST", "http://x")),
    ],
)
async def test_sdk_errors_become_a_generation_error_without_raw_text(error: Exception) -> None:
    generator, _ = _generator(error)

    with pytest.raises(RagGenerationError) as info:
        await generator.generate(QUERY, CONTEXT)

    assert "sk-very-secret" not in str(info.value)
    assert info.value.__cause__ is error


def test_rag_estimate_converts_to_a_strict_json_schema() -> None:
    schema = to_strict_json_schema(RagEstimate)

    assert schema["type"] == "object"
    assert "total_hours" not in schema["properties"]
