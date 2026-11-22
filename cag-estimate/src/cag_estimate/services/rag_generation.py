"""Cited estimate generation through the OpenAI Responses API (strict structured output).

Provider failures never leak: the cause is chained, the raised message is generic or
names only the validation rule that failed.
"""

from typing import Any, Protocol

import structlog
from pydantic import ValidationError

from cag_estimate.prompts.loader import render_rag_estimation_prompt
from cag_estimate.schemas.rag_estimation import RagEstimate
from cag_estimate.services.rag_errors import RagGenerationError, RagOutputInvalidError

log = structlog.get_logger()

_VALUE_ERROR_PREFIX = "Value error, "
_MAX_RULE_CHARS = 500


class ResponsesApi(Protocol):
    """The slice of ``AsyncOpenAI().responses`` the generator uses."""

    async def parse(self, **kwargs: Any) -> Any: ...


class ResponsesClient(Protocol):
    """An async OpenAI client; tests inject a fake exposing the same async ``parse``."""

    @property
    def responses(self) -> ResponsesApi: ...


class RagGenerator:
    """Asks the model for a ``RagEstimate`` grounded in the given context block."""

    def __init__(self, client: ResponsesClient, model: str, max_output_tokens: int) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    async def generate(
        self, query: str, context_block: str, feedback: str | None = None
    ) -> RagEstimate:
        """Return the parsed estimate.

        Raises:
            RagOutputInvalidError: The answer broke the ``RagEstimate`` integrity rules.
            RagGenerationError: The provider failed or returned no parsed answer.
        """
        system, user = render_rag_estimation_prompt(query, context_block, feedback)
        response = await self._parse(system, user)
        if response.output_parsed is None:
            raise RagGenerationError("the model returned no parsable estimate")
        return response.output_parsed

    async def _parse(self, system: str, user: str) -> Any:
        try:
            return await self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=RagEstimate,
                max_output_tokens=self._max_output_tokens,
            )
        except ValidationError as exc:
            raise RagOutputInvalidError(_failed_rules(exc)) from exc
        except Exception as exc:
            log.warning("rag_generation_failed", error_type=type(exc).__name__)
            raise RagGenerationError("the generation request failed") from exc


def _failed_rules(exc: ValidationError) -> str:
    """The validation messages (rules, not input values), short enough for a retry prompt."""
    messages = [error["msg"].removeprefix(_VALUE_ERROR_PREFIX) for error in exc.errors()]
    return " ".join(messages)[:_MAX_RULE_CHARS]
