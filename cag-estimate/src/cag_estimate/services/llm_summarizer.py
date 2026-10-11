"""LLM-backed summarizer for evicted session turns.

Plugs into ``HistoryCompactor`` behind the ``Summarizer`` protocol. Any failure
(provider error, empty answer) propagates so the compactor can fall back to the
deterministic extractive summary. The call bypasses the exact cache: the prompt
holds conversation content, which must not be stored in Redis.
"""

from typing import Any, Protocol

from pydantic import ValidationError

from cag_estimate.prompts import SummaryTurn, render_summary_prompt
from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.services.session_compression import collapse_whitespace

SUMMARY_MAX_TOKENS = 600
DIGEST_FALLBACK_CHARS = 300


class TextCompleter(Protocol):
    """The slice of ``LLMWrapper`` this summarizer needs."""

    def complete(self, **kwargs: Any) -> dict[str, Any]: ...


def assistant_digest(content: str) -> str:
    """Compact view of a stored assistant message (the full estimate JSON)."""
    try:
        estimation = ProjectEstimation.model_validate_json(content)
    except ValidationError:
        return collapse_whitespace(content)[:DIGEST_FALLBACK_CHARS]
    return (
        f"project_name: {estimation.project_name}; "
        f"total_hours: {estimation.summary.total_hours}; "
        f"total_cost_usd: {estimation.summary.total_cost_usd}; "
        f"{len(estimation.tasks)} tasks"
    )


def clip_at_word(text: str, limit: int) -> str:
    """Whitespace-collapsed ``text`` cut to ``limit`` characters at a word boundary."""
    collapsed = collapse_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    if collapsed[limit].isspace():
        return collapsed[:limit]
    head, separator, _tail = collapsed[:limit].rpartition(" ")
    return head if separator else collapsed[:limit]


class LlmSummarizer:
    """Folds evicted (user, assistant) pairs into the cumulative summary with one LLM call."""

    def __init__(self, wrapper: TextCompleter, max_chars: int, model: str | None = None) -> None:
        self._wrapper = wrapper
        self._max_chars = max_chars
        self._model = model

    def summarize(self, previous_summary: str | None, evicted: list[tuple[str, str]]) -> str:
        turns = [self._to_turn(user, assistant) for user, assistant in evicted]
        system, user = render_summary_prompt(previous_summary, turns, self._max_chars)
        result = self._wrapper.complete(
            system_prompt=system,
            user_message=user,
            model_override=self._model,
            max_tokens=SUMMARY_MAX_TOKENS,
            use_cache=False,
        )
        summary = clip_at_word(result["estimation"], self._max_chars)
        if not summary:
            raise ValueError("The summarizer returned an empty summary")
        return summary

    @staticmethod
    def _to_turn(user: str, assistant: str) -> SummaryTurn:
        # Anchors that overflowed the cap arrive without an assistant reply.
        return SummaryTurn(user, assistant_digest(assistant) if assistant else None)
