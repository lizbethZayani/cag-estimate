"""Session-aware estimation: conversation memory, attachments, metadata updates.

Session estimates BYPASS the exact and semantic caches on purpose: the answer
depends on the conversation history and accumulated project metadata, not only
on the submitted text, so a cache keyed on the text would serve stale or
wrong-context answers (and would hide the memory behaviour being exercised).
"""

from typing import Any

import structlog

from cag_estimate.guardrails.input import check_input
from cag_estimate.guardrails.output import enforce_output
from cag_estimate.prompts import render_estimation_prompt
from cag_estimate.schemas.estimation import EstimationRequest, ProjectEstimation
from cag_estimate.schemas.session import SessionEstimationResponse
from cag_estimate.services.attachments import ExtractedAttachment, build_combined_transcript
from cag_estimate.services.estimation import MAX_TOKENS, PROMPT_VERSION
from cag_estimate.services.llm_wrapper import LLMWrapper
from cag_estimate.services.metadata_extractor import extract_metadata
from cag_estimate.services.sessions import Session

log = structlog.get_logger()

MAX_HISTORY_TRANSCRIPT_CHARS = 4000
_TRUNCATION_MARKER = "\n[... transcript truncated in conversation history ...]"


def build_history_user_turn(transcript: str, attachments: list[ExtractedAttachment]) -> str:
    """Compact user message stored in the sliding window.

    The full text (transcript + attachments) goes to the LLM once, in the current
    prompt. Keeping it in history would put tens of thousands of characters in
    every later turn, so only a truncated transcript and the attachment names are kept.
    """
    if len(transcript) > MAX_HISTORY_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_HISTORY_TRANSCRIPT_CHARS] + _TRUNCATION_MARKER
    if not attachments:
        return transcript
    names = ", ".join(attachment.filename for attachment in attachments)
    return f"{transcript}\n\n[attachments: {names}]"


class SessionEstimationService:
    """Produces estimates that remember the conversation they belong to."""

    def __init__(self, wrapper: LLMWrapper, moderation_client: Any | None = None) -> None:
        self.wrapper = wrapper
        self.moderation_client = moderation_client

    def estimate(
        self,
        session: Session,
        transcript: str,
        attachments: list[ExtractedAttachment],
        hourly_rate: int,
    ) -> SessionEstimationResponse:
        """Estimate within ``session``; history and metadata change only on success.

        The session lock is held for the whole read-modify-write, so concurrent
        requests on one session run one after another.
        """
        with session.lock:
            combined = build_combined_transcript(transcript, attachments)
            check_input(combined, openai_client=self.moderation_client)
            result = enforce_output(self._generate(session, combined, hourly_rate), hourly_rate)
            self._remember(session, transcript, attachments, combined, result)
            return SessionEstimationResponse(
                result=result,
                prompt_version=PROMPT_VERSION,
                cached=False,
                session_id=session.session_id,
                project_metadata=session.metadata,
                turns=len(session.history),
                attachments=[attachment.filename for attachment in attachments],
            )

    def _generate(self, session: Session, combined: str, hourly_rate: int) -> ProjectEstimation:
        request = EstimationRequest(transcription=combined, hourly_rate=hourly_rate)
        system, user = render_estimation_prompt(
            request, version=PROMPT_VERSION, project_metadata=session.metadata
        )
        result, _meta = self.wrapper.complete_structured(
            system_prompt=system,
            user_message=user,
            response_model=ProjectEstimation,
            max_tokens=MAX_TOKENS,
            history=session.messages_for_llm()[1:],
        )
        return result

    @staticmethod
    def _remember(
        session: Session,
        transcript: str,
        attachments: list[ExtractedAttachment],
        combined: str,
        result: ProjectEstimation,
    ) -> None:
        session.metadata.merge(extract_metadata(result, combined))
        session.history.add_turn(
            build_history_user_turn(transcript, attachments), result.model_dump_json()
        )
