"""Session-aware estimation: conversation memory, attachments, metadata updates.

Session estimates BYPASS the exact and semantic caches on purpose: the answer
depends on the conversation history and accumulated project metadata, not only
on the submitted text, so a cache keyed on the text would serve stale or
wrong-context answers (and would hide the memory behaviour being exercised).
"""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from cag_estimate.config import get_settings
from cag_estimate.guardrails.input import check_input
from cag_estimate.guardrails.output import enforce_output
from cag_estimate.prompts import render_estimation_prompt
from cag_estimate.schemas.acb import ACBSessionResponse, CriticFeedback
from cag_estimate.schemas.estimation import EstimationRequest, ProjectEstimation
from cag_estimate.schemas.session import SessionEstimationResponse
from cag_estimate.services.attachments import ExtractedAttachment, build_combined_transcript
from cag_estimate.services.boss import Boss
from cag_estimate.services.critic import Critic, build_estimate_digest
from cag_estimate.services.llm_wrapper import LLMWrapper
from cag_estimate.services.metadata_extractor import extract_metadata
from cag_estimate.services.session_compression import HistoryCompactor
from cag_estimate.services.sessions import Session
from cag_estimate.services.tier_resolver import Tier, resolve_tier

# Sessions use the conversational prompt (updated FULL estimate + audience); the
# non-session /estimate path keeps v1. A constant, not a setting: nothing needs
# to switch it per deployment, and the response reports the version used.
SESSION_PROMPT_VERSION = "v2"
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


@dataclass(frozen=True)
class _Prepared:
    """Input checked and audience resolved: everything needed before the first LLM call."""

    combined: str
    tier: Tier
    rule: str


def _previous_digest(session: Session) -> str | None:
    """Digest of the last estimate in the history; None when absent or unparseable."""
    if not session.history.turns:
        return None
    try:
        previous = ProjectEstimation.model_validate_json(session.history.turns[-1][1])
    except ValidationError:
        return None
    return build_estimate_digest(previous)


class SessionEstimationService:
    """Produces estimates that remember the conversation they belong to."""

    def __init__(
        self,
        wrapper: LLMWrapper,
        moderation_client: Any | None = None,
        compactor: HistoryCompactor | None = None,
        critic: Critic | None = None,
        max_iterations: int | None = None,
    ) -> None:
        settings = get_settings()
        self.wrapper = wrapper
        self.moderation_client = moderation_client
        # None restores the plain sliding window: evicted pairs are simply dropped.
        self.compactor = compactor
        self.critic = critic or Critic(wrapper, settings.critic_model)
        self.boss = Boss(max_iterations or settings.acb_max_iterations)

    def estimate(
        self,
        session: Session,
        transcript: str,
        attachments: list[ExtractedAttachment],
        hourly_rate: int,
        tier: Tier | None = None,
    ) -> SessionEstimationResponse:
        """Estimate within ``session``; history, metadata and tier change only on success.

        The audience tier is resolved on the combined text (``tier`` overrides it).

        The session lock is held for the whole read-modify-write, so concurrent
        requests on one session run one after another.
        """
        with session.lock:
            prepared = self._prepare(session, transcript, attachments, tier)
            result = self._draft(session, prepared, hourly_rate, None)
            return self._commit(session, transcript, attachments, prepared, result)

    def estimate_with_acb(
        self,
        session: Session,
        transcript: str,
        attachments: list[ExtractedAttachment],
        hourly_rate: int,
        tier: Tier | None = None,
    ) -> ACBSessionResponse:
        """Like :meth:`estimate`, but an independent critic reviews each draft.

        Costs up to ``max_iterations`` x (draft + review) LLM calls. The session is
        changed once, from the final draft, and only after the whole loop succeeded.
        """
        with session.lock:
            prepared = self._prepare(session, transcript, attachments, tier)
            previous = _previous_digest(session)
            outcome = self.boss.run(
                lambda feedback: self._draft(session, prepared, hourly_rate, feedback),
                lambda draft: self.critic.review(
                    transcript=prepared.combined,
                    metadata=session.metadata,
                    tier=prepared.tier,
                    result=draft,
                    previous_digest=previous,
                ),
            )
            response = self._commit(session, transcript, attachments, prepared, outcome.result)
            return ACBSessionResponse(**dict(response), acb=outcome.trace)

    def _prepare(
        self,
        session: Session,
        transcript: str,
        attachments: list[ExtractedAttachment],
        tier: Tier | None,
    ) -> _Prepared:
        combined = build_combined_transcript(transcript, attachments)
        check_input(combined, openai_client=self.moderation_client)
        resolved, rule = resolve_tier(combined, session.metadata, tier)
        return _Prepared(combined, resolved, rule)

    def _draft(
        self,
        session: Session,
        prepared: _Prepared,
        hourly_rate: int,
        feedback: CriticFeedback | None,
    ) -> ProjectEstimation:
        request = EstimationRequest(transcription=prepared.combined, hourly_rate=hourly_rate)
        system, user = render_estimation_prompt(
            request,
            version=SESSION_PROMPT_VERSION,
            project_metadata=session.metadata,
            tier=prepared.tier,
            critic_feedback=feedback,
        )
        result, _meta = self.wrapper.complete_structured(
            system_prompt=system,
            user_message=user,
            response_model=ProjectEstimation,
            history=session.messages_for_llm()[1:],
        )
        return enforce_output(result, hourly_rate)

    def _commit(
        self,
        session: Session,
        transcript: str,
        attachments: list[ExtractedAttachment],
        prepared: _Prepared,
        result: ProjectEstimation,
    ) -> SessionEstimationResponse:
        session.metadata.merge(extract_metadata(result, prepared.combined))
        evicted = session.history.add_turn(
            build_history_user_turn(transcript, attachments), result.model_dump_json()
        )
        if self.compactor is not None:
            self.compactor.compact(session.history, evicted)
        session.last_resolved_tier, session.last_tier_rule = prepared.tier, prepared.rule
        return SessionEstimationResponse(
            result=result,
            prompt_version=SESSION_PROMPT_VERSION,
            resolved_tier=prepared.tier,
            tier_rule=prepared.rule,
            cached=False,
            session_id=session.session_id,
            project_metadata=session.metadata.model_copy(deep=True),
            turns=len(session.history),
            attachments=[attachment.filename for attachment in attachments],
        )
