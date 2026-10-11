"""HTTP layer for conversational (session) estimations.

Sessions live in process memory: they are lost on restart and not shared
between workers. Session estimates bypass the exact and semantic caches,
because the answer depends on the conversation, not only on the submitted text.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, TypeVar

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from cag_estimate.config import get_settings
from cag_estimate.dependencies import get_session_estimation_service, get_session_store
from cag_estimate.guardrails.input import InputGuardrailViolation
from cag_estimate.routers.estimations import UPSTREAM_ERROR, rejection_body
from cag_estimate.schemas.acb import ACBSessionResponse
from cag_estimate.schemas.estimation import MAX_HOURLY_RATE, MIN_HOURLY_RATE
from cag_estimate.schemas.session import ProjectMetadata, SessionCreated, SessionEstimationResponse
from cag_estimate.services.attachments import (
    AttachmentError,
    AttachmentTooLargeError,
    ExtractedAttachment,
    UnsupportedAttachmentError,
    extract_uploads,
)
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import Session, SessionNotFoundError, SessionStore
from cag_estimate.services.tier_resolver import Tier

log = structlog.get_logger()

ResponseT = TypeVar("ResponseT", bound=SessionEstimationResponse)

router = APIRouter(prefix="/api/v1", tags=["sessions"])

Store = Annotated[SessionStore, Depends(get_session_store)]
Service = Annotated[SessionEstimationService, Depends(get_session_estimation_service)]

SESSION_NOT_FOUND = "Session not found."
EMPTY_TRANSCRIPT = "The transcript must not be empty."
TRANSCRIPT_TOO_LONG = "The transcript exceeds the allowed length."
ATTACHMENT_TOO_LARGE = "An attachment or the combined text exceeds the allowed size."
ATTACHMENT_UNSUPPORTED = "Unsupported attachment type."
ATTACHMENT_INVALID = "An attachment could not be read."

Transcript = Annotated[str, Form()]
HourlyRate = Annotated[int, Form(ge=MIN_HOURLY_RATE, le=MAX_HOURLY_RATE)]
AudienceTier = Annotated[Tier | None, Form()]


class SessionSnapshot(SessionCreated):
    """Read-only view of a session, meant for a debugging panel."""

    project_metadata: ProjectMetadata
    turns: int
    max_turns: int
    message_count: int
    anchors_count: int
    summary_chars: int
    last_tier: str | None
    last_tier_rule: str | None
    created_at: datetime
    updated_at: datetime


def _get_session(store: SessionStore, session_id: str) -> Session:
    try:
        return store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND) from exc


def _validate_transcript(transcript: str) -> None:
    if not transcript.strip():
        raise HTTPException(status_code=422, detail=EMPTY_TRANSCRIPT)
    if len(transcript) > get_settings().transcript_max_chars:
        raise HTTPException(status_code=422, detail=TRANSCRIPT_TOO_LONG)


def _attachment_http_error(exc: AttachmentError) -> HTTPException:
    if isinstance(exc, AttachmentTooLargeError):
        return HTTPException(status_code=413, detail=ATTACHMENT_TOO_LARGE)
    if isinstance(exc, UnsupportedAttachmentError):
        return HTTPException(status_code=415, detail=ATTACHMENT_UNSUPPORTED)
    return HTTPException(status_code=422, detail=ATTACHMENT_INVALID)


@router.post("/sessions", response_model=SessionCreated, status_code=201)
def create_session(store: Store) -> SessionCreated:
    """Open a new in-memory conversation (volatile: lost on restart)."""
    return SessionCreated(session_id=store.create().session_id)


@router.get("/sessions/{session_id}", response_model=SessionSnapshot)
def get_session(session_id: str, store: Store) -> SessionSnapshot:
    """Return project metadata, window usage and timestamps (404 if unknown or expired)."""
    session = _get_session(store, session_id)
    turns = len(session.history)
    return SessionSnapshot(
        session_id=session.session_id,
        project_metadata=session.metadata.model_copy(deep=True),
        turns=turns,
        max_turns=session.history.max_turns,
        message_count=turns * 2,
        anchors_count=len(session.history.anchors),
        summary_chars=len(session.history.summary or ""),
        last_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, store: Store) -> Response:
    """Discard a session now instead of waiting for eviction (404 if unknown or expired)."""
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND)
    return Response(status_code=204)


def _handle_estimate(
    store: SessionStore,
    session_id: str,
    transcript: str,
    attachments: list[UploadFile] | None,
    estimate: Callable[[Session, list[ExtractedAttachment]], ResponseT],
) -> ResponseT | JSONResponse:
    """Shared request flow of both estimate routes: validate, read uploads, map errors."""
    uploads = attachments or []
    try:
        _validate_transcript(transcript)
        session = _get_session(store, session_id)
        return estimate(session, extract_uploads(uploads))
    except AttachmentError as exc:
        log.info("session_attachment_rejected", error_type=type(exc).__name__)
        raise _attachment_http_error(exc) from exc
    except InputGuardrailViolation as exc:
        log.info("session_estimate_rejected", reason=exc.reason)
        return JSONResponse(status_code=400, content=rejection_body(exc.reason))
    except HTTPException:
        raise
    except Exception as exc:
        log.error("session_estimate_failed", error_type=type(exc).__name__, error=str(exc))
        raise HTTPException(status_code=502, detail=UPSTREAM_ERROR) from exc
    finally:
        for upload in uploads:
            upload.file.close()


@router.post("/sessions/{session_id}/estimate", response_model=SessionEstimationResponse)
def estimate_in_session(
    session_id: str,
    store: Store,
    service: Service,
    transcript: Transcript,
    hourly_rate: HourlyRate = 40,
    tier: AudienceTier = None,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> SessionEstimationResponse | JSONResponse:
    """Estimate within a session (``multipart/form-data``: transcript + optional files).

    Attachments (PDF, DOCX, txt, md, csv, json) are converted to text and appended to
    the transcript. The optional ``tier`` field (executive, pm, developer, default)
    forces the audience; without it the tier is inferred. Session estimates bypass the
    caches; the session is in-memory only. Errors: unknown session 404, too large 413,
    unsupported type 415, unreadable attachment or bad form 422, rejected input 400
    ``{"reason", "message"}``, any other failure a generic 502.
    """
    return _handle_estimate(
        store,
        session_id,
        transcript,
        attachments,
        lambda session, extracted: service.estimate(
            session, transcript, extracted, hourly_rate, tier
        ),
    )


@router.post("/sessions/{session_id}/estimate-acb", response_model=ACBSessionResponse)
def estimate_reviewed_in_session(
    session_id: str,
    store: Store,
    service: Service,
    transcript: Transcript,
    hourly_rate: HourlyRate = 40,
    tier: AudienceTier = None,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> ACBSessionResponse | JSONResponse:
    """Estimate within a session with an independent critic reviewing each draft.

    Same multipart contract and errors as ``/estimate``; the response adds ``acb``
    (iterations, final decision, caveats). Cost: up to ``ACB_MAX_ITERATIONS`` x
    (draft + review) LLM calls, so it is slower and more expensive than ``/estimate``.
    Like all session estimates it bypasses the caches. The session changes once,
    from the final estimate; open issues are reported in ``acb.caveats`` and never
    written into the estimate.
    """
    return _handle_estimate(
        store,
        session_id,
        transcript,
        attachments,
        lambda session, extracted: service.estimate_with_acb(
            session, transcript, extracted, hourly_rate, tier
        ),
    )
