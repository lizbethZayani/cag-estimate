"""HTTP layer for conversational (session) estimations.

Sessions live in process memory: they are lost on restart and not shared
between workers. Session estimates bypass the exact and semantic caches,
because the answer depends on the conversation, not only on the submitted text.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cag_estimate.config import get_settings
from cag_estimate.dependencies import get_session_estimation_service, get_session_store
from cag_estimate.guardrails.input import InputGuardrailViolation
from cag_estimate.routers.estimations import UPSTREAM_ERROR, rejection_body
from cag_estimate.schemas.estimation import MAX_HOURLY_RATE, MIN_HOURLY_RATE
from cag_estimate.schemas.session import ProjectMetadata, SessionCreated, SessionEstimationResponse
from cag_estimate.services.attachments import (
    AttachmentError,
    AttachmentTooLargeError,
    ExtractedAttachment,
    UnsupportedAttachmentError,
    extract_attachment_text,
)
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import Session, SessionNotFoundError, SessionStore

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["sessions"])

Store = Annotated[SessionStore, Depends(get_session_store)]
Service = Annotated[SessionEstimationService, Depends(get_session_estimation_service)]

MAX_TRANSCRIPT_CHARS = 80_000
READ_CHUNK_BYTES = 64 * 1024
SESSION_NOT_FOUND = "Session not found."
EMPTY_TRANSCRIPT = "The transcript must not be empty."
ATTACHMENT_TOO_LARGE = "An attachment or the combined text exceeds the allowed size."
ATTACHMENT_UNSUPPORTED = "Unsupported attachment type."
ATTACHMENT_INVALID = "An attachment could not be read."

Transcript = Annotated[str, Form(max_length=MAX_TRANSCRIPT_CHARS)]
HourlyRate = Annotated[int, Form(ge=MIN_HOURLY_RATE, le=MAX_HOURLY_RATE)]


class SessionSnapshot(SessionCreated):
    """Read-only view of a session, meant for a debugging panel."""

    project_metadata: ProjectMetadata
    turns: int


def _get_session(store: SessionStore, session_id: str) -> Session:
    try:
        return store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND) from exc


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` + 1 bytes so an oversized file is never fully buffered."""
    chunks: list[bytes] = []
    size = 0
    while chunk := upload.file.read(READ_CHUNK_BYTES):
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            break
    return b"".join(chunks)


def _extract_all(uploads: list[UploadFile]) -> list[ExtractedAttachment]:
    settings = get_settings()
    if len(uploads) > settings.attachment_max_files:
        raise AttachmentTooLargeError(
            f"Too many attachments; the maximum is {settings.attachment_max_files}."
        )
    extracted: list[ExtractedAttachment] = []
    for upload in uploads:
        filename = upload.filename or ""
        data = _read_limited(upload, settings.attachment_max_bytes)
        text = extract_attachment_text(filename, upload.content_type, data)
        extracted.append(ExtractedAttachment(filename=filename, text=text))
    return extracted


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
    """Return the accumulated project metadata and turn count (404 if unknown)."""
    session = _get_session(store, session_id)
    return SessionSnapshot(
        session_id=session.session_id,
        project_metadata=session.metadata.model_copy(deep=True),
        turns=len(session.history),
    )


@router.post("/sessions/{session_id}/estimate", response_model=SessionEstimationResponse)
def estimate_in_session(
    session_id: str,
    store: Store,
    service: Service,
    transcript: Transcript,
    hourly_rate: HourlyRate = 40,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> SessionEstimationResponse | JSONResponse:
    """Estimate within a session (``multipart/form-data``: transcript + optional files).

    Attachments (PDF, txt, md, csv, json) are converted to text and appended to
    the transcript. Session estimates bypass the caches; the session is
    in-memory only. Errors: unknown session 404, too large 413, unsupported
    type 415, unreadable attachment or bad form 422, rejected input 400
    ``{"reason", "message"}``, any other failure a generic 502.
    """
    uploads = attachments or []
    try:
        if not transcript.strip():
            raise HTTPException(status_code=422, detail=EMPTY_TRANSCRIPT)
        session = _get_session(store, session_id)
        extracted = _extract_all(uploads)
        return service.estimate(session, transcript, extracted, hourly_rate)
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
