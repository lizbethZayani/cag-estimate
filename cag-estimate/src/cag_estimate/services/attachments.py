"""Text extraction for uploaded attachments (PDF and plain-text formats).

Attachments are turned into text inside this service and appended to the
transcript, so the same guardrails and prompt apply to everything the LLM sees.
Error messages are safe to show to API clients: they never carry library
exception text.
"""

import io
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

import structlog
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from cag_estimate.config import get_settings

log = structlog.get_logger()

_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
_PDF_EXTENSION = ".pdf"
_TEXT_CONTENT_TYPES = {"text/plain", "text/markdown", "text/csv", "application/json"}
_PDF_CONTENT_TYPE = "application/pdf"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_DASH_RUNS = re.compile(r"-{3,}")
_FALLBACK_FILENAME = "attachment"


class AttachmentError(Exception):
    """An attachment could not be used; the message is safe to show to clients."""


class UnsupportedAttachmentError(AttachmentError):
    """The attachment type is not supported."""


class AttachmentTooLargeError(AttachmentError):
    """A file, the file count, or the combined text exceeds a configured limit."""


@dataclass(frozen=True)
class ExtractedAttachment:
    filename: str
    text: str


def extract_attachment_text(filename: str, content_type: str | None, data: bytes) -> str:
    """Return the text of one attachment, deciding the type by extension first."""
    if len(data) > get_settings().attachment_max_bytes:
        raise AttachmentTooLargeError(f"Attachment '{_safe_name(filename)}' is too large.")
    if _is_pdf(filename, content_type):
        return _extract_pdf(filename, data)
    if _is_text(filename, content_type):
        return data.decode("utf-8", errors="replace")
    raise UnsupportedAttachmentError(f"Unsupported attachment type: '{_safe_name(filename)}'.")


def build_combined_transcript(transcript: str, attachments: list[ExtractedAttachment]) -> str:
    """Append each attachment after the transcript under a sanitized separator."""
    settings = get_settings()
    if len(attachments) > settings.attachment_max_files:
        raise AttachmentTooLargeError(
            f"Too many attachments; the maximum is {settings.attachment_max_files}."
        )
    parts = [transcript]
    for attachment in attachments:
        parts.append(
            f"\n\n--- attachment: {_safe_name(attachment.filename)} ---\n{attachment.text}"
        )
    combined = "".join(parts)
    if len(combined) > settings.attachment_max_chars_total:
        raise AttachmentTooLargeError(
            f"Transcript and attachments exceed {settings.attachment_max_chars_total} characters."
        )
    return combined


def _extension(filename: str) -> str:
    return PurePosixPath(_safe_name(filename)).suffix.lower()


def _is_pdf(filename: str, content_type: str | None) -> bool:
    extension = _extension(filename)
    return extension == _PDF_EXTENSION or (not extension and content_type == _PDF_CONTENT_TYPE)


def _is_text(filename: str, content_type: str | None) -> bool:
    extension = _extension(filename)
    return extension in _TEXT_EXTENSIONS or (not extension and content_type in _TEXT_CONTENT_TYPES)


def _safe_name(filename: str) -> str:
    """Strip path components and control characters so a name cannot inject prompt text."""
    base = PureWindowsPath(PurePosixPath(filename).name).name
    cleaned = _DASH_RUNS.sub("-", _CONTROL_CHARS.sub("", base)).strip()
    return cleaned or _FALLBACK_FILENAME


def _extract_pdf(filename: str, data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except (PyPdfError, ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        log.warning("pdf_unreadable", error_type=type(exc).__name__)
        raise AttachmentError(f"PDF '{_safe_name(filename)}' could not be read.") from exc
    if not text:
        raise AttachmentError(
            f"PDF '{_safe_name(filename)}' has no extractable text (it may be a scanned image)."
        )
    return text
