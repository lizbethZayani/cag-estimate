"""Request id correlation: validation of incoming ids and access to the bound id."""

import re
import uuid

from structlog.contextvars import get_contextvars

REQUEST_ID_HEADER = "X-Request-ID"

_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def resolve_request_id(incoming: str | None) -> str:
    """Accept a well-formed incoming id, otherwise generate a fresh hex id."""
    if incoming is not None and _VALID_REQUEST_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    """The request id bound to the current request, if any."""
    return get_contextvars().get("request_id")
