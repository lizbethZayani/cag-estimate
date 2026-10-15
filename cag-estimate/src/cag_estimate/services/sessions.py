"""In-memory conversational state: project metadata, sliding-window history, sessions.

Volatility is accepted at this stage: sessions live in a process-local dict, so
state is lost on restart, is not shared across workers/processes, and is never
persisted. Later stages move it to Redis or a database.
"""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from cag_estimate.config import DEFAULT_SESSION_MAX_SESSIONS, DEFAULT_SESSION_MAX_TURNS
from cag_estimate.schemas.session import ProjectMetadata
from cag_estimate.services.tier_resolver import Tier

log = structlog.get_logger()

DEFAULT_MAX_TURNS = DEFAULT_SESSION_MAX_TURNS
DEFAULT_MAX_SESSIONS = DEFAULT_SESSION_MAX_SESSIONS

SystemPromptProvider = Callable[[ProjectMetadata], str]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Anchor:
    """A hard-constraint user message pinned for the whole session (``kind`` + compact text)."""

    kind: str
    text: str


CONTEXT_HEADER = "[Earlier conversation context - the turns below are the live thread]"
CONTEXT_ACKNOWLEDGEMENT = "Understood."


def _render_context(anchors: list[Anchor], summary: str | None) -> str:
    sections = [CONTEXT_HEADER]
    if anchors:
        lines = "\n".join(f"- ({anchor.kind}) {anchor.text}" for anchor in anchors)
        sections.append(f"Pinned constraints from earlier turns:\n{lines}")
    if summary:
        sections.append(f"Summary of earlier turns:\n{summary}")
    return "\n".join(sections)


class ConversationHistory:
    """Sliding window of user/assistant pairs plus the context folded out of it.

    The system prompt is regenerated on every ``to_messages_list`` call from the
    current metadata, so it always reflects what is known and is never trimmed.
    Pairs pushed out of the window are returned by ``add_turn``; a compactor can
    pin their anchors and summary back through ``update_context``.
    """

    def __init__(
        self,
        metadata: ProjectMetadata,
        system_prompt_provider: SystemPromptProvider,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._metadata = metadata
        self._system_prompt_provider = system_prompt_provider
        self._max_turns = max_turns
        self._turns: list[tuple[str, str]] = []
        self._anchors: list[Anchor] = []
        self._summary: str | None = None

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    @property
    def anchors(self) -> list[Anchor]:
        return list(self._anchors)

    @property
    def summary(self) -> str | None:
        return self._summary

    def __len__(self) -> int:
        return len(self._turns)

    def add_turn(self, user_content: str, assistant_content: str) -> list[tuple[str, str]]:
        """Append one complete pair; return the oldest pairs pushed out of the window."""
        self._turns.append((user_content, assistant_content))
        evicted = self._turns[: -self._max_turns]
        del self._turns[: -self._max_turns]
        return evicted

    def update_context(self, anchors: list[Anchor], summary: str | None) -> None:
        """Replace the pinned anchors and the cumulative summary of evicted turns."""
        self._anchors = list(anchors)
        self._summary = summary or None

    def to_messages_list(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt_provider(self._metadata)}]
        if self._anchors or self._summary:
            messages.append(
                {"role": "user", "content": _render_context(self._anchors, self._summary)}
            )
            messages.append({"role": "assistant", "content": CONTEXT_ACKNOWLEDGEMENT})
        for user_content, assistant_content in self._turns:
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
        return messages


class Session:
    """One conversation: metadata plus bounded history."""

    def __init__(
        self,
        session_id: str,
        system_prompt_provider: SystemPromptProvider,
        max_turns: int = DEFAULT_MAX_TURNS,
        clock: Clock = _utc_now,
    ) -> None:
        self.session_id = session_id
        self.metadata = ProjectMetadata()
        self.history = ConversationHistory(self.metadata, system_prompt_provider, max_turns)
        self._clock = clock
        self.created_at = clock()
        self.updated_at = self.created_at
        # Audience tier of the last successful turn (None before the first one).
        self.last_resolved_tier: Tier | None = None
        self.last_tier_rule: str | None = None
        # Successful turns so far; unlike len(history) it keeps growing past the window.
        self.turn_count = 0
        # Held by callers for a whole read-modify-write (e.g. one estimate) so
        # concurrent requests on the same session cannot interleave their turns.
        self.lock = threading.Lock()

    def touch(self) -> None:
        self.updated_at = self._clock()

    def messages_for_llm(self) -> list[dict[str, str]]:
        """System prompt + stored pairs; the caller appends the pending user message."""
        return self.history.to_messages_list()


class SessionNotFoundError(KeyError):
    """Raised when a session id is unknown (never created, evicted, or restarted away)."""


def _default_system_prompt(_metadata: ProjectMetadata) -> str:
    return "You are a software project estimation assistant."


class SessionStore:
    """Thread-safe in-memory store with an LRU cap and optional idle expiry.

    A session idle for more than ``idle_ttl_seconds`` (0 disables expiry) is treated
    as gone. Expiry is lazy: it is only checked at store entry points (``create``
    purges, ``get`` and ``delete`` check one id), never by a background thread. A
    session whose ``lock`` is held is being used by a request, so it is never expired.
    """

    def __init__(
        self,
        system_prompt_provider: SystemPromptProvider = _default_system_prompt,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        idle_ttl_seconds: int = 0,
        clock: Clock = _utc_now,
    ) -> None:
        self._system_prompt_provider = system_prompt_provider
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._idle_ttl_seconds = idle_ttl_seconds
        self._clock = clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _is_expired(self, session: Session) -> bool:
        if self._idle_ttl_seconds <= 0 or session.lock.locked():
            return False
        return (self._clock() - session.updated_at).total_seconds() > self._idle_ttl_seconds

    def _purge_expired(self) -> None:
        """Drop every expired session; the caller holds the store lock."""
        for session_id in [sid for sid, s in self._sessions.items() if self._is_expired(s)]:
            del self._sessions[session_id]
            log.info("session_expired", session_id=session_id, ttl=self._idle_ttl_seconds)

    def create(self) -> Session:
        session = Session(
            str(uuid.uuid4()), self._system_prompt_provider, self._max_turns, self._clock
        )
        with self._lock:
            self._purge_expired()
            self._sessions[session.session_id] = session
            while len(self._sessions) > self._max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                log.warning("session_evicted", session_id=evicted_id, cap=self._max_sessions)
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and self._is_expired(session):
                del self._sessions[session_id]
                log.info("session_expired", session_id=session_id, ttl=self._idle_ttl_seconds)
                session = None
            if session is None:
                raise SessionNotFoundError(session_id)
            self._sessions.move_to_end(session_id)
            session.touch()
        return session

    def delete(self, session_id: str) -> bool:
        """Remove a session; return False when it was unknown or already expired."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            return session is not None and not self._is_expired(session)
