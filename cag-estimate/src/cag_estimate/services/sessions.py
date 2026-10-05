"""In-memory conversational state: project metadata, sliding-window history, sessions.

Volatility is accepted at this stage: sessions live in a process-local dict, so
state is lost on restart, is not shared across workers/processes, and is never
persisted. Later stages move it to Redis or a database.
"""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from cag_estimate.config import DEFAULT_SESSION_MAX_SESSIONS, DEFAULT_SESSION_MAX_TURNS
from cag_estimate.schemas.session import ProjectMetadata

log = structlog.get_logger()

DEFAULT_MAX_TURNS = DEFAULT_SESSION_MAX_TURNS
DEFAULT_MAX_SESSIONS = DEFAULT_SESSION_MAX_SESSIONS

SystemPromptProvider = Callable[[ProjectMetadata], str]


class ConversationHistory:
    """Sliding window of user/assistant pairs; the system prompt is never stored.

    The system prompt is regenerated on every ``to_messages_list`` call from the
    current metadata, so it always reflects what is known and is never trimmed.
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

    @property
    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        """Append one complete pair and drop the oldest pairs beyond the window."""
        self._turns.append((user_content, assistant_content))
        del self._turns[: -self._max_turns]

    def to_messages_list(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt_provider(self._metadata)}]
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
    ) -> None:
        self.session_id = session_id
        self.metadata = ProjectMetadata()
        self.history = ConversationHistory(self.metadata, system_prompt_provider, max_turns)
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def messages_for_llm(self) -> list[dict[str, str]]:
        """System prompt + stored pairs; the caller appends the pending user message."""
        return self.history.to_messages_list()


class SessionNotFoundError(KeyError):
    """Raised when a session id is unknown (never created, evicted, or restarted away)."""


def _default_system_prompt(_metadata: ProjectMetadata) -> str:
    return "You are a software project estimation assistant."


class SessionStore:
    """Thread-safe in-memory store with an LRU cap on the number of sessions."""

    def __init__(
        self,
        system_prompt_provider: SystemPromptProvider = _default_system_prompt,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self._system_prompt_provider = system_prompt_provider
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create(self) -> Session:
        session = Session(str(uuid.uuid4()), self._system_prompt_provider, self._max_turns)
        with self._lock:
            self._sessions[session.session_id] = session
            while len(self._sessions) > self._max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                log.warning("session_evicted", session_id=evicted_id, cap=self._max_sessions)
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            self._sessions.move_to_end(session_id)
        session.touch()
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
