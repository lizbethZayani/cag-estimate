"""Offline unit tests for the in-memory session state."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from cag_estimate.services.sessions import (
    DEFAULT_MAX_TURNS,
    ConversationHistory,
    ProjectMetadata,
    Session,
    SessionNotFoundError,
    SessionStore,
)


def render(metadata: ProjectMetadata) -> str:
    return f"SYSTEM project={metadata.project_name}"


def make_history(max_turns: int = 3, metadata: ProjectMetadata | None = None):
    metadata = metadata or ProjectMetadata()
    return ConversationHistory(metadata, render, max_turns=max_turns), metadata


# --- ProjectMetadata -------------------------------------------------------


def test_metadata_defaults_are_empty():
    metadata = ProjectMetadata()
    assert metadata.is_empty()
    assert metadata.mentioned_technologies == []


def test_metadata_with_any_value_is_not_empty():
    assert not ProjectMetadata(project_name="Acme").is_empty()
    assert not ProjectMetadata(mentioned_technologies=["Rails"]).is_empty()


def test_merge_overwrites_only_with_non_empty_values():
    metadata = ProjectMetadata(project_name="Acme", assumed_team_size=4, agreed_scope="MVP")
    metadata.merge(ProjectMetadata(project_name=None, assumed_team_size=None, agreed_scope=""))
    assert metadata.project_name == "Acme"
    assert metadata.assumed_team_size == 4
    assert metadata.agreed_scope == "MVP"


def test_merge_overwrites_with_new_values():
    metadata = ProjectMetadata(project_name="Acme", assumed_team_size=4)
    metadata.merge(ProjectMetadata(project_name="Globex", assumed_team_size=6, agreed_scope="API"))
    assert (metadata.project_name, metadata.assumed_team_size, metadata.agreed_scope) == (
        "Globex",
        6,
        "API",
    )


def test_merge_unions_technologies_case_insensitively_keeping_first_casing():
    metadata = ProjectMetadata(mentioned_technologies=["Python", "Redis"])
    metadata.merge(ProjectMetadata(mentioned_technologies=["python", "FastAPI", "REDIS"]))
    assert metadata.mentioned_technologies == ["Python", "Redis", "FastAPI"]


def test_technologies_are_deduplicated_on_construction():
    metadata = ProjectMetadata(mentioned_technologies=["Rails", "rails", "Docker"])
    assert metadata.mentioned_technologies == ["Rails", "Docker"]


# --- ConversationHistory ---------------------------------------------------


def test_default_max_turns_constant_is_six():
    assert DEFAULT_MAX_TURNS == 6


def test_empty_history_returns_only_system_message():
    history, _ = make_history()
    assert history.to_messages_list() == [{"role": "system", "content": "SYSTEM project=None"}]
    assert len(history) == 0


def test_no_drop_below_limit():
    history, _ = make_history(max_turns=3)
    for i in range(3):
        history.add_turn(f"u{i}", f"a{i}")
    assert len(history) == 3
    assert history.turns[0] == ("u0", "a0")


def test_oldest_pairs_dropped_above_limit():
    history, _ = make_history(max_turns=3)
    for i in range(4):
        history.add_turn(f"u{i}", f"a{i}")
    assert history.turns == [("u1", "a1"), ("u2", "a2"), ("u3", "a3")]


def test_exactly_max_turns_pairs_kept_after_eight_turns():
    history, _ = make_history(max_turns=6)
    for i in range(8):
        history.add_turn(f"u{i}", f"a{i}")
    messages = history.to_messages_list()
    assert len(history) == 6
    assert len(messages) == 1 + 6 * 2
    assert messages[1] == {"role": "user", "content": "u2"}
    assert messages[-1] == {"role": "assistant", "content": "a7"}


def test_system_prompt_is_first_and_alternation_is_preserved_after_trimming():
    history, _ = make_history(max_turns=2)
    for i in range(5):
        history.add_turn(f"u{i}", f"a{i}")
    messages = history.to_messages_list()
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user", "assistant"]
    assert sum(1 for m in messages if m["role"] == "system") == 1


def test_system_prompt_regenerated_from_current_metadata():
    history, metadata = make_history()
    history.add_turn("u", "a")
    metadata.merge(ProjectMetadata(project_name="Acme"))
    assert history.to_messages_list()[0]["content"] == "SYSTEM project=Acme"


def test_system_prompt_survives_trimming_with_updated_metadata():
    history, metadata = make_history(max_turns=1)
    for i in range(3):
        history.add_turn(f"u{i}", f"a{i}")
    metadata.merge(ProjectMetadata(project_name="Late"))
    messages = history.to_messages_list()
    assert messages[0]["content"] == "SYSTEM project=Late"
    assert len(messages) == 3


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError):
        ConversationHistory(ProjectMetadata(), render, max_turns=0)


# --- Session ---------------------------------------------------------------


def test_session_messages_for_llm_delegates_to_history():
    session = Session("abc", render, max_turns=2)
    session.history.add_turn("u", "a")
    assert session.messages_for_llm() == session.history.to_messages_list()


def test_session_touch_updates_updated_at():
    session = Session("abc", render, max_turns=2)
    before = session.updated_at
    session.touch()
    assert session.updated_at >= before
    assert session.created_at <= session.updated_at


# --- SessionStore ----------------------------------------------------------


def make_store(max_sessions: int = 10) -> SessionStore:
    return SessionStore(render, max_turns=3, max_sessions=max_sessions)


def test_create_returns_session_with_uuid4_id():
    session = make_store().create()
    assert uuid.UUID(session.session_id).version == 4
    assert session.metadata.is_empty()


def test_get_returns_the_same_session():
    store = make_store()
    session = store.create()
    assert store.get(session.session_id) is session


def test_get_unknown_raises_not_found():
    with pytest.raises(SessionNotFoundError):
        make_store().get("nope")


def test_delete_removes_session_and_unknown_is_noop():
    store = make_store()
    session = store.create()
    store.delete(session.session_id)
    store.delete(session.session_id)
    with pytest.raises(SessionNotFoundError):
        store.get(session.session_id)


def test_cap_evicts_least_recently_used():
    store = make_store(max_sessions=2)
    first = store.create()
    second = store.create()
    store.get(first.session_id)  # first becomes most recently used
    third = store.create()
    assert len(store) == 2
    assert store.get(first.session_id) is first
    assert store.get(third.session_id) is third
    with pytest.raises(SessionNotFoundError):
        store.get(second.session_id)


def test_concurrent_creates_produce_unique_ids():
    store = make_store(max_sessions=500)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: store.create().session_id, range(200)))
    assert len(set(ids)) == 200
    assert len(store) == 200


def test_get_session_store_is_a_singleton_honouring_settings():
    from cag_estimate.dependencies import get_session_store

    get_session_store.cache_clear()
    assert get_session_store() is get_session_store()
    get_session_store.cache_clear()
