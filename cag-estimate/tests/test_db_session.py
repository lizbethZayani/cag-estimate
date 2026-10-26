from types import SimpleNamespace

import pytest

from cag_estimate.config import get_settings
from cag_estimate.db.session import (
    create_async_engine_from_settings,
    dispose_async_engine,
    get_async_session_factory,
)

URL = "postgresql+asyncpg://user:pass@localhost:5432/unit"


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()
    create_async_engine_from_settings.cache_clear()
    get_async_session_factory.cache_clear()
    yield
    get_settings.cache_clear()
    create_async_engine_from_settings.cache_clear()
    get_async_session_factory.cache_clear()


def test_engine_requires_a_database_url(monkeypatch) -> None:
    monkeypatch.setattr(
        "cag_estimate.db.session.get_settings", lambda: SimpleNamespace(database_url=None)
    )

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_async_engine_from_settings()


def test_engine_and_factory_are_cached_and_configured(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)

    engine = create_async_engine_from_settings()
    factory = get_async_session_factory()

    assert engine is create_async_engine_from_settings()
    assert factory is get_async_session_factory()
    assert engine.url.render_as_string(hide_password=False) == URL
    assert factory.kw["expire_on_commit"] is False
    assert factory.kw["autoflush"] is False


async def test_dispose_without_engine_is_a_noop() -> None:
    await dispose_async_engine()

    assert create_async_engine_from_settings.cache_info().currsize == 0


async def test_dispose_resets_the_caches(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    first = create_async_engine_from_settings()

    await dispose_async_engine()

    assert create_async_engine_from_settings.cache_info().currsize == 0
    assert create_async_engine_from_settings() is not first
