"""Lazy async engine and session factory built from ``Settings.database_url``."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cag_estimate.config import get_settings


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    url = get_settings().database_url
    if url is None:
        raise RuntimeError("DATABASE_URL is not set; the database is not configured.")
    return create_async_engine(url, pool_pre_ping=True)


@lru_cache
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        create_async_engine_from_settings(), expire_on_commit=False, autoflush=False
    )


async def dispose_async_engine() -> None:
    """Dispose the engine only if it was ever created, then reset the caches."""
    if create_async_engine_from_settings.cache_info().currsize:
        await create_async_engine_from_settings().dispose()
    get_async_session_factory.cache_clear()
    create_async_engine_from_settings.cache_clear()
