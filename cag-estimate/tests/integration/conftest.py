"""Fixtures shared by the live pgvector tests (compose container, never production)."""

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncpg
import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from cag_estimate.config import get_settings

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://estimator:estimator@localhost:5433/estimator_test"
)
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"
CONNECT_TIMEOUT_SECONDS = 2
SessionFactory = async_sessionmaker[AsyncSession]


async def probe_database() -> None:
    connection = await asyncpg.connect(
        TEST_DATABASE_URL.replace("+asyncpg", ""), timeout=CONNECT_TIMEOUT_SECONDS
    )
    await connection.close()


async def scalar(factory: SessionFactory, sql: str) -> object:
    async with factory() as session:
        return await session.scalar(text(sql))


@pytest.fixture(scope="module")
def migrated_database() -> Iterator[str]:
    try:
        asyncio.run(probe_database())
    except (OSError, asyncpg.PostgresError, TimeoutError) as error:
        pytest.skip(f"pgvector test database not reachable at {TEST_DATABASE_URL}: {error}")

    patch = pytest.MonkeyPatch()
    patch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    patch.setenv("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "test-key"))
    get_settings.cache_clear()
    config = Config(str(ALEMBIC_INI))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield TEST_DATABASE_URL
    patch.undo()
    get_settings.cache_clear()


@pytest.fixture
async def session_factory(migrated_database: str) -> AsyncIterator[SessionFactory]:
    engine = create_async_engine(migrated_database)
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE documents RESTART IDENTITY CASCADE"))
    yield async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    await engine.dispose()
