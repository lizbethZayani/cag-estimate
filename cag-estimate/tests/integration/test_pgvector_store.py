"""Live tests against the compose pgvector container (never the production database)."""

import asyncio
import math
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncpg
import pytest
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from cag_estimate.config import get_settings
from cag_estimate.db.models import EMBEDDING_DIMENSIONS, ChunkRow, DocumentRow
from cag_estimate.embedding_pipeline.repository import ChunkStore
from cag_estimate.embedding_pipeline.schemas import EmbeddedChunk

pytestmark = pytest.mark.integration

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


def vector(*head: float) -> list[float]:
    """A 1536-d vector that is zero except for its first coordinates."""
    return [*head, *([0.0] * (EMBEDDING_DIMENSIONS - len(head)))]


def make_chunk(chunk_id: str, embedding: list[float] | None) -> EmbeddedChunk:
    return EmbeddedChunk.model_construct(
        chunk_id=chunk_id,
        text=f"content {chunk_id}",
        metadata={"budget_id": "b1", "tech_stack": ["python", "fastapi"], "hours": 12.5},
        token_count=4,
        embedding=embedding,
    )


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return 1 - dot / (math.hypot(*a) * math.hypot(*b))


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


async def seed(factory: SessionFactory, chunks: list[EmbeddedChunk]) -> int:
    async with factory() as session, session.begin():
        document_id = await ChunkStore.add_document(
            session, source_path="budgets/b1.json", document_type="budget", metadata={"year": 2024}
        )
        await ChunkStore.add_chunks(
            session, document_id=document_id, chunks=chunks, chunk_type="component"
        )
    return document_id


async def scalar(factory: SessionFactory, sql: str) -> object:
    async with factory() as session:
        return await session.scalar(text(sql))


def test_models_match_the_migration(migrated_database: str) -> None:
    command.check(Config(str(ALEMBIC_INI)))


async def test_document_and_chunks_round_trip(session_factory: SessionFactory) -> None:
    embeddings = [vector(1.0, 0.5), vector(0.25, 0.125), vector(0.0, 0.75)]
    chunks = [make_chunk(f"c{i}", embedding) for i, embedding in enumerate(embeddings)]

    document_id = await seed(session_factory, chunks)

    async with session_factory() as session:
        document = await session.get(DocumentRow, document_id)
        rows = (await session.scalars(select(ChunkRow).order_by(ChunkRow.id))).all()

    assert document is not None
    assert document.metadata_ == {"year": 2024}
    assert document.ingested_at is not None
    assert [row.id for row in rows] == [1, 2, 3]
    assert {row.document_id for row in rows} == {document_id}
    assert rows[0].metadata_ == chunks[0].metadata
    assert rows[0].content == "content c0"
    for row, expected in zip(rows, embeddings, strict=True):
        assert list(row.embedding) == pytest.approx(expected, abs=1e-6)


async def test_find_document_id(session_factory: SessionFactory) -> None:
    document_id = await seed(session_factory, [make_chunk("c0", vector(1.0))])

    async with session_factory() as session:
        assert await ChunkStore.find_document_id(session, "budgets/b1.json") == document_id
        assert await ChunkStore.find_document_id(session, "unknown.json") is None


async def test_deleting_a_document_removes_its_chunks(session_factory: SessionFactory) -> None:
    document_id = await seed(session_factory, [make_chunk("c0", vector(1.0))])
    assert await scalar(session_factory, "SELECT count(*) FROM chunks") == 1

    async with session_factory() as session, session.begin():
        await session.execute(text(f"DELETE FROM documents WHERE id = {document_id}"))

    assert await scalar(session_factory, "SELECT count(*) FROM chunks") == 0


async def test_rollback_leaves_no_document(session_factory: SessionFactory) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async with session_factory() as session, session.begin():
            await ChunkStore.add_document(
                session, source_path="rolled-back.json", document_type="budget", metadata={}
            )
            raise RuntimeError("boom")

    assert await scalar(session_factory, "SELECT count(*) FROM documents") == 0


async def test_search_orders_by_cosine_distance(session_factory: SessionFactory) -> None:
    near, mid, far = vector(1.0), vector(0.9, 0.1), vector(0.0, 1.0)
    query = vector(1.0)
    await seed(
        session_factory,
        [make_chunk("far", far), make_chunk("near", near), make_chunk("mid", mid)],
    )

    async with session_factory() as session:
        hits = await ChunkStore.search(session, query, 10)

    assert [hit.content for hit in hits] == ["content near", "content mid", "content far"]
    for hit, vector_ in zip(hits, [near, mid, far], strict=True):
        assert hit.distance == pytest.approx(cosine_distance(query, vector_), abs=1e-6)
    assert hits[0].metadata["tech_stack"] == ["python", "fastapi"]
    assert hits[0].chunk_type == "component"


async def test_search_limits_to_k_and_skips_chunks_without_embedding(
    session_factory: SessionFactory,
) -> None:
    await seed(
        session_factory,
        [
            make_chunk("none", None),
            make_chunk("a", vector(1.0)),
            make_chunk("b", vector(0.9, 0.1)),
            make_chunk("c", vector(0.0, 1.0)),
        ],
    )

    async with session_factory() as session:
        top_two = await ChunkStore.search(session, vector(1.0), 2)
        everything = await ChunkStore.search(session, vector(1.0), 10)

    assert [hit.content for hit in top_two] == ["content a", "content b"]
    assert len(everything) == 3
    assert "content none" not in [hit.content for hit in everything]


async def test_catalog_has_gin_but_no_vector_index(session_factory: SessionFactory) -> None:
    async with session_factory() as session:
        definitions = (
            await session.scalars(
                text("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'")
            )
        ).all()
        extension = await session.scalar(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )

    assert extension == 1
    assert any("ix_chunks_metadata_gin" in d and "USING gin" in d for d in definitions)
    assert not any("hnsw" in d.lower() or "ivfflat" in d.lower() for d in definitions)
