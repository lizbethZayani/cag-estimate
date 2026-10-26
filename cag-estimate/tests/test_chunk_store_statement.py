from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

from cag_estimate.db.models import ChunkRow
from cag_estimate.embedding_pipeline.repository import (
    ChunkHit,
    ChunkStore,
    build_search_statement,
)
from cag_estimate.embedding_pipeline.schemas import EmbeddedChunk


def compiled_sql(k: int = 5) -> str:
    statement = build_search_statement([0.5, 0.25], k)
    return str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


def make_chunk(chunk_id: str) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"budget_id": "b1", "tech": ["python"]},
        token_count=3,
        embedding=[0.1, 0.2],
    )


def fake_session() -> MagicMock:
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


def test_search_statement_orders_by_cosine_distance() -> None:
    sql = compiled_sql()

    assert "<=>" in sql
    assert "<->" not in sql
    assert "<#>" not in sql
    assert "ORDER BY" in sql
    assert "LIMIT 5" in sql


def test_search_statement_skips_chunks_without_embedding() -> None:
    assert "chunks.embedding IS NOT NULL" in compiled_sql()


def test_search_statement_limit_follows_k() -> None:
    assert "LIMIT 12" in compiled_sql(12)


async def test_add_chunks_builds_rows_and_does_not_commit() -> None:
    session = fake_session()
    chunks = [make_chunk("c1"), make_chunk("c2")]

    count = await ChunkStore.add_chunks(
        session, document_id=7, chunks=chunks, chunk_type="component"
    )

    assert count == 2
    (rows,) = session.add_all.call_args.args
    assert all(isinstance(row, ChunkRow) for row in rows)
    assert [row.content for row in rows] == ["text c1", "text c2"]
    assert rows[0].document_id == 7
    assert rows[0].chunk_type == "component"
    assert rows[0].metadata_ == {"budget_id": "b1", "tech": ["python"]}
    assert rows[0].embedding == [0.1, 0.2]
    session.commit.assert_not_called()


async def test_add_document_flushes_and_returns_the_id() -> None:
    session = fake_session()
    session.add.side_effect = lambda row: setattr(row, "id", 42)

    document_id = await ChunkStore.add_document(
        session, source_path="a.json", document_type="budget", metadata={"k": 1}
    )

    assert document_id == 42
    session.flush.assert_awaited_once()
    session.commit.assert_not_called()


async def test_find_document_id_returns_none_when_missing() -> None:
    session = fake_session()
    session.scalar = AsyncMock(return_value=None)

    assert await ChunkStore.find_document_id(session, "missing.json") is None


async def test_search_maps_rows_to_hits() -> None:
    session = fake_session()
    row = MagicMock(
        id=1, document_id=2, chunk_type="component", content="c", metadata_={"a": 1}, distance=0.25
    )
    session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    hits = await ChunkStore.search(session, [0.1, 0.2], 3)

    assert hits == [
        ChunkHit(
            chunk_id=1,
            document_id=2,
            chunk_type="component",
            content="c",
            distance=0.25,
            metadata={"a": 1},
        )
    ]
