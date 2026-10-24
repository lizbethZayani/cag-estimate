"""Persistence and cosine search of embedded chunks; the caller owns the transaction."""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from cag_estimate.db.models import ChunkRow, DocumentRow
from cag_estimate.embedding_pipeline.schemas import EmbeddedChunk


@dataclass(frozen=True)
class ChunkHit:
    """One search result; ``distance`` is the cosine distance (0 = identical direction)."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]


def build_search_statement(query_vector: list[float], k: int) -> Select[Any]:
    """Top ``k`` chunks by cosine distance (``<=>``) to ``query_vector``, nearest first."""
    distance = ChunkRow.embedding.cosine_distance(query_vector)
    return (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.metadata_,
            distance.label("distance"),
        )
        # Not a metadata filter: a chunk without embedding has no distance to rank by.
        .where(ChunkRow.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )


class ChunkStore:
    """Stateless repository: every method takes the session and never commits."""

    @staticmethod
    async def find_document_id(session: AsyncSession, source_path: str) -> int | None:
        statement = select(DocumentRow.id).where(DocumentRow.source_path == source_path).limit(1)
        return await session.scalar(statement)

    @staticmethod
    async def add_document(
        session: AsyncSession,
        *,
        source_path: str,
        document_type: str,
        metadata: dict[str, Any],
    ) -> int:
        row = DocumentRow(source_path=source_path, document_type=document_type, metadata_=metadata)
        session.add(row)
        await session.flush()
        return row.id

    @staticmethod
    async def add_chunks(
        session: AsyncSession,
        *,
        document_id: int,
        chunks: list[EmbeddedChunk],
        chunk_type: str,
    ) -> int:
        rows = [
            ChunkRow(
                document_id=document_id,
                chunk_type=chunk_type,
                content=chunk.text,
                embedding=chunk.embedding,
                metadata_=chunk.metadata,
            )
            for chunk in chunks
        ]
        session.add_all(rows)
        return len(rows)

    @staticmethod
    async def search(session: AsyncSession, query_vector: list[float], k: int) -> list[ChunkHit]:
        result = await session.execute(build_search_statement(query_vector, k))
        return [
            ChunkHit(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=row.distance,
                metadata=row.metadata_,
            )
            for row in result.all()
        ]
