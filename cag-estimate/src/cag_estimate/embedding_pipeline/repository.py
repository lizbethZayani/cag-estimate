"""Persistence, cosine search and full-text search of chunks; the caller owns the transaction."""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from cag_estimate.db.models import ChunkRow, DocumentRow
from cag_estimate.embedding_pipeline.schemas import EmbeddedChunk

# Must match the configuration of the generated ``content_tsv`` column (migration 0002).
TEXT_SEARCH_CONFIG = "english"
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ChunkHit:
    """One search result.

    ``distance`` is the cosine distance (0 = identical direction), None for a lexical-only
    hit. ``score`` is the ranking score of the branch that produced the hit (``ts_rank_cd``
    for the lexical branch), None when that branch has no score.
    """

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float | None
    metadata: dict[str, Any]
    score: float | None = None


def build_or_tsquery_text(query: str) -> str:
    """Turn free text into an OR ``tsquery`` string such as ``"oauth | jwt"``.

    ``plainto_tsquery`` ANDs every lexeme, so a long project description would match
    nothing; OR semantics ranked by ``ts_rank_cd`` give usable lexical recall. Keeping only
    lowercase alphanumeric tokens makes ``to_tsquery`` syntax errors impossible. Single
    characters are dropped and duplicates removed (first occurrence wins). Returns ``""``
    when no token is left.
    """
    tokens = (token for token in _TOKEN.findall(query.lower()) if len(token) > 1)
    return " | ".join(dict.fromkeys(tokens))


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


def build_lexical_statement(query_text: str, k: int) -> Select[Any]:
    """Top ``k`` chunks whose ``content_tsv`` matches the OR ``tsquery`` text, best rank first."""
    # The configuration is a constant, inlined so the statement also renders with literal binds.
    tsquery = func.to_tsquery(literal_column(f"'{TEXT_SEARCH_CONFIG}'"), query_text)
    rank = func.ts_rank_cd(ChunkRow.content_tsv, tsquery)
    return (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.metadata_,
            rank.label("score"),
        )
        .where(ChunkRow.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc(), ChunkRow.id.asc())
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

    @staticmethod
    async def search_lexical(session: AsyncSession, query_text: str, k: int) -> list[ChunkHit]:
        """Full-text search; empty without touching the database when no token is left."""
        or_text = build_or_tsquery_text(query_text)
        if not or_text:
            return []
        result = await session.execute(build_lexical_statement(or_text, k))
        return [
            ChunkHit(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=None,
                metadata=row.metadata_,
                score=row.score,
            )
            for row in result.all()
        ]
