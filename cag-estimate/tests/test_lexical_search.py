"""Offline tests for the lexical (full-text) branch of the chunk store."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from cag_estimate.embedding_pipeline.repository import (
    ChunkHit,
    ChunkStore,
    build_lexical_statement,
    build_or_tsquery_text,
)


def compile_statement(query_text: str = "oauth | jwt", k: int = 5, literal: bool = False) -> str:
    statement = build_lexical_statement(query_text, k)
    kwargs = {"literal_binds": True} if literal else {}
    return str(statement.compile(dialect=postgresql.dialect(), compile_kwargs=kwargs))


def test_or_text_joins_lowercased_tokens_with_or() -> None:
    assert build_or_tsquery_text("OAuth 2.0 JWT authentication") == "oauth | jwt | authentication"


def test_or_text_strips_punctuation_and_splits_on_it() -> None:
    assert build_or_tsquery_text("e-commerce, (checkout)!") == "commerce | checkout"


def test_or_text_dedupes_preserving_first_occurrence() -> None:
    assert build_or_tsquery_text("Store store STORE catalog store") == "store | catalog"


def test_or_text_drops_single_character_tokens() -> None:
    assert build_or_tsquery_text("a b cd e fg") == "cd | fg"


@pytest.mark.parametrize("query", ["", "   ", "!!! ???", "a b c"])
def test_or_text_is_empty_when_no_token_is_left(query: str) -> None:
    assert build_or_tsquery_text(query) == ""


def test_or_text_neutralises_sql_and_tsquery_syntax() -> None:
    text = build_or_tsquery_text("a'; drop table & ! (x:*) <-> y")

    assert text == "drop | table"
    assert not any(char in text for char in "'&!():*<>-;")


def test_lexical_statement_uses_or_tsquery_and_rank() -> None:
    sql = compile_statement(literal=True)

    assert "to_tsquery('english'" in sql
    assert "@@" in sql
    assert "ts_rank_cd" in sql
    assert "ORDER BY" in sql
    assert "LIMIT 5" in sql


def test_lexical_statement_breaks_rank_ties_by_chunk_id() -> None:
    sql = compile_statement(literal=True)

    order_by = sql.split("ORDER BY", 1)[1]
    assert "DESC" in order_by
    assert "chunks.id ASC" in order_by


def test_lexical_statement_exposes_the_rank_as_score() -> None:
    assert "AS score" in compile_statement()


def test_lexical_statement_binds_the_query_text() -> None:
    statement = build_lexical_statement("oauth | jwt", 5)
    sql = str(statement.compile(dialect=postgresql.dialect()))
    params = statement.compile(dialect=postgresql.dialect()).params

    assert "oauth" not in sql
    assert "oauth | jwt" in params.values()


def test_lexical_statement_limit_follows_k() -> None:
    assert "LIMIT 12" in compile_statement(k=12, literal=True)


async def test_search_lexical_skips_the_database_when_no_token_is_left() -> None:
    session = MagicMock()
    session.execute = AsyncMock()

    hits = await ChunkStore.search_lexical(session, "a ! ?", 5)

    assert hits == []
    session.execute.assert_not_called()


async def test_search_lexical_maps_rows_to_hits_without_distance() -> None:
    session = MagicMock()
    row = MagicMock(
        id=1, document_id=2, chunk_type="component", content="c", metadata_={"a": 1}, score=0.5
    )
    session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))

    hits = await ChunkStore.search_lexical(session, "oauth jwt", 3)

    assert hits == [
        ChunkHit(
            chunk_id=1,
            document_id=2,
            chunk_type="component",
            content="c",
            distance=None,
            metadata={"a": 1},
            score=0.5,
        )
    ]


def test_chunk_hit_score_defaults_to_none() -> None:
    hit = ChunkHit(
        chunk_id=1, document_id=1, chunk_type="t", content="c", distance=0.1, metadata={}
    )

    assert hit.score is None
