from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, ForeignKey, Index
from sqlalchemy.dialects import postgresql

from cag_estimate.db.base import Base
from cag_estimate.db.models import EMBEDDING_DIMENSIONS, ChunkRow, DocumentRow

DOCUMENTS = DocumentRow.__table__
CHUNKS = ChunkRow.__table__


def index_columns(table) -> dict[str, list[str]]:
    return {
        index.name: [column.name for column in index.columns]
        for index in table.indexes
        if isinstance(index, Index)
    }


def test_metadata_has_exactly_the_two_tables() -> None:
    assert set(Base.metadata.tables) == {"documents", "chunks"}


def test_documents_columns() -> None:
    assert [column.name for column in DOCUMENTS.columns] == [
        "id",
        "source_path",
        "document_type",
        "ingested_at",
        "metadata",
    ]
    assert isinstance(DOCUMENTS.c.id.type, BigInteger)
    assert DOCUMENTS.c.document_type.type.length == 50
    assert DOCUMENTS.c.ingested_at.type.timezone is True
    assert not DOCUMENTS.c.source_path.nullable


def test_chunks_columns() -> None:
    assert [column.name for column in CHUNKS.columns] == [
        "id",
        "document_id",
        "chunk_type",
        "content",
        "embedding",
        "metadata",
        "created_at",
    ]
    assert CHUNKS.c.chunk_type.type.length == 50
    assert CHUNKS.c.created_at.type.timezone is True


def test_metadata_attribute_maps_to_the_metadata_column() -> None:
    assert DocumentRow.__mapper__.columns["metadata_"].name == "metadata"
    assert ChunkRow.__mapper__.columns["metadata_"].name == "metadata"


def test_metadata_columns_are_jsonb_with_an_empty_object_default() -> None:
    for table in (DOCUMENTS, CHUNKS):
        column = table.c.metadata
        assert isinstance(column.type, postgresql.JSONB)
        assert not column.nullable
        assert str(column.server_default.arg) == "'{}'::jsonb"


def test_embedding_is_a_nullable_vector_of_1536_dimensions() -> None:
    column = CHUNKS.c.embedding

    assert isinstance(column.type, Vector)
    assert column.type.dim == EMBEDDING_DIMENSIONS == 1536
    assert column.nullable


def test_chunks_reference_documents_with_cascade() -> None:
    (foreign_key,) = CHUNKS.c.document_id.foreign_keys

    assert isinstance(foreign_key, ForeignKey)
    assert foreign_key.column.table is DOCUMENTS
    assert foreign_key.ondelete == "CASCADE"
    assert not CHUNKS.c.document_id.nullable


def test_indexes_match_the_migration() -> None:
    assert index_columns(DOCUMENTS) == {"ix_documents_source_path": ["source_path"]}
    assert index_columns(CHUNKS) == {
        "ix_chunks_document_id": ["document_id"],
        "ix_chunks_chunk_type": ["chunk_type"],
        "ix_chunks_metadata_gin": ["metadata"],
    }
    gin = next(i for i in CHUNKS.indexes if i.name == "ix_chunks_metadata_gin")
    assert gin.dialect_options["postgresql"]["using"] == "gin"
    assert not next(i for i in DOCUMENTS.indexes).unique


def test_there_is_no_vector_index() -> None:
    for index in CHUNKS.indexes:
        assert "embedding" not in [column.name for column in index.columns]
        assert index.dialect_options["postgresql"]["using"] not in ("hnsw", "ivfflat")


def test_relationships_cascade_deletes_to_chunks() -> None:
    relationship = DocumentRow.__mapper__.relationships["chunks"]

    assert relationship.cascade.delete_orphan
    assert relationship.cascade.delete
    assert relationship.passive_deletes is True
    assert ChunkRow.__mapper__.relationships["document"].mapper.class_ is DocumentRow
