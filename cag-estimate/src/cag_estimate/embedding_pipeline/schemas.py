"""Schemas of the embedding pipeline: historical budgets in, embedded chunks and ingest stats out.

Unknown fields in the input are ignored so a richer budget file still loads.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

NonEmptyStr = Field(min_length=1)

Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]
MIN_YEAR = 2000
MAX_YEAR = 2100


class ClientMetadata(BaseModel):
    """Who the budget was written for."""

    name: str = NonEmptyStr
    sector: Sector
    country: str = NonEmptyStr


class BudgetComponent(BaseModel):
    """One estimated piece of work inside a budget; it becomes exactly one chunk."""

    component_id: str = NonEmptyStr
    name: str = NonEmptyStr
    description: str = NonEmptyStr
    tech_stack: list[str] = Field(default_factory=list)
    estimated_hours: int | float = Field(ge=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    """A historical budget with its components."""

    budget_id: str = NonEmptyStr
    client_metadata: ClientMetadata
    project_summary: str = NonEmptyStr
    main_technology: str = NonEmptyStr
    year: int = Field(ge=MIN_YEAR, le=MAX_YEAR)
    total_estimated_hours: int | float = Field(ge=0)
    components: list[BudgetComponent]

    @field_validator("components")
    @classmethod
    def _require_components(cls, value: list[BudgetComponent]) -> list[BudgetComponent]:
        if not value:
            raise ValueError("a budget needs at least one component to be chunked")
        return value


class Chunk(BaseModel):
    """A retrievable unit: ``text`` is embedded, ``metadata`` is only used to filter."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    """A chunk together with its embedding vector."""

    embedding: list[float]


class IngestRequest(BaseModel):
    """Body of ``POST /embeddings/ingest``: one budget and where it came from."""

    source_path: str = NonEmptyStr
    document_type: str = Field(min_length=1, max_length=50)
    content: Budget


class IngestResponse(BaseModel):
    """Response of ``POST /embeddings/ingest``."""

    document_id: int
    chunks_created: int = Field(ge=0)
    embedding_dimension: int
    ingestion_time_ms: int = Field(ge=0)
