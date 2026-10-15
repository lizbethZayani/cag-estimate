"""Schemas of the embedding pipeline: historical budgets in, embedded chunks out.

Unknown fields in the input are ignored so a richer budget file still loads.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

NonEmptyStr = Field(min_length=1)


class ClientMetadata(BaseModel):
    """Who the budget was written for."""

    name: str = NonEmptyStr
    sector: str = NonEmptyStr
    country: str = NonEmptyStr


class BudgetComponent(BaseModel):
    """One estimated piece of work inside a budget; it becomes exactly one chunk."""

    component_id: str = NonEmptyStr
    name: str = NonEmptyStr
    description: str = NonEmptyStr
    tech_stack: list[str] = Field(default_factory=list)
    estimated_hours: int | float = Field(ge=0)
    complexity: str = NonEmptyStr
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    """A historical budget with its components."""

    budget_id: str = NonEmptyStr
    client_metadata: ClientMetadata
    project_summary: str = NonEmptyStr
    main_technology: str = NonEmptyStr
    year: int
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
    """Body of ``POST /embeddings/ingest``."""

    budgets: list[Budget]

    @field_validator("budgets")
    @classmethod
    def _require_budgets(cls, value: list[Budget]) -> list[Budget]:
        if not value:
            raise ValueError("send at least one budget to ingest")
        return value


class IngestStats(BaseModel):
    """Totals of one ingestion run; the cost is an estimate from token counts."""

    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    """Response of ``POST /embeddings/ingest``."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
