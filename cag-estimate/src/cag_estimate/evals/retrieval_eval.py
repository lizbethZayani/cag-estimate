"""Retrieval evaluation: golden set, precision@k, aggregation and markdown tables.

Pure logic compared across four search configurations; the only I/O is the golden set loader.
This is separate from the estimation eval in the sibling modules.
"""

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

GOLDEN_SET_PATH = Path(__file__).parent / "retrieval_golden_set.json"
DEFAULT_K = 5

BudgetId = Annotated[str, StringConstraints(pattern=r"^BUD-\d{4}-\d{3}$")]


class GoldenQuery(BaseModel):
    """A project description and the budgets a good retriever must surface for it."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_budget_ids: list[BudgetId] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("relevant_budget_ids")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("relevant_budget_ids must not contain duplicates")
        return value


class GoldenSet(BaseModel):
    """The annotated queries the retrieval configurations are measured on."""

    version: int
    queries: list[GoldenQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_query_ids(self) -> "GoldenSet":
        ids = [q.id for q in self.queries]
        if len(set(ids)) != len(ids):
            raise ValueError("query ids must be unique")
        return self


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> GoldenSet:
    return GoldenSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RetrievalConfig:
    """One way of calling ``POST /search``."""

    name: str
    search_mode: Literal["vector", "hybrid"]
    rerank: bool


CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig("A", "vector", False),
    RetrievalConfig("B", "hybrid", False),
    RetrievalConfig("C", "vector", True),
    RetrievalConfig("D", "hybrid", True),
)


@dataclass(frozen=True)
class QueryMeasurement:
    """Precision and timings of one configuration on one golden query."""

    config: str
    query_id: str
    precision: float
    latencies_ms: tuple[float, ...]
    server_times_ms: tuple[int, ...]


def precision_at_k(
    returned_budget_ids: Sequence[str], relevant: Collection[str], k: int = DEFAULT_K
) -> float:
    """Share of the first ``k`` results whose parent budget is relevant.

    Missing slots (fewer than ``k`` results) count as not relevant, so the divisor is always
    ``k``. Caveat: results are component chunks, so one relevant budget can fill several of
    the ``k`` slots and a perfect score does not mean ``k`` distinct budgets were found.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    hits = sum(1 for budget_id in returned_budget_ids[:k] if budget_id in relevant)
    return hits / k


def median_latency_ms(latencies_ms: Sequence[float]) -> float:
    if not latencies_ms:
        raise ValueError("no latencies to aggregate")
    return float(median(latencies_ms))


def _of_config(measurements: Sequence[QueryMeasurement], config: str) -> list[QueryMeasurement]:
    return [m for m in measurements if m.config == config]


def mean_precision(measurements: Sequence[QueryMeasurement], config: str) -> float:
    return fmean(m.precision for m in _of_config(measurements, config))


def _comparison_row(config: RetrievalConfig, measurements: Sequence[QueryMeasurement]) -> str:
    own = _of_config(measurements, config.name)
    latency = median_latency_ms([ms for m in own for ms in m.latencies_ms])
    server = median_latency_ms([ms for m in own for ms in m.server_times_ms])
    return (
        f"| {config.name} | {config.search_mode} | {'on' if config.rerank else 'off'} "
        f"| {mean_precision(measurements, config.name):.2f} | {latency:.1f} | {server:.1f} |"
    )


def _comparison_table(
    configs: Sequence[RetrievalConfig], measurements: Sequence[QueryMeasurement]
) -> str:
    header = (
        "| Config | Search | Reranking | Precision@5 (mean) | Latency median (ms) "
        "| Server search_time median (ms) |"
    )
    separator = "|---|---|---|---|---|---|"
    rows = [_comparison_row(config, measurements) for config in configs]
    return "\n".join([header, separator, *rows])


def _per_query_table(
    configs: Sequence[RetrievalConfig], measurements: Sequence[QueryMeasurement]
) -> str:
    header = "| Query | " + " | ".join(c.name for c in configs) + " |"
    separator = "|---" * (len(configs) + 1) + "|"
    precision = {(m.config, m.query_id): m.precision for m in measurements}
    query_ids = list(dict.fromkeys(m.query_id for m in measurements))
    rows = [
        f"| {query_id} | "
        + " | ".join(f"{precision[(c.name, query_id)]:.2f}" for c in configs)
        + " |"
        for query_id in query_ids
    ]
    return "\n".join([header, separator, *rows])


def render_markdown_table(
    measurements: Sequence[QueryMeasurement], configs: Sequence[RetrievalConfig] = CONFIGS
) -> str:
    """The configuration comparison table followed by the per-query precision table."""
    return (
        _comparison_table(configs, measurements) + "\n\n" + _per_query_table(configs, measurements)
    )
