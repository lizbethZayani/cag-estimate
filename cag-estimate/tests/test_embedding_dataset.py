"""Offline checks of the official sample dataset against the embedding pipeline."""

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.schemas import (
    Budget,
    Chunk,
    Complexity,
    IngestRequest,
    Sector,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
EXPECTED_BUDGETS = 15
EXPECTED_CHUNKS = 52
METADATA_KEYS = {
    "budget_id",
    "component_id",
    "client_sector",
    "main_technology",
    "year",
    "complexity",
    "estimated_hours",
}
CHUNK_ID_SHAPE = re.compile(r"^BUD-[^:]+::[^:]+$")
FIRST_HEADER = "[Project: Mobile banking API with OAuth 2.0 authentication and PSD2 compliance]"


class WordCounter:
    """Offline token counter: one token per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture(scope="module")
def raw_dataset() -> Any:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def requests(raw_dataset: Any) -> list[IngestRequest]:
    return [
        IngestRequest(
            source_path=f"data/budgets/{budget['budget_id']}.json",
            document_type="historical_budget",
            content=budget,
        )
        for budget in raw_dataset
    ]


@pytest.fixture(scope="module")
def budgets(requests: list[IngestRequest]) -> list[Budget]:
    return [request.content for request in requests]


@pytest.fixture(scope="module")
def chunks(budgets: list[Budget]) -> list[Chunk]:
    return JSONStructuralChunker(token_counter=WordCounter()).chunk(budgets)


def test_file_is_a_json_list(raw_dataset):
    assert isinstance(raw_dataset, list)


def test_all_budgets_parse_as_ingest_requests(requests):
    assert len(requests) == EXPECTED_BUDGETS
    assert len({request.source_path for request in requests}) == EXPECTED_BUDGETS


def test_one_chunk_per_component(chunks):
    assert len(chunks) == EXPECTED_CHUNKS


def test_chunk_ids_are_unique_and_well_formed(chunks):
    ids = [chunk.chunk_id for chunk in chunks]

    assert len(set(ids)) == len(ids)
    assert all(CHUNK_ID_SHAPE.match(chunk_id) for chunk_id in ids)


def test_every_chunk_has_the_seven_metadata_keys(chunks):
    assert all(set(chunk.metadata) == METADATA_KEYS for chunk in chunks)


def test_sectors_belong_to_the_closed_set(chunks):
    allowed = set(get_args(Sector))

    assert allowed
    assert {chunk.metadata["client_sector"] for chunk in chunks} <= allowed


def test_complexities_belong_to_the_closed_set(chunks):
    allowed = set(get_args(Complexity))

    assert allowed
    assert {chunk.metadata["complexity"] for chunk in chunks} <= allowed


def test_budget_totals_match_the_sum_of_their_components(budgets):
    for budget in budgets:
        assert budget.total_estimated_hours == sum(
            component.estimated_hours for component in budget.components
        ), budget.budget_id


def test_first_chunk_starts_with_the_project_summary_header(chunks):
    assert chunks[0].text.startswith(FIRST_HEADER)


def test_no_chunk_text_is_empty(chunks):
    assert all(chunk.text.strip() for chunk in chunks)
