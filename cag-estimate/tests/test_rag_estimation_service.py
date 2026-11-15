"""Tests for the RAG estimation service: retrieve, generate, verify, retry once, reject."""

import pytest
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from cag_estimate.schemas.rag_estimation import RagEstimate
from cag_estimate.services.rag_errors import (
    DanglingCitationError,
    RagGenerationError,
    RagOutputInvalidError,
)
from cag_estimate.services.rag_estimation import RagEstimationService
from tests.rag_fakes import (
    EVIDENCE_TEXT,
    QUERY,
    FakeRetriever,
    make_estimate,
    make_item,
    make_response,
)

REAL_ID = "BUD-1::CART-001"
INVENTED_ID = "BUD-2099-999::FAKE-001"
TOP_K = 7


class FakeGenerator:
    """Replays scripted outcomes (estimates or exceptions) and records every call."""

    def __init__(self, *outcomes: RagEstimate | Exception) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, str, str | None]] = []

    async def generate(
        self, query: str, context_block: str, feedback: str | None = None
    ) -> RagEstimate:
        self.calls.append((query, context_block, feedback))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _clean_state():
    clear_contextvars()
    yield
    clear_contextvars()
    structlog.reset_defaults()


@pytest.fixture
def events() -> list[dict]:
    captured: list[dict] = []

    def capture(_logger, method, event_dict):
        captured.append({**event_dict, "level": method})
        raise structlog.DropEvent

    structlog.configure(processors=[structlog.contextvars.merge_contextvars, capture])
    return captured


def _service(generator: FakeGenerator, items=None) -> tuple[RagEstimationService, FakeRetriever]:
    retriever = FakeRetriever(make_response(items if items is not None else [make_item()]))
    return RagEstimationService(retriever, generator, top_k=TOP_K), retriever


def _event(events: list[dict], name: str) -> dict:
    (match,) = [e for e in events if e["event"] == name]
    return match


async def test_happy_path_returns_a_verified_estimate() -> None:
    estimate = make_estimate(REAL_ID)
    generator = FakeGenerator(estimate)
    service, _ = _service(generator)

    result = await service.estimate(QUERY)

    assert result.estimate is estimate
    assert result.citation_report.grounded_lines == 1
    assert [c.chunk_id for c in result.chunks] == [REAL_ID]
    assert result.prompt_version == "v1"
    assert len(generator.calls) == 1
    assert generator.calls[0][2] is None
    assert f'id="{REAL_ID}"' in generator.calls[0][1]


async def test_retrieval_uses_hybrid_without_rerank_and_the_configured_k() -> None:
    service, retriever = _service(FakeGenerator(make_estimate(REAL_ID)))

    await service.estimate(QUERY)

    (request,) = retriever.requests
    assert (request.query, request.k) == (QUERY, TOP_K)
    assert request.search_mode == "hybrid"
    assert request.rerank is False


async def test_no_usable_chunks_returns_insufficient_without_calling_the_generator() -> None:
    generator = FakeGenerator()
    service, _ = _service(generator, items=[])

    result = await service.estimate(QUERY)

    assert generator.calls == []
    assert result.estimate.lines == []
    assert result.estimate.confidence == "insufficient"
    assert "no historical budget" in (result.estimate.insufficient_context_explanation or "").lower()
    assert result.citation_report.total_lines == 0
    assert result.chunks == []


async def test_items_missing_ids_are_skipped_and_all_unusable_means_insufficient() -> None:
    generator = FakeGenerator()
    service, _ = _service(generator, items=[make_item(component_id=None)])

    result = await service.estimate(QUERY)

    assert generator.calls == []
    assert result.estimate.confidence == "insufficient"


async def test_dangling_then_valid_is_accepted_after_one_corrective_retry(events) -> None:
    good = make_estimate(REAL_ID)
    generator = FakeGenerator(make_estimate(INVENTED_ID), good)
    service, _ = _service(generator)

    result = await service.estimate(QUERY, request_id="req-1")

    assert result.estimate is good
    assert len(generator.calls) == 2
    feedback = generator.calls[1][2]
    assert feedback is not None and INVENTED_ID in feedback
    report_event = _event(events, "citation_report")
    assert report_event["attempts"] == 2
    assert report_event["request_id"] == "req-1"
    assert report_event["dangling_chunk_ids"] == []


async def test_dangling_twice_is_rejected(events) -> None:
    generator = FakeGenerator(make_estimate(INVENTED_ID), make_estimate(INVENTED_ID))
    service, _ = _service(generator)

    with pytest.raises(DanglingCitationError) as info:
        await service.estimate(QUERY, request_id="req-2")

    assert info.value.report.dangling_chunk_ids == [INVENTED_ID]
    assert len(generator.calls) == 2
    rejected = _event(events, "citations_rejected")
    assert rejected["level"] == "warning"
    assert rejected["dangling_chunk_ids"] == [INVENTED_ID]
    assert rejected["attempts"] == 2
    assert rejected["request_id"] == "req-2"
    assert not [e for e in events if e["event"] == "citation_report"]


async def test_invalid_output_then_valid_is_accepted(events) -> None:
    good = make_estimate(REAL_ID)
    generator = FakeGenerator(RagOutputInvalidError("a grounded line must cite a source"), good)
    service, _ = _service(generator)

    result = await service.estimate(QUERY)

    assert result.estimate is good
    feedback = generator.calls[1][2]
    assert feedback is not None and "a grounded line must cite a source" in feedback
    assert _event(events, "citation_report")["attempts"] == 2


async def test_invalid_twice_raises_the_output_error() -> None:
    generator = FakeGenerator(RagOutputInvalidError("rule one"), RagOutputInvalidError("rule two"))
    service, _ = _service(generator)

    with pytest.raises(RagOutputInvalidError, match="rule two"):
        await service.estimate(QUERY)

    assert len(generator.calls) == 2


async def test_a_generation_error_is_not_retried() -> None:
    generator = FakeGenerator(RagGenerationError("provider down"))
    service, _ = _service(generator)

    with pytest.raises(RagGenerationError):
        await service.estimate(QUERY)

    assert len(generator.calls) == 1


async def test_invalid_then_dangling_is_rejected_as_dangling() -> None:
    generator = FakeGenerator(RagOutputInvalidError("rule"), make_estimate(INVENTED_ID))
    service, _ = _service(generator)

    with pytest.raises(DanglingCitationError):
        await service.estimate(QUERY)


async def test_request_id_falls_back_to_the_bound_context(events) -> None:
    bind_contextvars(request_id="ctx-9")
    service, _ = _service(FakeGenerator(make_estimate(REAL_ID)))

    await service.estimate(QUERY)

    assert _event(events, "citation_report")["request_id"] == "ctx-9"


async def test_accepted_report_event_carries_the_counts(events) -> None:
    service, _ = _service(FakeGenerator(make_estimate(REAL_ID)))

    await service.estimate(QUERY, request_id="req-3")

    event = _event(events, "citation_report")
    assert event["level"] == "info"
    assert event["total_lines"] == 1
    assert event["grounded_lines"] == 1
    assert event["dangling_lines"] == 0
    assert event["insufficient_lines"] == 0
    assert event["attempts"] == 1
    assert event["prompt_version"] == "v1"


async def test_the_no_chunk_path_logs_a_report_with_zero_attempts(events) -> None:
    service, _ = _service(FakeGenerator(), items=[])

    await service.estimate(QUERY, request_id="req-4")

    event = _event(events, "citation_report")
    assert (event["total_lines"], event["attempts"]) == (0, 0)


async def test_logs_never_contain_the_query_or_the_evidence(events) -> None:
    generator = FakeGenerator(make_estimate(INVENTED_ID), make_estimate(REAL_ID))
    service, _ = _service(generator)

    await service.estimate(QUERY, request_id="req-5")

    dumped = repr(events)
    assert QUERY not in dumped
    assert EVIDENCE_TEXT not in dumped
    assert "Shopping cart" not in dumped
