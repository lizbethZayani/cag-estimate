"""Offline tests for the RAG generator and estimation service dependency builders."""

from cag_estimate import dependencies
from cag_estimate.config import Settings
from cag_estimate.services.rag_estimation import RagEstimationService
from cag_estimate.services.rag_generation import RagGenerator


def _patch_settings(monkeypatch, **overrides) -> Settings:
    settings = Settings(anthropic_api_key="test", **overrides)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    return settings


def test_build_rag_generator_needs_an_openai_key(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key=None)

    assert dependencies.build_rag_generator() is None


def test_build_rag_generator_takes_the_generation_settings(monkeypatch):
    _patch_settings(
        monkeypatch,
        openai_api_key="sk-test",
        generation_model="gpt-test",
        generation_max_output_tokens=123,
    )

    generator = dependencies.build_rag_generator()

    assert isinstance(generator, RagGenerator)
    assert generator._model == "gpt-test"
    assert generator._max_output_tokens == 123


def test_get_rag_generator_is_cached(monkeypatch):
    dependencies.get_rag_generator.cache_clear()
    _patch_settings(monkeypatch, openai_api_key="sk-test")
    try:
        assert dependencies.get_rag_generator() is dependencies.get_rag_generator()
    finally:
        dependencies.get_rag_generator.cache_clear()


def test_build_rag_estimation_service_needs_a_retriever(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="sk-test")
    monkeypatch.setattr(dependencies, "get_semantic_retriever", lambda: None)
    monkeypatch.setattr(dependencies, "get_rag_generator", lambda: object())

    assert dependencies.build_rag_estimation_service() is None


def test_build_rag_estimation_service_needs_a_generator(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key=None)
    monkeypatch.setattr(dependencies, "get_semantic_retriever", lambda: object())
    monkeypatch.setattr(dependencies, "get_rag_generator", lambda: None)

    assert dependencies.build_rag_estimation_service() is None


def test_build_rag_estimation_service_is_built_with_the_settings(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="sk-test", rag_top_k=7)
    retriever, generator = object(), object()
    monkeypatch.setattr(dependencies, "get_semantic_retriever", lambda: retriever)
    monkeypatch.setattr(dependencies, "get_rag_generator", lambda: generator)

    service = dependencies.build_rag_estimation_service()

    assert isinstance(service, RagEstimationService)
    assert service._retriever is retriever
    assert service._generator is generator
    assert service._top_k == 7


def test_get_rag_estimation_service_is_cached(monkeypatch):
    dependencies.get_rag_estimation_service.cache_clear()
    _patch_settings(monkeypatch, openai_api_key="sk-test")
    monkeypatch.setattr(dependencies, "get_semantic_retriever", lambda: object())
    monkeypatch.setattr(dependencies, "get_rag_generator", lambda: object())
    try:
        first = dependencies.get_rag_estimation_service()
        assert first is not None
        assert first is dependencies.get_rag_estimation_service()
    finally:
        dependencies.get_rag_estimation_service.cache_clear()
