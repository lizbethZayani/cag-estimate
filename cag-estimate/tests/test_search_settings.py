import pytest
from pydantic import ValidationError

from cag_estimate.config import Settings


@pytest.fixture(autouse=True)
def api_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for name in ("SEARCH_MODE", "RERANK_ENABLED", "RERANK_MODEL", "RRF_K", "RECALL_K"):
        monkeypatch.delenv(name, raising=False)


def test_search_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.search_mode == "vector"
    assert settings.rerank_enabled is False
    assert settings.rerank_model == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert settings.rrf_k == 60
    assert settings.recall_k == 50


def test_search_settings_are_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MODE", "hybrid")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_MODEL", "some/other-model")
    monkeypatch.setenv("RRF_K", "10")
    monkeypatch.setenv("RECALL_K", "20")

    settings = Settings(_env_file=None)

    assert settings.search_mode == "hybrid"
    assert settings.rerank_enabled is True
    assert settings.rerank_model == "some/other-model"
    assert settings.rrf_k == 10
    assert settings.recall_k == 20


def test_invalid_search_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MODE", "fuzzy")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
