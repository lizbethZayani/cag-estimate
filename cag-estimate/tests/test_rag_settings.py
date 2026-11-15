import pytest

from cag_estimate.config import Settings

ENV_NAMES = ("GENERATION_MODEL", "RAG_TOP_K", "GENERATION_MAX_OUTPUT_TOKENS")


@pytest.fixture(autouse=True)
def api_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_generation_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.generation_model == "gpt-4o-mini"
    assert settings.rag_top_k == 10
    assert settings.generation_max_output_tokens == 4000


def test_generation_settings_are_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GENERATION_MODEL", "gpt-4o")
    monkeypatch.setenv("RAG_TOP_K", "4")
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "1500")

    settings = Settings(_env_file=None)

    assert settings.generation_model == "gpt-4o"
    assert settings.rag_top_k == 4
    assert settings.generation_max_output_tokens == 1500
