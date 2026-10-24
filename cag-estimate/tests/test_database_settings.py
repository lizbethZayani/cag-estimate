from cag_estimate.config import Settings

DATABASE_URL = "postgresql+asyncpg://user:pass@db:5432/app"


def test_database_url_is_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)

    assert Settings().database_url == DATABASE_URL


def test_database_url_defaults_to_none(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert Settings(_env_file=None).database_url is None
