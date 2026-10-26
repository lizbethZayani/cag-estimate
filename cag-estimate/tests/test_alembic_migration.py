import io
from pathlib import Path

import pytest
from alembic.config import Config

from alembic import command
from cag_estimate.config import get_settings

PROJECT_DIR = Path(__file__).resolve().parent.parent
ALEMBIC_INI = PROJECT_DIR / "alembic.ini"
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/offline"


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def render_sql(run) -> str:
    buffer = io.StringIO()
    config = Config(str(ALEMBIC_INI), output_buffer=buffer)
    run(config)
    return buffer.getvalue()


@pytest.fixture
def upgrade_sql() -> str:
    return render_sql(lambda config: command.upgrade(config, "head", sql=True))


def test_upgrade_creates_extension_and_tables(upgrade_sql: str) -> None:
    assert "CREATE EXTENSION IF NOT EXISTS vector" in upgrade_sql
    assert "CREATE TABLE documents" in upgrade_sql
    assert "CREATE TABLE chunks" in upgrade_sql
    assert "vector(1536)" in upgrade_sql.lower()
    assert "ON DELETE CASCADE" in upgrade_sql


def test_upgrade_creates_the_scalar_and_gin_indexes(upgrade_sql: str) -> None:
    for name in (
        "ix_documents_source_path",
        "ix_chunks_document_id",
        "ix_chunks_chunk_type",
        "ix_chunks_metadata_gin",
    ):
        assert f"CREATE INDEX {name}" in upgrade_sql
    assert "USING gin" in upgrade_sql


def test_upgrade_has_no_vector_index_and_no_unique_source_path(upgrade_sql: str) -> None:
    lowered = upgrade_sql.lower()

    assert "hnsw" not in lowered
    assert "ivfflat" not in lowered
    assert "UNIQUE INDEX ix_documents_source_path" not in upgrade_sql


def test_downgrade_drops_the_tables() -> None:
    sql = render_sql(lambda config: command.downgrade(config, "0001:base", sql=True))

    assert "DROP TABLE chunks" in sql
    assert "DROP TABLE documents" in sql
    assert "DROP EXTENSION" not in sql


def test_alembic_ini_has_no_credentials() -> None:
    assert "estimator:estimator@" not in ALEMBIC_INI.read_text()
