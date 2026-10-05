"""Tests for structlog configuration driven by ``Settings.log_level``."""

import logging

import pytest
import structlog

from cag_estimate.logging_config import configure_logging


@pytest.fixture(autouse=True)
def _reset_structlog():
    yield
    structlog.reset_defaults()


def _effective_level() -> int:
    config = structlog.get_config()
    wrapper_class = config["wrapper_class"]
    return next(
        level
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
        if wrapper_class(None, [], {}).is_enabled_for(level)
    )


def test_level_is_applied():
    configure_logging("WARNING")

    assert _effective_level() == logging.WARNING


def test_level_is_case_insensitive():
    configure_logging("info")

    assert _effective_level() == logging.INFO


def test_unknown_level_falls_back_to_info():
    configure_logging("nonsense")

    assert _effective_level() == logging.INFO


def test_configuring_twice_is_idempotent():
    configure_logging("ERROR")
    first = structlog.get_config()
    configure_logging("ERROR")

    assert structlog.get_config() == first


def test_app_startup_applies_settings_log_level(monkeypatch):
    from fastapi.testclient import TestClient

    from cag_estimate import config
    from cag_estimate.main import app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    config.get_settings.cache_clear()
    try:
        with TestClient(app):
            assert _effective_level() == logging.ERROR
    finally:
        config.get_settings.cache_clear()
