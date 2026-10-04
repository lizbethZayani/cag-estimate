"""Structlog configuration driven by ``Settings.log_level``."""

import logging

import structlog

DEFAULT_LEVEL = logging.INFO


def resolve_level(name: str) -> int:
    """Map a level name (case-insensitive) to a stdlib level, defaulting to INFO."""
    level = logging.getLevelName(name.upper())
    return level if isinstance(level, int) else DEFAULT_LEVEL


def configure_logging(level_name: str) -> None:
    """Configure structlog once with a level filter; repeated calls are harmless."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(resolve_level(level_name)),
    )
