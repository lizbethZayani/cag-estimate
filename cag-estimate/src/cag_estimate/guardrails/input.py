"""
Input guardrails, run in order before the cache and the LLM are touched.

1. Moderation (OpenAI moderations endpoint): only when a client is provided,
   and it fails open (logs a warning) on network/auth errors.
2. Prompt injection: case-insensitive regexes over well-known attack phrases.
3. PII: email, IBAN and phone heuristics, deliberately conservative so plain
   years, dates and hour counts are not flagged.

Every layer raises ``InputGuardrailViolation``; the ``reason`` lets the HTTP
layer pick a safe message without echoing the offending text.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import structlog

from cag_estimate.guardrails.errors import GuardrailError

log = structlog.get_logger()

Reason = Literal["moderation", "prompt_injection", "pii"]

_FLAGS = re.IGNORECASE | re.DOTALL

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(previous|prior|all|the)\s+(instructions?|prompts?|rules?)", _FLAGS),
    re.compile(r"</?\s*(system|instructions?|prompt)\s*>", _FLAGS),
    re.compile(r"new\s+instructions?\s*[:.\-]", _FLAGS),
    re.compile(r"forget\s+(everything|all|previous)", _FLAGS),
    re.compile(r"\byou\s+are\s+now\b", _FLAGS),
    re.compile(r"\bdisregard\b.{0,40}\b(instructions?|prompts?|rules?|context|previous|prior)", _FLAGS),
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# IBAN: 2 letters, 2 check digits, then 10-30 alphanumerics (uppercase only).
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
# Phone: optional +CC prefix then 9-13 digits with optional separators. Needs
# at least 9 digits, so years, ISO dates (8 digits) and hour counts pass.
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){8,12}\d")
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (_EMAIL_RE, _IBAN_RE, _PHONE_RE)


class InputGuardrailViolation(GuardrailError):
    """Raised by ``check_input`` when a layer rejects the description."""

    def __init__(self, message: str, *, reason: Reason) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason


def check_input(text: str, *, openai_client: Any | None = None) -> None:
    """Run moderation, prompt-injection and PII checks; raise on the first hit."""
    if openai_client is not None:
        _check_moderation(text, openai_client)
    _check_prompt_injection(text)
    _check_pii(text)


def _check_moderation(text: str, openai_client: Any) -> None:
    try:
        response = openai_client.moderations.create(input=text)
    except Exception as exc:  # noqa: BLE001 - network/auth failures fail open
        log.warning("moderation_call_failed", error_type=type(exc).__name__)
        return
    if getattr(response.results[0], "flagged", False):
        log.info("moderation_flagged")
        raise InputGuardrailViolation("Input flagged by moderation.", reason="moderation")


def _check_prompt_injection(text: str) -> None:
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            log.info("prompt_injection_detected", pattern=pattern.pattern)
            raise InputGuardrailViolation(
                "Instruction-like text detected.", reason="prompt_injection"
            )


def _check_pii(text: str) -> None:
    if any(pattern.search(text) for pattern in _PII_PATTERNS):
        log.info("pii_detected")
        raise InputGuardrailViolation("Personal data detected.", reason="pii")
