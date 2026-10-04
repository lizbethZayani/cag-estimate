"""Shared base class for guardrail rejections."""


class GuardrailError(Exception):
    """Base class for requests or results rejected by a guardrail."""
