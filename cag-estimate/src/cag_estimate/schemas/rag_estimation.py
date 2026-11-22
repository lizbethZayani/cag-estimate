"""
Cited RAG estimate schemas with an integrity rule.

These models are sent to the OpenAI Responses API as a strict ``text_format``,
so LLM-facing fields carry no defaults and the semantic rules live in
validators that run after parsing. Validation messages are actionable: a
corrective retry feeds them back to the model.

The model never supplies totals: ``RagEstimate.total_hours`` is computed from
the grounded lines and is not part of the validation schema.
"""

from typing import Literal

from pydantic import BaseModel, computed_field, field_validator, model_validator

Confidence = Literal["high", "medium", "low", "insufficient"]


def _non_blank(value: str, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must not be blank.")
    return stripped


class SourceReference(BaseModel):
    """A pointer to the retrieved chunk that supports a line, with verbatim evidence."""

    chunk_id: str  # "<budget_id>::<component_id>", e.g. "BUD-2024-005::CART-002"
    document_id: str  # the budget id, e.g. "BUD-2024-005"
    evidence: str

    @field_validator("chunk_id")
    @classmethod
    def check_chunk_id(cls, value: str) -> str:
        return _non_blank(value, "chunk_id")

    @field_validator("document_id")
    @classmethod
    def check_document_id(cls, value: str) -> str:
        return _non_blank(value, "document_id")

    @field_validator("evidence")
    @classmethod
    def check_evidence(cls, value: str) -> str:
        return _non_blank(value, "evidence")


class EstimateLineItem(BaseModel):
    """One estimated component; ``hours`` exists only when the line is grounded."""

    component: str
    hours: float | None
    rationale: str
    grounded: bool
    sources: list[SourceReference]

    @model_validator(mode="after")
    def check_integrity(self) -> "EstimateLineItem":
        if self.grounded:
            self._check_grounded()
        else:
            self._check_ungrounded()
        return self

    def _check_grounded(self) -> None:
        if not self.sources:
            raise ValueError(
                f'Line "{self.component}": a grounded line must cite at least one source. '
                "Add a source or mark the line as not grounded."
            )
        if self.hours is None:
            raise ValueError(
                f'Line "{self.component}": a grounded line must have hours. '
                "Provide the hours supported by the cited sources."
            )
        if self.hours < 0:
            raise ValueError(f'Line "{self.component}": hours must be >= 0, got {self.hours}.')

    def _check_ungrounded(self) -> None:
        if self.sources:
            raise ValueError(
                f'Line "{self.component}": an ungrounded line must not cite sources. '
                "Remove the sources or mark the line as grounded."
            )
        if self.hours is not None:
            raise ValueError(
                f'Line "{self.component}": an ungrounded line must not invent hours. '
                "Set hours to null."
            )
        if not self.rationale.strip():
            raise ValueError(
                f'Line "{self.component}": an ungrounded line must explain what data is '
                "missing in its rationale."
            )


class RagEstimate(BaseModel):
    """A cited estimate: grounded lines carry hours and sources, the rest say why not."""

    lines: list[EstimateLineItem]
    assumptions: list[str]
    confidence: Confidence
    insufficient_context_explanation: str | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_hours(self) -> float | None:
        """Sum of the grounded lines' hours; None when no line is grounded."""
        grounded = [line for line in self.lines if line.grounded]
        if not grounded:
            return None
        return sum(line.hours or 0.0 for line in grounded)

    @model_validator(mode="after")
    def check_confidence_coupling(self) -> "RagEstimate":
        if any(line.grounded for line in self.lines):
            if self.confidence == "insufficient":
                raise ValueError(
                    'confidence must not be "insufficient" when at least one line is grounded.'
                )
            return self
        if self.confidence != "insufficient":
            raise ValueError('confidence must be "insufficient" when no line is grounded.')
        if not (self.insufficient_context_explanation or "").strip():
            raise ValueError(
                "insufficient_context_explanation must explain what context is missing "
                "when no line is grounded."
            )
        return self
