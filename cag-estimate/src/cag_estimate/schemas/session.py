"""Session-scoped schemas shared by the prompt and session layers."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from cag_estimate.schemas.estimation import EstimationResponse


def _merge_unique(current: list[str], extra: list[str]) -> list[str]:
    """Union preserving first-seen order and original casing (case-insensitive)."""
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*current, *extra]:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


class ProjectMetadata(BaseModel):
    """Facts about the project accumulated over the conversation."""

    project_name: str | None = None
    assumed_team_size: int | None = None
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None

    @field_validator("mentioned_technologies")
    @classmethod
    def _dedupe_technologies(cls, value: list[str]) -> list[str]:
        return _merge_unique([], value)

    def is_empty(self) -> bool:
        return not (
            self.project_name
            or self.assumed_team_size
            or self.mentioned_technologies
            or self.agreed_scope
        )

    def merge(self, update: "ProjectMetadata") -> None:
        """Overwrite with non-empty values from ``update``; union technologies."""
        if update.project_name:
            self.project_name = update.project_name
        if update.assumed_team_size:
            self.assumed_team_size = update.assumed_team_size
        if update.agreed_scope:
            self.agreed_scope = update.agreed_scope
        self.mentioned_technologies = _merge_unique(
            self.mentioned_technologies, update.mentioned_technologies
        )


class SessionCreated(BaseModel):
    """Response of ``POST /sessions``."""

    session_id: str


class TurnObservation(BaseModel):
    """Per-turn measurements for stress testing the conversation memory.

    ``tokens_*``, ``cost_usd`` and ``latency_ms`` sum every LLM call of the turn
    (the actor plus the summarizer when compaction ran). ``latency_ms`` is the
    server-side sum of those calls, not the wall-clock of the HTTP request.
    ``cache_hit_kind`` is always ``"none"`` here: session estimates bypass both
    caches. The context-size fields describe the session AFTER the turn committed.
    """

    turn_index: int = Field(ge=1)
    session_id: str
    enriched_transcript_chars: int
    attachments_total_chars: int
    messages_in_window: int
    anchors_count: int
    summary_chars: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    cache_hit_kind: Literal["none", "exact", "semantic"]
    last_resolved_tier: str | None


class SessionEstimationResponse(EstimationResponse):
    """A regular estimation plus the session context it was produced in.

    ``cached`` is always False: session estimates bypass the caches.
    """

    session_id: str
    project_metadata: ProjectMetadata
    turns: int
    attachments: list[str] = Field(default_factory=list)
    resolved_tier: str
    tier_rule: str
    observation: TurnObservation | None = None
