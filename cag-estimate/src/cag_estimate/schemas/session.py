"""Session-scoped schemas shared by the prompt and session layers."""

from pydantic import BaseModel, Field, field_validator


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
