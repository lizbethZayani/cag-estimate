"""Heuristic project-metadata extraction from a structured estimate.

Why a heuristic and not an extra LLM call: the estimate we already received
contains the project name, team size and a meeting summary, so deriving the
metadata from it is free, deterministic, offline and adds no failure mode (or
latency/cost) to every turn. Limitation: technologies are found by scanning a
curated vocabulary, so only known technologies are detected.
"""

import re

from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.schemas.session import ProjectMetadata

MAX_SCOPE_CHARS = 300

# Canonical name -> extra spellings to look for besides the name itself.
# Ambiguous tokens ("Go", "R") are deliberately left out.
_TECHNOLOGY_SPELLINGS: dict[str, tuple[str, ...]] = {
    "Python": (),
    "FastAPI": (),
    "Django": (),
    "Flask": (),
    "React": (),
    "Next.js": ("NextJS",),
    "Vue": ("Vue.js",),
    "Angular": (),
    "TypeScript": (),
    "JavaScript": (),
    "Node.js": ("NodeJS",),
    "Rails": ("Ruby on Rails",),
    "Ruby": (),
    "Java": (),
    "Kotlin": (),
    "PHP": (),
    "Laravel": (),
    "C++": (),
    "C#": (),
    ".NET": (),
    "PostgreSQL": ("Postgres",),
    "MySQL": (),
    "MongoDB": (),
    "Redis": (),
    "Docker": (),
    "Kubernetes": (),
    "AWS": (),
    "Azure": (),
    "GCP": (),
    "Stripe": (),
    "Terraform": (),
    "GraphQL": (),
    "REST": ("REST API", "RESTful"),
    "OAuth": (),
    "Tailwind": ("Tailwind CSS",),
}


def _compile(spellings: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(spelling) for spelling in spellings)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


_SPELLING_ONLY = frozenset({"REST"})  # the bare name is a common English word

_TECHNOLOGY_PATTERNS: dict[str, re.Pattern[str]] = {
    name: _compile(aliases if name in _SPELLING_ONLY else (name, *aliases))
    for name, aliases in _TECHNOLOGY_SPELLINGS.items()
}


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _truncate_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut.rsplit(" ", 1)[0] if text[limit] != " " else cut


def _scope(meeting_summary: str) -> str | None:
    return _truncate_at_word(_collapse(meeting_summary), MAX_SCOPE_CHARS) or None


def _team_size(team_size: str) -> int | None:
    match = re.search(r"\d+", team_size)
    return int(match.group()) if match else None


def _estimate_text(result: ProjectEstimation) -> list[str]:
    parts: list[str] = []
    for task in result.tasks:
        parts.extend([task.name, task.description, *task.includes])
    return parts


def _technologies(text: str) -> list[str]:
    return [name for name, pattern in _TECHNOLOGY_PATTERNS.items() if pattern.search(text)]


def extract_metadata(result: ProjectEstimation, transcript: str) -> ProjectMetadata:
    """Derive session metadata from an estimate and the transcript it came from."""
    text = "\n".join([transcript, *_estimate_text(result)])
    return ProjectMetadata(
        project_name=result.project_name or None,
        assumed_team_size=_team_size(result.summary.team_size),
        mentioned_technologies=_technologies(text),
        agreed_scope=_scope(result.meeting_summary),
    )
