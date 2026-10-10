"""Audience tier resolution: who is the estimate for, executives, PMs or developers?

This is a HEURISTIC, not a classifier: it looks for vocabulary cues in the text
of the turn and for how many technologies the project already mentions. It is
pure and offline, and it can be wrong, so callers can always force a tier.

Precedence is defined in ONE place, ``_RULES``: the first rule that matches
decides, in this order:

1. ``override``         an explicit tier chosen by the caller always wins;
2. ``executive_cues``   business vocabulary (CEO, board, ROI, ...);
3. ``pm_cues``          delivery vocabulary (sprint, roadmap, milestone, ...);
4. ``developer_cues``   technical vocabulary (API, schema, framework names, ...);
5. ``technology_count`` the project already mentions 3 or more technologies;
6. ``default``          nothing matched (reported as the rule name, not a rule).

Executive beats PM beats developer because a mixed message ("the CEO wants the
API ready by the next sprint") is addressed to its most senior reader.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from cag_estimate.schemas.session import ProjectMetadata

MIN_TECHNOLOGIES_FOR_DEVELOPER = 3
DEFAULT_RULE = "default"


class Tier(StrEnum):
    """Audience the estimate is written for (values are the wire names)."""

    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass(frozen=True)
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata
    override: Tier | None


@dataclass(frozen=True)
class TierRule:
    """One link of the chain: ``decide`` returns a tier, or None to pass to the next rule."""

    name: str
    decide: Callable[[ResolutionContext], Tier | None]


EXECUTIVE_CUES = (
    "ceo",
    "cfo",
    "cto",
    "coo",
    "board",
    "investor",
    "investors",
    "budget approval",
    "roi",
    "return on investment",
    "executive summary",
    "c-level",
    "shareholders",
)
PM_CUES = (
    "sprint",
    "sprints",
    "backlog",
    "roadmap",
    "milestone",
    "milestones",
    "stakeholder",
    "stakeholders",
    "timeline",
    "deliverable",
    "deliverables",
    "scrum",
    "kanban",
)
DEVELOPER_CUES = (
    "api",
    "apis",
    "endpoint",
    "endpoints",
    "schema",
    "microservice",
    "microservices",
    "refactor",
    "refactoring",
    "framework",
    "sdk",
    "database migration",
    "ci/cd",
    "react",
    "django",
    "fastapi",
    "flask",
    "node.js",
    "typescript",
    "kubernetes",
    "docker",
    "terraform",
    "graphql",
    "postgresql",
    "redis",
    "kafka",
)


def _cue_pattern(cues: tuple[str, ...]) -> re.Pattern[str]:
    """Case-insensitive, word-bounded alternation of the cues (longest first)."""
    alternatives = "|".join(re.escape(cue) for cue in sorted(cues, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _cue_rule(name: str, tier: Tier, cues: tuple[str, ...]) -> TierRule:
    pattern = _cue_pattern(cues)
    return TierRule(name, lambda context: tier if pattern.search(context.transcript) else None)


def _override(context: ResolutionContext) -> Tier | None:
    return context.override


def _technology_count(context: ResolutionContext) -> Tier | None:
    many = len(context.metadata.mentioned_technologies) >= MIN_TECHNOLOGIES_FOR_DEVELOPER
    return Tier.DEVELOPER if many else None


_RULES: tuple[TierRule, ...] = (
    TierRule("override", _override),
    _cue_rule("executive_cues", Tier.EXECUTIVE, EXECUTIVE_CUES),
    _cue_rule("pm_cues", Tier.PM, PM_CUES),
    _cue_rule("developer_cues", Tier.DEVELOPER, DEVELOPER_CUES),
    TierRule("technology_count", _technology_count),
)


def resolve_tier(
    transcript: str, metadata: ProjectMetadata, override: Tier | None
) -> tuple[Tier, str]:
    """Return ``(tier, rule_name)``: the tier for this turn and the rule that decided it.

    ``metadata`` is what the session knew BEFORE this turn. Falls back to
    ``(Tier.DEFAULT, "default")`` when no rule matches.
    """
    context = ResolutionContext(transcript, metadata, override)
    for rule in _RULES:
        tier = rule.decide(context)
        if tier is not None:
            return tier, rule.name
    return Tier.DEFAULT, DEFAULT_RULE
