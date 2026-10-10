"""History compression: pin hard-constraint turns, summarize the rest of what is evicted.

When the sliding window overflows, the evicted pairs are not simply lost. User
messages that state a hard constraint (NDA, frozen scope, locked budget,
compliance, hard deadline...) become permanent *anchors*; every other evicted
pair is folded into one cumulative summary. Summaries come from a ``Summarizer``
(an LLM-backed one can be plugged in later); the deterministic
``ExtractiveSummarizer`` is the fallback and the default.
"""

import re
from typing import Protocol

import structlog

from cag_estimate.services.sessions import Anchor, ConversationHistory

log = structlog.get_logger()

ANCHOR_TEXT_MAX_CHARS = 500
SUMMARY_BULLET_MAX_CHARS = 200
_ELLIPSIS = "..."

# (kind, pattern): data, not branching. Patterns are conservative on purpose: a miss
# only demotes the turn to the summary, a false positive stays in every prompt.
_ANCHOR_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(pattern, re.IGNORECASE))
    for kind, pattern in (
        ("nda", r"\b(?:nda|non[- ]?disclosure|under embargo|legal hold)\b"),
        (
            "signed_contract",
            r"\b(?:signed|countersigned)\s+(?:the\s+|a\s+|our\s+)?(?:contract|sow|msa|agreement)\b",
        ),
        (
            "scope_frozen",
            r"\b(?:scope|backlog)\s+(?:is\s+|has been\s+)?(?:frozen|locked|final|fixed)\b",
        ),
        (
            "budget_locked",
            r"\b(?:budget|cap|ceiling)\s+(?:is\s+)?(?:locked|fixed|approved|capped)\b",
        ),
        ("compliance", r"\b(?:hipaa|gdpr|sox|pci(?:[- ]?dss)?|fda|iso[- ]?27001)\b"),
        ("deadline_hard", r"\bhard\s+deadline\b|\bmust\s+go\s+live\s+by\b"),
        ("contractual", r"\bcontractually\s+(?:bound|required|obliged)\b"),
        (
            "explicit_commitment",
            r"\b(?:we|the\s+(?:client|customer))\s+(?:agreed|committed)\s+to\b",
        ),
    )
)

# A negation shortly before a match ("no NDA", "don't have an NDA") cancels that match.
# Limitation: only the same clause is looked at, and paraphrases are not understood.
_NEGATION_BEFORE = re.compile(
    r"(?:\b(?:no|not|without|never|cannot)\b|n't\b)[^.!?;\n]{0,25}$", re.IGNORECASE
)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def clip(text: str, limit: int) -> str:
    """Whitespace-collapsed ``text`` cut to ``limit`` characters, ending in an ellipsis."""
    collapsed = collapse_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - len(_ELLIPSIS)] + _ELLIPSIS


class AnchorDetector:
    """Heuristic detector of user messages that state a hard, durable constraint."""

    def detect(self, user_text: str) -> str | None:
        """Return the anchor kind of ``user_text``, or None for ordinary text."""
        for kind, pattern in _ANCHOR_RULES:
            if any(not self._negated(user_text, match) for match in pattern.finditer(user_text)):
                return kind
        return None

    @staticmethod
    def _negated(text: str, match: re.Match[str]) -> bool:
        return _NEGATION_BEFORE.search(text[: match.start()]) is not None


class Summarizer(Protocol):
    """Folds evicted (user, assistant) pairs into a cumulative summary."""

    def summarize(self, previous_summary: str | None, evicted: list[tuple[str, str]]) -> str: ...


class ExtractiveSummarizer:
    """Deterministic fallback: one bullet per evicted user message, oldest bullets dropped first."""

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars

    def summarize(self, previous_summary: str | None, evicted: list[tuple[str, str]]) -> str:
        bullets = previous_summary.splitlines() if previous_summary else []
        bullets += [f"- {clip(user, SUMMARY_BULLET_MAX_CHARS)}" for user, _assistant in evicted]
        while len(bullets) > 1 and len("\n".join(bullets)) > self._max_chars:
            bullets.pop(0)
        return "\n".join(bullets)[: self._max_chars]


class HistoryCompactor:
    """Moves evicted pairs into the history's anchors and cumulative summary."""

    def __init__(
        self,
        detector: AnchorDetector,
        summarizer: Summarizer,
        max_anchors: int,
        summary_max_chars: int,
    ) -> None:
        self._detector = detector
        self._summarizer = summarizer
        self._max_anchors = max_anchors
        self._fallback = ExtractiveSummarizer(summary_max_chars)

    def compact(self, history: ConversationHistory, evicted: list[tuple[str, str]]) -> None:
        """Pin anchors and summarize the rest; never raises (the turn already succeeded)."""
        if not evicted:
            return
        anchors, rest = self._split_anchors(history.anchors, evicted)
        pinned, overflow = self._cap_anchors(anchors)
        to_summarize = [(anchor.text, "") for anchor in overflow] + rest
        summary = history.summary
        if to_summarize:
            summary = self._summarize(summary, to_summarize)
        history.update_context(pinned, summary)

    def _split_anchors(
        self, known: list[Anchor], evicted: list[tuple[str, str]]
    ) -> tuple[list[Anchor], list[tuple[str, str]]]:
        anchors = list(known)
        rest: list[tuple[str, str]] = []
        for user, assistant in evicted:
            kind = self._detector.detect(user)
            if kind is None:
                rest.append((user, assistant))
                continue
            anchor = Anchor(kind, clip(user, ANCHOR_TEXT_MAX_CHARS))
            if anchor not in anchors:
                anchors.append(anchor)
        return anchors, rest

    def _cap_anchors(self, anchors: list[Anchor]) -> tuple[list[Anchor], list[Anchor]]:
        excess = max(len(anchors) - self._max_anchors, 0)
        return anchors[excess:], anchors[:excess]

    def _summarize(self, previous: str | None, pairs: list[tuple[str, str]]) -> str:
        try:
            return self._summarizer.summarize(previous, pairs)
        except Exception as exc:  # noqa: BLE001 - compaction must never fail a successful turn
            log.warning("session_summary_failed", error_type=type(exc).__name__)
            return self._fallback.summarize(previous, pairs)
