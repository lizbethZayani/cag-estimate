"""Offline unit tests for anchor detection, summarizing and history compaction."""

import pytest
from structlog.testing import capture_logs

from cag_estimate.services.session_compression import (
    AnchorDetector,
    ExtractiveSummarizer,
    HistoryCompactor,
)
from cag_estimate.services.sessions import Anchor, ConversationHistory, ProjectMetadata

ASSISTANT_JSON = '{"project_name": "SECRET_ASSISTANT_PAYLOAD"}'


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Everything here is under an NDA, do not share it.", "nda"),
        ("We have a non-disclosure agreement with them.", "nda"),
        ("We signed the contract last week.", "signed_contract"),
        ("The client countersigned the SOW yesterday.", "signed_contract"),
        ("The scope is frozen after kickoff.", "scope_frozen"),
        ("Backlog locked, no more features.", "scope_frozen"),
        ("The budget is fixed at 50k USD.", "budget_locked"),
        ("Our ceiling is capped at 80 hours total.", "budget_locked"),
        ("Patient data means we must satisfy HIPAA.", "compliance"),
        ("EU users so GDPR applies; also ISO 27001.", "compliance"),
        ("There is a hard deadline on March 1st.", "deadline_hard"),
        ("It must go live by June.", "deadline_hard"),
        ("We are contractually bound to use Oracle.", "contractual"),
        ("We agreed to deliver a pilot first.", "explicit_commitment"),
        ("The customer committed to weekly demos.", "explicit_commitment"),
    ],
)
def test_detector_flags_hard_constraints(text: str, kind: str):
    assert AnchorDetector().detect(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        "We use Go and R for the analytics module.",
        "Build a bookstore web shop with Stripe payments.",
        "The deadline is flexible, around spring.",
        "",
        "We need a signed-in user area.",
    ],
)
def test_detector_ignores_normal_text(text: str):
    assert AnchorDetector().detect(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "No NDA needed for this one.",
        "We don't have an NDA yet.",
        "This is not subject to HIPAA.",
        "They haven't signed the contract.",
        "Without GDPR concerns, keep it simple.",
    ],
)
def test_detector_skips_simple_negations(text: str):
    assert AnchorDetector().detect(text) is None


def test_detector_anchors_when_a_later_mention_is_not_negated():
    assert AnchorDetector().detect("No NDA for them. But our own NDA applies to you.") == "nda"


def test_extractive_summarizer_adds_one_bullet_per_pair_without_assistant_json():
    summarizer = ExtractiveSummarizer(max_chars=2000)

    summary = summarizer.summarize(
        None, [("First  request\nwith   gaps", ASSISTANT_JSON), ("Second", "x")]
    )

    assert summary.splitlines() == ["- First request with gaps", "- Second"]
    assert "SECRET_ASSISTANT_PAYLOAD" not in summary


def test_extractive_summarizer_is_cumulative_and_truncates_long_user_text():
    summarizer = ExtractiveSummarizer(max_chars=2000)

    summary = summarizer.summarize("- old bullet", [("w" * 500, "a")])

    lines = summary.splitlines()
    assert lines[0] == "- old bullet"
    assert len(lines[1]) <= 2 + 200 + 3


def test_extractive_summarizer_drops_oldest_bullets_to_respect_the_cap():
    summarizer = ExtractiveSummarizer(max_chars=40)
    previous = "- aaaaaaaaaaaaaaaaaaaa\n- bbbbbbbbbbbbbbbbbbbb"

    summary = summarizer.summarize(previous, [("cccccccccccccccccccc", "a")])

    assert len(summary) <= 40
    assert "aaaa" not in summary
    assert summary.splitlines()[-1] == "- cccccccccccccccccccc"


def test_extractive_summarizer_cuts_a_single_bullet_longer_than_the_cap():
    summary = ExtractiveSummarizer(max_chars=10).summarize(None, [("z" * 100, "a")])

    assert len(summary) == 10


class RecordingSummarizer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str | None, list[tuple[str, str]]]] = []

    def summarize(self, previous_summary: str | None, evicted: list[tuple[str, str]]) -> str:
        self.calls.append((previous_summary, list(evicted)))
        if self.error:
            raise self.error
        return "SUMMARY:" + "|".join(user for user, _ in evicted)


def make_history() -> ConversationHistory:
    return ConversationHistory(ProjectMetadata(), lambda _m: "SYS", max_turns=6)


def make_compactor(summarizer, max_anchors: int = 5, summary_max_chars: int = 2000):
    return HistoryCompactor(AnchorDetector(), summarizer, max_anchors, summary_max_chars)


def test_compactor_stores_anchor_user_text_only_and_summarizes_the_rest():
    history = make_history()
    summarizer = RecordingSummarizer()

    make_compactor(summarizer).compact(
        history, [("Everything is under NDA", ASSISTANT_JSON), ("plain request", ASSISTANT_JSON)]
    )

    assert history.anchors == [Anchor("nda", "Everything is under NDA")]
    assert history.summary == "SUMMARY:plain request"
    assert summarizer.calls == [(None, [("plain request", ASSISTANT_JSON)])]


def test_compactor_does_not_store_the_same_anchor_twice_across_compactions():
    history = make_history()
    compactor = make_compactor(RecordingSummarizer())

    compactor.compact(history, [("Under NDA", "a")])
    compactor.compact(history, [("Under NDA", "a")])

    assert history.anchors == [Anchor("nda", "Under NDA")]


def test_compactor_passes_the_previous_summary_to_the_next_compaction():
    history = make_history()
    summarizer = RecordingSummarizer()
    compactor = make_compactor(summarizer)

    compactor.compact(history, [("one", "a")])
    compactor.compact(history, [("two", "a")])

    assert summarizer.calls[1][0] == "SUMMARY:one"
    assert history.summary == "SUMMARY:two"


def test_compactor_skips_the_summarizer_when_everything_evicted_is_an_anchor():
    history = make_history()
    summarizer = RecordingSummarizer()

    make_compactor(summarizer).compact(history, [("Under NDA", "a")])

    assert summarizer.calls == []
    assert history.summary is None


def test_compactor_folds_the_oldest_anchor_into_the_summary_when_over_the_cap():
    history = make_history()
    summarizer = RecordingSummarizer()
    compactor = make_compactor(summarizer, max_anchors=1)

    compactor.compact(history, [("Under NDA", "a"), ("HIPAA applies", "b")])

    assert history.anchors == [Anchor("compliance", "HIPAA applies")]
    assert len(summarizer.calls) == 1
    assert [user for user, _ in summarizer.calls[0][1]] == ["Under NDA"]
    assert "Under NDA" in (history.summary or "")


def test_compactor_falls_back_to_the_extractive_summary_when_the_summarizer_fails():
    history = make_history()
    compactor = make_compactor(RecordingSummarizer(error=RuntimeError("boom")))

    with capture_logs() as logs:
        compactor.compact(history, [("plain request", ASSISTANT_JSON)])

    assert history.summary == "- plain request"
    assert any(entry["event"] == "session_summary_failed" for entry in logs)


def test_compactor_with_nothing_evicted_changes_nothing():
    history = make_history()
    summarizer = RecordingSummarizer()

    make_compactor(summarizer).compact(history, [])

    assert summarizer.calls == []
    assert history.anchors == []
    assert history.summary is None
