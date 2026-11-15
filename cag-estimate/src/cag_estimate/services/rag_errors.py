"""Errors of the cited RAG estimate flow; none of them carries raw provider text."""

from cag_estimate.services.citations import CitationReport


class RagGenerationError(Exception):
    """The generator produced no usable answer (provider failure, refusal or truncation)."""


class RagOutputInvalidError(Exception):
    """The answer broke a validation rule; the message names the failed rule only."""


class DanglingCitationError(Exception):
    """The answer cites sources that were never retrieved (a quality failure)."""

    def __init__(self, report: CitationReport) -> None:
        super().__init__("the estimate cites sources that were not retrieved")
        self.report = report
