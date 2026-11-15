"""HTTP envelope of ``POST /api/v1/estimate/rag`` (separate from the LLM-facing schema)."""

from typing import Annotated

from pydantic import BaseModel, StringConstraints

from cag_estimate.schemas.rag_estimation import RagEstimate
from cag_estimate.services.citations import CitationReport


class RagEstimateRequest(BaseModel):
    """The project description to estimate."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class RagEstimateResponse(BaseModel):
    """A cited estimate with its verification report and the chunks it was grounded on."""

    estimate: RagEstimate
    citation_report: CitationReport
    prompt_version: str
    retrieved_chunk_ids: list[str]
