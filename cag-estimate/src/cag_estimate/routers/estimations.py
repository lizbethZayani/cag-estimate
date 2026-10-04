"""HTTP layer for project estimation endpoints."""

import json
from collections.abc import Iterator
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from cag_estimate.dependencies import get_estimation_service
from cag_estimate.schemas.estimation import EstimationRequest, EstimationResponse
from cag_estimate.services.estimation import EstimationService

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

Service = Annotated[EstimationService, Depends(get_estimation_service)]

UPSTREAM_ERROR = "The estimation service is temporarily unavailable. Please try again."
EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
EMPTY_COST = {"input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _done_event(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "done",
        "model": result.get("model"),
        "provider": result.get("provider"),
        "tokens_used": result.get("usage", EMPTY_USAGE),
        "cost_breakdown": result.get("cost_breakdown", EMPTY_COST),
        "finish_reason": result.get("finish_reason", "stop"),
    }


@router.post("/estimate", response_model=EstimationResponse)
def estimate_project(request: EstimationRequest, service: Service) -> EstimationResponse:
    """Return a validated, structured project estimation."""
    try:
        return service.estimate(request)
    except Exception as exc:
        log.error("estimate_failed", error_type=type(exc).__name__, error=str(exc))
        raise HTTPException(status_code=502, detail=UPSTREAM_ERROR) from exc


@router.post("/estimate/stream")
def estimate_project_stream(request: EstimationRequest, service: Service) -> StreamingResponse:
    """Stream the estimation as Server-Sent Events (token / done / error)."""

    def event_stream() -> Iterator[str]:
        result: dict[str, Any] = {}
        try:
            for text in service.stream(request, result):
                yield _sse({"type": "token", "content": text})
            yield _sse(_done_event(result))
        except Exception as exc:  # noqa: BLE001 - SSE boundary: never leak, always emit error event
            log.error("estimate_stream_failed", error_type=type(exc).__name__, error=str(exc))
            yield _sse({"type": "error", "message": UPSTREAM_ERROR})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
