"""HTTP layer for project estimation endpoints."""

import json
from collections.abc import Iterator
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from cag_estimate.dependencies import get_estimation_service
from cag_estimate.guardrails.input import InputGuardrailViolation
from cag_estimate.schemas.estimation import EstimationRequest, EstimationResponse
from cag_estimate.services.estimation import EstimationService

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

Service = Annotated[EstimationService, Depends(get_estimation_service)]

UPSTREAM_ERROR = "The estimation service is temporarily unavailable. Please try again."
EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
EMPTY_COST = {"input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0}


REJECTION_MESSAGES = {
    "moderation": "The description was flagged by content moderation. Please rephrase it.",
    "prompt_injection": "The description contains instruction-like text. Please describe the project only.",
    "pii": "The description appears to contain personal data. Please remove it and try again.",
}


def rejection_body(reason: str) -> dict[str, str]:
    return {"reason": reason, "message": REJECTION_MESSAGES[reason]}


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
def estimate_project(
    request: EstimationRequest, service: Service
) -> EstimationResponse | JSONResponse:
    """Return a validated, structured project estimation.

    Rejected input -> 400 ``{"reason", "message"}``; output guardrail
    violations and any other failure -> generic 502.
    """
    try:
        return service.estimate(request)
    except InputGuardrailViolation as exc:
        log.info("estimate_rejected", reason=exc.reason)
        return JSONResponse(status_code=400, content=rejection_body(exc.reason))
    except Exception as exc:
        log.error("estimate_failed", error_type=type(exc).__name__, error=str(exc))
        raise HTTPException(status_code=502, detail=UPSTREAM_ERROR) from exc


@router.post("/estimate/stream")
def estimate_project_stream(request: EstimationRequest, service: Service) -> StreamingResponse:
    """Stream the estimation as Server-Sent Events (token / done / error).

    A rejected input is reported as a single ``error`` event (HTTP 200)
    carrying the safe message plus ``reason``, not as an HTTP 400: the
    Streamlit client calls ``raise_for_status()`` and would otherwise show a
    raw "400 Client Error" instead of the guardrail message.
    """

    def event_stream() -> Iterator[str]:
        result: dict[str, Any] = {}
        try:
            for text in service.stream(request, result):
                yield _sse({"type": "token", "content": text})
            yield _sse(_done_event(result))
        except InputGuardrailViolation as exc:
            log.info("estimate_stream_rejected", reason=exc.reason)
            yield _sse({"type": "error", **rejection_body(exc.reason)})
        except Exception as exc:  # noqa: BLE001 - SSE boundary: never leak, always emit error event
            log.error("estimate_stream_failed", error_type=type(exc).__name__, error=str(exc))
            yield _sse({"type": "error", "message": UPSTREAM_ERROR})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
