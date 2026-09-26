"""
Routers for project estimation endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from cag_estimate.context.examples import ESTIMATION_EXAMPLES
from cag_estimate.services.llm_wrapper import get_llm_wrapper

router = APIRouter(prefix="/api/v1", tags=["estimations"])


class EstimationRequest(BaseModel):
    """Request model for project estimation."""

    transcription: str = Field(..., description="Meeting transcription to estimate")
    hourly_rate: Optional[int] = Field(
        default=40, description="Hourly rate for cost calculation (default: $40)"
    )


class TaskEstimate(BaseModel):
    """Individual task estimation."""

    task_id: int
    name: str
    description: str
    estimated_hours: int
    estimated_cost_usd: int
    complexity: str
    includes: list[str]


class ProjectSummary(BaseModel):
    """Project summary with totals."""

    total_hours: int
    total_cost_usd: int
    team_size: str
    estimated_duration_weeks: int
    hourly_rate: int
    assumptions: Optional[list[str]] = None


class ProjectEstimation(BaseModel):
    """Project estimation result."""

    project_name: str
    meeting_summary: str
    tasks: list[TaskEstimate]
    summary: ProjectSummary


class EstimationResponse(BaseModel):
    """Response model for estimation endpoint."""

    estimation: ProjectEstimation
    model: str = Field(description="LLM model used")
    provider: str = Field(description="LLM provider used")
    tokens_used: dict = Field(
        description="Token usage breakdown: input, output, total"
    )
    cost_breakdown: dict = Field(
        description="Cost breakdown: input_cost, output_cost, total_cost"
    )


def build_system_prompt(output_format: str = "json") -> str:
    """
    Build the system prompt with role definition and context examples.
    Core of the CAG architecture where context drives the calls.

    Args:
        output_format: "json" for the structured /estimate endpoint, or
            "markdown" for the /estimate/stream endpoint, where the raw
            generated tokens are displayed to the user in real time, so
            they must already be human-readable instead of raw JSON.
    """
    examples_context = json.dumps(ESTIMATION_EXAMPLES, indent=2)

    system_prompt = f"""You are an expert software estimation specialist with deep knowledge in project planning and technical assessment.

Your role is to:
1. Analyze meeting transcriptions that describe software projects
2. Extract key requirements, features, and technical considerations
3. Generate detailed project estimations based on previous estimation examples
4. Provide accurate hour and cost estimates for individual tasks
5. Calculate team size, duration, and overall project metrics

REFERENCE EXAMPLES FOR ESTIMATION:

{examples_context}

ESTIMATION GUIDELINES:
- Break down projects into logical, manageable tasks
- Estimate hours based on task complexity (Simple: 4-8h, Medium: 12-24h, High: 24-48h+)
- Calculate costs using: hours × hourly_rate (typically $40-$60/hour depending on seniority)
- Consider team composition and communication overhead
- Include testing, documentation, and deployment tasks
- Add 15-20% buffer for integration and unforeseen issues
- Provide detailed task descriptions with included work items
"""

    if output_format == "markdown":
        system_prompt += """
OUTPUT FORMAT:
Respond directly in this Markdown format. Do NOT use JSON and do NOT wrap the
response in code fences — this text is streamed straight to the user as-is:

**Project:** <project name>

**Summary:**
- **Total Hours:** <number>
- **Total Cost:** $<number>
- **Team Size:** <e.g. "2 developers">
- **Duration:** <number> weeks
- **Hourly Rate:** $<number>/hour

**Meeting Summary:**
<concise summary of the meeting>

**Tasks Breakdown:**
1. **<task name>** (<Simple|Medium|High>)
   - Hours: <number> | Cost: $<number>
2. **<task name>** (<Simple|Medium|High>)
   - Hours: <number> | Cost: $<number>

**Key Assumptions:**
- <assumption 1>
- <assumption 2>

Be thorough but realistic in your estimations. Use the provided examples as benchmarks."""
    else:
        system_prompt += """
OUTPUT FORMAT:
Return a JSON object with this structure:
{
    "project_name": "string",
    "meeting_summary": "string (concise summary of meeting)",
    "tasks": [
        {
            "task_id": number,
            "name": "string",
            "description": "string",
            "estimated_hours": number,
            "estimated_cost_usd": number,
            "complexity": "Simple|Medium|High",
            "includes": ["list", "of", "deliverables"]
        }
    ],
    "summary": {
        "total_hours": number,
        "total_cost_usd": number,
        "team_size": "string (e.g., '2 developers')",
        "estimated_duration_weeks": number,
        "hourly_rate": number,
        "assumptions": ["list", "of", "key", "assumptions"]
    }
}

Be thorough but realistic in your estimations. Use the provided examples as benchmarks."""

    return system_prompt


@router.post("/estimate", response_model=EstimationResponse)
async def estimate_project(request: EstimationRequest) -> EstimationResponse:
    """
    Generate a project estimation based on meeting transcription.

    This endpoint uses the CAG (Context-Augmented Generation) architecture where:
    - System prompt: Instructions + reference examples from context
    - User message: Meeting transcription to estimate
    - Response: Detailed project estimation with task breakdown

    The actual LLM call goes through LLMWrapper (LiteLLM), which transparently
    falls back to a secondary provider if the primary one fails, and serves
    repeated requests from an exact-match Redis cache. This endpoint never
    knows — or needs to know — which provider actually answered.

    Args:
        request: EstimationRequest containing meeting transcription and optional hourly_rate

    Returns:
        EstimationResponse with estimation, model info, token usage, and costs

    Raises:
        HTTPException: If estimation fails or response parsing fails
    """
    wrapper = get_llm_wrapper()

    try:
        # Build system prompt with instructions and context examples
        system_prompt = build_system_prompt()

        # Prepare user message with meeting transcription
        user_message = f"""Based on the following meeting transcription, provide a detailed project estimation with task breakdown, hours, and costs.

MEETING TRANSCRIPTION:
{request.transcription}

INSTRUCTIONS:
- Use the hourly rate of ${request.hourly_rate}/hour for cost calculations
- Be specific and realistic with estimates
- Ensure all tasks are clearly defined with measurable deliverables
- Include tasks for testing, documentation, and deployment
- Provide assumptions made during estimation

Return only valid JSON, no additional text."""

        # Call the LLM through the wrapper (cache + provider fallback)
        result = wrapper.complete(
            system_prompt=system_prompt, user_message=user_message, max_tokens=4096
        )
        response_text = result["estimation"]

        # Parse JSON from response
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                estimation_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse LLM response as JSON: {str(e)}",
            )

        # Validate and convert to ProjectEstimation model
        estimation = ProjectEstimation(**estimation_data)

        # Build response — model/provider reflect whichever deployment actually
        # answered (primary or fallback), not just the configured default
        return EstimationResponse(
            estimation=estimation,
            model=result["model"],
            provider=result["provider"],
            tokens_used=result["usage"],
            cost_breakdown=result["cost_breakdown"],
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}"
        )


@router.post("/estimate/stream")
async def estimate_project_stream(request: EstimationRequest):
    """
    Stream a project estimation token-by-token using Server-Sent Events (SSE).

    Unlike /estimate, the model is prompted to respond directly in Markdown
    (see build_system_prompt(output_format="markdown")) instead of JSON, so
    each raw token is already human-readable and can be shown to the user
    the instant it's generated, with no post-processing needed.

    Like /estimate, the actual call goes through LLMWrapper.complete_stream()
    (cache + provider fallback) — this endpoint just relays chunks as SSE.

    Event types sent to the client, one JSON object per "data:" line:
    - {"type": "token", "content": "..."}   one text chunk, as generated by the model
    - {"type": "done", "model": ..., "provider": ...,
       "tokens_used": {...}, "cost_breakdown": {...}, "finish_reason": ...}
    - {"type": "error", "message": "..."}
    """
    wrapper = get_llm_wrapper()
    system_prompt = build_system_prompt(output_format="markdown")

    user_message = f"""Based on the following meeting transcription, provide a detailed project estimation with task breakdown, hours, and costs.

MEETING TRANSCRIPTION:
{request.transcription}

INSTRUCTIONS:
- Use the hourly rate of ${request.hourly_rate}/hour for cost calculations
- Be specific and realistic with estimates
- Ensure all tasks are clearly defined with measurable deliverables
- Include tasks for testing, documentation, and deployment
- Provide assumptions made during estimation

Respond only with the Markdown content described in the system prompt, no additional commentary."""

    def event_stream():
        # Populated in place by complete_stream() once the generator is exhausted
        stream_result: dict = {}

        try:
            for text in wrapper.complete_stream(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=4096,
                result=stream_result,
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

            done_payload = {
                "type": "done",
                "model": stream_result.get("model"),
                "provider": stream_result.get("provider"),
                "tokens_used": stream_result.get(
                    "usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                ),
                "cost_breakdown": stream_result.get(
                    "cost_breakdown",
                    {"input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0},
                ),
                "finish_reason": stream_result.get("finish_reason", "stop"),
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
