"""
Routers for project estimation endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import anthropic

from cag_estimate.config import get_settings
from cag_estimate.context.examples import ESTIMATION_EXAMPLES

router = APIRouter(prefix="/api/v1", tags=["estimations"])

# Token pricing for Claude 3.5 Sonnet (per 1M tokens)
TOKEN_PRICES = {
    "input": 0.003,      # $3 per 1M input tokens
    "output": 0.015,     # $15 per 1M output tokens
}


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


def calculate_token_cost(input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate the cost of tokens used.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Dictionary with cost breakdown
    """
    input_cost = (input_tokens / 1_000_000) * TOKEN_PRICES["input"]
    output_cost = (output_tokens / 1_000_000) * TOKEN_PRICES["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


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

    Args:
        request: EstimationRequest containing meeting transcription and optional hourly_rate

    Returns:
        EstimationResponse with estimation, model info, token usage, and costs

    Raises:
        HTTPException: If estimation fails or response parsing fails
    """
    settings = get_settings()

    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

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

        # Call Anthropic API with CAG pattern
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract response text
        response_text = message.content[0].text

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

        # Extract token usage
        tokens_used = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }

        # Calculate costs
        cost_breakdown = calculate_token_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )

        # Build response
        return EstimationResponse(
            estimation=estimation,
            model=settings.llm_model,
            provider=settings.llm_provider,
            tokens_used=tokens_used,
            cost_breakdown=cost_breakdown,
        )

    except HTTPException:
        raise
    except anthropic.APIError as e:
        raise HTTPException(
            status_code=500, detail=f"Anthropic API error: {str(e)}"
        )
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

    Event types sent to the client, one JSON object per "data:" line:
    - {"type": "token", "content": "..."}   one text chunk, as generated by the model
    - {"type": "done", "model": ..., "provider": ...,
       "tokens_used": {...}, "cost_breakdown": {...}, "finish_reason": ...}
    - {"type": "error", "message": "..."}
    """
    settings = get_settings()
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
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        try:
            with client.messages.stream(
                model=settings.llm_model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

                final_message = stream.get_final_message()

            tokens_used = {
                "input_tokens": final_message.usage.input_tokens,
                "output_tokens": final_message.usage.output_tokens,
                "total_tokens": final_message.usage.input_tokens
                + final_message.usage.output_tokens,
            }
            cost_breakdown = calculate_token_cost(
                final_message.usage.input_tokens, final_message.usage.output_tokens
            )

            done_payload = {
                "type": "done",
                "model": settings.llm_model,
                "provider": settings.llm_provider,
                "tokens_used": tokens_used,
                "cost_breakdown": cost_breakdown,
                "finish_reason": final_message.stop_reason,
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
