"""
LLM Service for project estimation using CAG (Context-Augmented Generation) architecture.
This service uses the configured LLM provider to generate project estimations based on meeting transcriptions.
"""

import json
from typing import Optional
import anthropic

from cag_estimate.config import get_settings
from cag_estimate.context.examples import ESTIMATION_EXAMPLES


def get_llm_model() -> str:
    """
    Get the LLM model from configured environment variables.

    Returns:
        The model identifier string from LLM_MODEL env var
    """
    settings = get_settings()
    return settings.llm_model


def build_system_prompt() -> str:
    """
    Build the system prompt with role definition and context examples.
    This is the core of the CAG architecture where context drives the calls.
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

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
    "project_name": "string",
    "meeting_summary": "string (concise summary of meeting)",
    "tasks": [
        {{
            "task_id": number,
            "name": "string",
            "description": "string",
            "estimated_hours": number,
            "estimated_cost_usd": number,
            "complexity": "Simple|Medium|High",
            "includes": ["list", "of", "deliverables"]
        }}
    ],
    "summary": {{
        "total_hours": number,
        "total_cost_usd": number,
        "team_size": "string (e.g., '2 developers')",
        "estimated_duration_weeks": number,
        "hourly_rate": number,
        "assumptions": ["list", "of", "key", "assumptions"]
    }}
}}

Be thorough but realistic in your estimations. Use the provided examples as benchmarks."""

    return system_prompt


def estimate_project(meeting_transcription: str, hourly_rate: Optional[int] = None) -> dict:
    """
    Generate project estimation based on meeting transcription using the configured LLM provider.

    Args:
        meeting_transcription: The meeting transcript to analyze and estimate
        hourly_rate: Optional hourly rate for cost calculation (defaults to $40)

    Returns:
        Dictionary containing the generated estimation with tasks, hours, and costs

    Raises:
        ValueError: If the response cannot be parsed as JSON or provider is unsupported
    """
    if hourly_rate is None:
        hourly_rate = 40

    settings = get_settings()

    # Initialize Anthropic client with configured API key
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Get the model based on configured LLM provider
    model = get_llm_model()

    # Build the system prompt with instructions and examples
    system_prompt = build_system_prompt()

    # Prepare the user message with the meeting transcription
    user_message = f"""Based on the following meeting transcription, provide a detailed project estimation with task breakdown, hours, and costs.

MEETING TRANSCRIPTION:
{meeting_transcription}

INSTRUCTIONS:
- Use the hourly rate of ${hourly_rate}/hour for cost calculations
- Be specific and realistic with estimates
- Ensure all tasks are clearly defined with measurable deliverables
- Include tasks for testing, documentation, and deployment
- Provide assumptions made during estimation

Return only valid JSON, no additional text."""

    # Make API call to configured LLM provider
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract and parse the response
    response_text = message.content[0].text

    # Try to extract JSON from the response
    try:
        # Look for JSON content in the response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            estimation = json.loads(json_str)
            return estimation
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}\n\nResponse: {response_text}")


def format_estimation_report(estimation: dict) -> str:
    """
    Format the estimation dictionary into a readable report.

    Args:
        estimation: The estimation dictionary returned by estimate_project

    Returns:
        Formatted string representation of the estimation
    """
    report = f"""
PROJECT ESTIMATION REPORT
==========================

Project Name: {estimation.get('project_name', 'N/A')}

Summary:
--------
Meeting Summary: {estimation.get('meeting_summary', 'N/A')}

Tasks:
------
"""

    for task in estimation.get("tasks", []):
        report += f"""
Task {task.get('task_id')}: {task.get('name')}
  Description: {task.get('description')}
  Complexity: {task.get('complexity')}
  Estimated Hours: {task.get('estimated_hours')}
  Estimated Cost: ${task.get('estimated_cost_usd')}
  Includes: {', '.join(task.get('includes', []))}
"""

    summary = estimation.get("summary", {})
    report += f"""
PROJECT SUMMARY
===============
Total Hours: {summary.get('total_hours')}
Total Cost: ${summary.get('total_cost_usd')}
Team Size: {summary.get('team_size')}
Duration: {summary.get('estimated_duration_weeks')} weeks
Hourly Rate: ${summary.get('hourly_rate')}

Assumptions:
{chr(10).join([f'  - {assumption}' for assumption in summary.get('assumptions', [])])}
"""

    return report
