"""Jinja2 loader for versioned prompt templates.

On-disk layout: ``prompts/<use_case>/<version>/<role>.j2``. Switching to a new
prompt version is a string change at the call site (``version="v2"``).
"""

import json
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from cag_estimate.context.examples import ESTIMATION_EXAMPLES
from cag_estimate.schemas.estimation import (
    COMPLEXITY_HOURS,
    MAX_HOURLY_RATE,
    MAX_TASKS,
    MIN_HOURLY_RATE,
    MIN_TASKS,
    EstimationRequest,
)

OutputFormat = Literal["json", "markdown"]
_OUTPUT_FORMATS = ("json", "markdown")
_BASE_DIR = Path(__file__).resolve().parent


def _to_pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
    keep_trailing_newline=True,
)
_env.filters["pretty_json"] = _to_pretty_json


def _render(role: str, version: str, context: dict[str, Any]) -> str:
    try:
        template = _env.get_template(f"estimation/{version}/{role}.j2")
    except TemplateNotFound as error:
        raise ValueError(f"Unknown estimation prompt version: {version!r}") from error
    return template.render(**context)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    output_format: OutputFormat = "json",
) -> tuple[str, str]:
    """Render the ``(system, user)`` prompts for the estimation use case.

    Args:
        request: The estimation request (description and hourly rate).
        version: Prompt template version directory, e.g. ``"v1"``.
        output_format: ``"json"`` for structured output, ``"markdown"`` for
            the human-readable streamed response.

    Raises:
        ValueError: If the version or the output format is unknown.
    """
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(f"Unknown output_format: {output_format!r}")
    context = {
        "description": request.transcription,
        "hourly_rate": request.hourly_rate,
        "output_format": output_format,
        "examples": ESTIMATION_EXAMPLES,
        "complexity_hours": COMPLEXITY_HOURS,
        "min_tasks": MIN_TASKS,
        "max_tasks": MAX_TASKS,
        "min_rate": MIN_HOURLY_RATE,
        "max_rate": MAX_HOURLY_RATE,
    }
    return _render("system", version, context), _render("user", version, context)
