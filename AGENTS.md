# Code Review Rules

## Python
- Target Python 3.11+; use type hints on public functions and modern unions (`X | None`).
- Keep layering: routers (HTTP only) -> services (orchestration) -> prompts / guardrails / cache / schemas.
- No business logic or prompt text inside routers; prompts live in versioned Jinja2 templates.
- Validate LLM output with Pydantic models; never parse JSON with string slicing.
- Never return raw exception text to API clients; map errors to 400 / 422 / 502.
- No secrets or API keys in code; read configuration through `Settings`.
- Use structlog for logging; no `print` in application code.
- Cache and guardrail failures must fail soft where documented, never crash a request.
- Keep functions small and single-purpose; no dead or duplicated code.
