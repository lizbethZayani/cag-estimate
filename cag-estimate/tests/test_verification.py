"""
Verification pipeline tests for project estimations.

This module contains comprehensive validation tests to ensure:
1. Schema validation - Response matches expected structure
2. Business logic validation - Calculations are correct
3. Reasonableness checks - Values are within realistic bounds
4. Content quality - Descriptions and assumptions are meaningful
5. Consistency checks - Related fields are logically consistent
"""

import json
import sys
from typing import Any, ClassVar

TASK_KEYS = (
    "task_id",
    "name",
    "description",
    "estimated_hours",
    "estimated_cost_usd",
    "complexity",
    "includes",
)
SUMMARY_KEYS = (
    "total_hours",
    "total_cost_usd",
    "team_size",
    "estimated_duration_weeks",
    "hourly_rate",
)
ESTIMATION_KEYS = ("project_name", "meeting_summary", "tasks", "summary")
RESPONSE_KEYS = ("estimation", "model", "provider", "tokens_used", "cost_breakdown")
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")
COST_KEYS = ("input_cost_usd", "output_cost_usd", "total_cost_usd")

HOURS_PER_PERSON_PER_WEEK = 40


def _missing(container: dict[str, Any], keys: tuple[str, ...], label: str) -> list[str]:
    """Return one error per key absent from ``container``, prefixed by ``label``."""
    return [f"Missing {label}.{key}" for key in keys if key not in container]


def _team_count(team_size: str) -> int | None:
    """Extract the head count from text like "2 developers", or None if unparsable."""
    try:
        return int(team_size.split()[0])
    except (ValueError, IndexError):
        return None


class EstimationValidator:
    """Validates project estimation responses."""

    # Complexity ranges (hours) - flexible ranges for real-world scenarios
    COMPLEXITY_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "Simple": (4, 12),
        "Medium": (10, 32),
        "High": (16, 56),
    }

    # Reasonable project ranges
    MIN_PROJECT_HOURS = 8
    MAX_PROJECT_HOURS = 500
    MIN_HOURLY_RATE = 30
    MAX_HOURLY_RATE = 150
    MIN_TASKS = 3
    MAX_TASKS = 20

    @staticmethod
    def _estimation_schema_errors(estimation: dict[str, Any]) -> list[str]:
        errors = []
        for key in ESTIMATION_KEYS:
            if key not in estimation:
                errors.append(f"Missing estimation.{key}")
        for i, task in enumerate(estimation.get("tasks", [])):
            errors.extend(f"Task {i} missing {key}" for key in TASK_KEYS if key not in task)
        if "summary" in estimation:
            errors.extend(_missing(estimation["summary"], SUMMARY_KEYS, "summary"))
        return errors

    @staticmethod
    def validate_schema(response: dict[str, Any]) -> list[str]:
        """
        Validate response schema structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = [f"Missing required key: {key}" for key in RESPONSE_KEYS if key not in response]
        if "estimation" in response:
            errors.extend(EstimationValidator._estimation_schema_errors(response["estimation"]))
        if "tokens_used" in response:
            errors.extend(_missing(response["tokens_used"], TOKEN_KEYS, "tokens_used"))
        if "cost_breakdown" in response:
            errors.extend(_missing(response["cost_breakdown"], COST_KEYS, "cost_breakdown"))
        return errors

    @staticmethod
    def _total_errors(tasks: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
        errors = []
        # Total hours: allow 15% tolerance for LLM variations, minimum 10 hours
        calculated_hours = sum(task.get("estimated_hours", 0) for task in tasks)
        expected_hours = summary.get("total_hours", 0)
        hours_tolerance = max(int(expected_hours * 0.15), 10)
        if abs(calculated_hours - expected_hours) > hours_tolerance:
            errors.append(
                f"Total hours mismatch: calculated={calculated_hours}, "
                f"expected={expected_hours} (tolerance: ±{hours_tolerance})"
            )

        # Total cost: allow 5% tolerance, minimum $50
        calculated_cost = sum(task.get("estimated_cost_usd", 0) for task in tasks)
        expected_cost = summary.get("total_cost_usd", 0)
        cost_tolerance = max(expected_cost * 0.05, 50)
        if abs(calculated_cost - expected_cost) > cost_tolerance:
            errors.append(
                f"Total cost mismatch: calculated=${calculated_cost:,.2f}, "
                f"expected=${expected_cost:,.2f} (tolerance: ±${cost_tolerance:,.2f})"
            )

        # Cost = hours * hourly_rate (approximately)
        hourly_rate = summary.get("hourly_rate", 0)
        if hourly_rate > 0:
            expected_cost_from_hours = calculated_hours * hourly_rate
            if abs(expected_cost_from_hours - calculated_cost) > hourly_rate:
                errors.append(
                    f"Cost calculation inconsistent: hours*rate={expected_cost_from_hours}, "
                    f"sum(costs)={calculated_cost}"
                )
        return errors

    @staticmethod
    def _duration_errors(tasks: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
        team_count = _team_count(summary.get("team_size", ""))
        if team_count is None:  # Skip if team_size format is unexpected
            return []
        calculated_hours = sum(task.get("estimated_hours", 0) for task in tasks)
        duration_weeks = summary.get("estimated_duration_weeks", 0)
        min_duration = calculated_hours / (HOURS_PER_PERSON_PER_WEEK * team_count)
        if duration_weeks < min_duration - 1:  # Allow 1 week tolerance
            return [
                (
                    f"Duration too short: {duration_weeks} weeks for {calculated_hours} hours "
                    f"with {team_count} people (minimum ~{min_duration:.1f} weeks)"
                )
            ]
        return []

    @staticmethod
    def validate_business_logic(estimation: dict[str, Any]) -> list[str]:
        """
        Validate business logic and calculations.

        Returns:
            List of validation errors (empty if valid)
        """
        if "tasks" not in estimation or "summary" not in estimation:
            return []
        tasks = estimation["tasks"]
        summary = estimation["summary"]
        return EstimationValidator._total_errors(
            tasks, summary
        ) + EstimationValidator._duration_errors(tasks, summary)

    @staticmethod
    def _task_hours_errors(tasks: list[dict[str, Any]]) -> list[str]:
        errors = []
        for i, task in enumerate(tasks):
            hours = task.get("estimated_hours", 0)
            complexity = task.get("complexity", "")
            if complexity not in EstimationValidator.COMPLEXITY_RANGES:
                continue
            min_hours, max_hours = EstimationValidator.COMPLEXITY_RANGES[complexity]
            if not (min_hours <= hours <= max_hours * 2):  # Allow 2x buffer for complex tasks
                errors.append(
                    f"Task {i} ({complexity}) hours inconsistent: "
                    f"{hours} hours (expected {min_hours}-{max_hours} for {complexity})"
                )
        return errors

    @staticmethod
    def _range_errors(summary: dict[str, Any]) -> list[str]:
        errors = []
        total_hours = summary.get("total_hours", 0)
        if total_hours < EstimationValidator.MIN_PROJECT_HOURS:
            errors.append(
                f"Project too small: {total_hours} hours "
                f"(minimum {EstimationValidator.MIN_PROJECT_HOURS})"
            )
        if total_hours > EstimationValidator.MAX_PROJECT_HOURS:
            errors.append(
                f"Project too large: {total_hours} hours "
                f"(maximum {EstimationValidator.MAX_PROJECT_HOURS})"
            )
        hourly_rate = summary.get("hourly_rate", 0)
        if hourly_rate < EstimationValidator.MIN_HOURLY_RATE:
            errors.append(
                f"Hourly rate too low: ${hourly_rate} "
                f"(minimum ${EstimationValidator.MIN_HOURLY_RATE})"
            )
        if hourly_rate > EstimationValidator.MAX_HOURLY_RATE:
            errors.append(
                f"Hourly rate too high: ${hourly_rate} "
                f"(maximum ${EstimationValidator.MAX_HOURLY_RATE})"
            )
        return errors

    @staticmethod
    def validate_reasonableness(estimation: dict[str, Any]) -> list[str]:
        """
        Validate values are within reasonable ranges.

        Returns:
            List of validation errors (empty if valid)
        """
        if "tasks" not in estimation or "summary" not in estimation:
            return []
        tasks = estimation["tasks"]
        errors = []
        if len(tasks) < EstimationValidator.MIN_TASKS:
            errors.append(f"Too few tasks: {len(tasks)} (minimum {EstimationValidator.MIN_TASKS})")
        if len(tasks) > EstimationValidator.MAX_TASKS:
            errors.append(f"Too many tasks: {len(tasks)} (maximum {EstimationValidator.MAX_TASKS})")
        errors.extend(EstimationValidator._task_hours_errors(tasks))
        errors.extend(EstimationValidator._range_errors(estimation["summary"]))
        return errors

    @staticmethod
    def _task_quality_errors(i: int, task: dict[str, Any]) -> list[str]:
        errors = []
        description = task.get("description", "").strip()
        if len(description) < 20:
            errors.append(f"Task {i} description too short: '{description}'")
        if len(description) > 500:
            errors.append(f"Task {i} description too long: {len(description)} chars")

        includes = task.get("includes", [])
        if len(includes) < 2:
            errors.append(f"Task {i} includes too few items: {len(includes)} (minimum 2)")
        if len(includes) > 15:
            errors.append(f"Task {i} includes too many items: {len(includes)} (maximum 15)")
        errors.extend(
            f"Task {i} has very short include: '{include}'"
            for include in includes
            if len(include) < 3
        )
        return errors

    @staticmethod
    def _text_quality_errors(estimation: dict[str, Any]) -> list[str]:
        errors = []
        project_name = estimation.get("project_name", "").strip()
        if len(project_name) < 5:
            errors.append(f"Project name too short: '{project_name}'")
        if len(project_name) > 100:
            errors.append(f"Project name too long: {len(project_name)} chars")

        meeting_summary = estimation.get("meeting_summary", "").strip()
        if len(meeting_summary) < 50:
            errors.append(f"Meeting summary too short: {len(meeting_summary)} chars (minimum 50)")
        if len(meeting_summary) > 2000:
            errors.append(f"Meeting summary too long: {len(meeting_summary)} chars (maximum 2000)")
        return errors

    @staticmethod
    def _assumption_errors(summary: dict[str, Any]) -> list[str]:
        assumptions = summary.get("assumptions", [])
        errors = []
        if len(assumptions) < 3:
            errors.append(f"Too few assumptions: {len(assumptions)} (minimum 3)")
        if len(assumptions) > 20:
            errors.append(f"Too many assumptions: {len(assumptions)} (maximum 20)")
        errors.extend(
            f"Assumption too short: '{assumption}'"
            for assumption in assumptions
            if len(assumption) < 10
        )
        return errors

    @staticmethod
    def validate_content_quality(estimation: dict[str, Any]) -> list[str]:
        """
        Validate content quality and meaningfulness.

        Returns:
            List of validation errors (empty if valid)
        """
        if "tasks" not in estimation or "summary" not in estimation:
            return []
        errors = []
        for i, task in enumerate(estimation["tasks"]):
            errors.extend(EstimationValidator._task_quality_errors(i, task))
        errors.extend(EstimationValidator._text_quality_errors(estimation))
        errors.extend(EstimationValidator._assumption_errors(estimation["summary"]))
        return errors

    @staticmethod
    def _complexity_errors(tasks: list[dict[str, Any]]) -> list[str]:
        """Check complexity roughly matches hours (lenient for real-world scenarios)."""
        errors = []
        for i, task in enumerate(tasks):
            hours = task.get("estimated_hours", 0)
            complexity = task.get("complexity", "").lower()
            if "simple" in complexity and hours > 16:
                errors.append(f"Task {i}: marked Simple but has {hours} hours (expected <16)")
            elif "high" in complexity and hours < 14:
                errors.append(
                    f"Task {i}: marked High but has only {hours} hours (expected >14 for High)"
                )
        return errors

    @staticmethod
    def _team_size_errors(summary: dict[str, Any]) -> list[str]:
        """Check the duration is plausible for the team size and total hours."""
        team_count = _team_count(summary.get("team_size", ""))
        if team_count is None:
            return []
        total_hours = summary.get("total_hours", 0)
        expected_min_duration = total_hours / (team_count * HOURS_PER_PERSON_PER_WEEK)
        actual_duration = summary.get("estimated_duration_weeks", 0)
        # Allow some buffer for meetings, communication, integration
        if actual_duration < expected_min_duration * 0.8:
            return [
                (
                    f"Duration too optimistic: {actual_duration} weeks vs "
                    f"calculated minimum {expected_min_duration:.1f} weeks"
                )
            ]
        return []

    @staticmethod
    def validate_consistency(estimation: dict[str, Any]) -> list[str]:
        """
        Validate logical consistency between related fields.

        Returns:
            List of validation errors (empty if valid)
        """
        if "tasks" not in estimation or "summary" not in estimation:
            return []
        tasks = estimation["tasks"]
        errors = EstimationValidator._complexity_errors(tasks)
        errors.extend(EstimationValidator._team_size_errors(estimation["summary"]))
        task_ids = [task.get("task_id") for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Duplicate task IDs found")
        return errors

    @classmethod
    def validate_complete(cls, response: dict[str, Any]) -> dict[str, Any]:
        """
        Run complete validation on estimation response.

        Returns:
            Dict with validation results and summary
        """
        schema_errors = cls.validate_schema(response)
        if schema_errors:
            return {
                "valid": False,
                "stage": "schema",
                "errors": schema_errors,
                "error_count": len(schema_errors),
            }

        estimation = response.get("estimation", {})

        business_errors = cls.validate_business_logic(estimation)
        reasonableness_errors = cls.validate_reasonableness(estimation)
        quality_errors = cls.validate_content_quality(estimation)
        consistency_errors = cls.validate_consistency(estimation)

        all_errors = business_errors + reasonableness_errors + quality_errors + consistency_errors

        return {
            "valid": len(all_errors) == 0,
            "stage": "complete",
            "business_logic_errors": business_errors,
            "reasonableness_errors": reasonableness_errors,
            "quality_errors": quality_errors,
            "consistency_errors": consistency_errors,
            "total_errors": len(all_errors),
            "all_errors": all_errors,
        }


def _task(
    task_id: int,
    name: str,
    description: str,
    hours: int,
    cost: int,
    complexity: str,
    includes: list[str],
) -> dict[str, Any]:
    """Build one sample task."""
    return {
        "task_id": task_id,
        "name": name,
        "description": description,
        "estimated_hours": hours,
        "estimated_cost_usd": cost,
        "complexity": complexity,
        "includes": includes,
    }


def _sample_tasks() -> list[dict[str, Any]]:
    return [
        _task(
            1,
            "User Authentication",
            "Implement user registration, login, and profile management",
            16,
            640,
            "Medium",
            ["Registration flow", "Login", "Profile management"],
        ),
        _task(
            2,
            "Location Services",
            "Integrate location services to find nearby grocery shops",
            20,
            800,
            "Medium",
            ["GPS integration", "Store database", "Maps API"],
        ),
        _task(
            3,
            "Price Comparison Engine",
            "Build algorithm to compare product prices across stores",
            32,
            1280,
            "High",
            ["Price scraping", "Data normalization", "Comparison logic", "Caching"],
        ),
        _task(
            4,
            "Food Labels & Ingredients",
            "Implement food label analysis and ingredient filtering",
            28,
            1120,
            "High",
            ["Label OCR", "Ingredient database", "Organic filter", "Natural food ranking"],
        ),
        _task(
            5,
            "AI Shopping Cart Agent",
            "Create AI chat agent for shopping cart creation and recommendations",
            36,
            1440,
            "High",
            ["NLP integration", "Recommendation engine", "Chat UI", "Cart optimization"],
        ),
        _task(
            6,
            "Testing & QA",
            "Comprehensive testing including unit, integration, and E2E tests",
            24,
            960,
            "Medium",
            ["Unit tests", "Integration tests", "E2E tests", "Bug fixes"],
        ),
    ]


def _sample_summary() -> dict[str, Any]:
    return {
        "total_hours": 156,
        "total_cost_usd": 6240,
        "team_size": "2 iOS developers + 1 backend engineer",
        "estimated_duration_weeks": 6,
        "hourly_rate": 40,
        "assumptions": [
            "iOS 15+ target",
            "Third-party price data APIs available",
            "Organic food label database accessible",
            "Team has experience with Swift and backend APIs",
        ],
    }


def _sample_response() -> dict[str, Any]:
    """Sample valid estimation response."""
    return {
        "estimation": {
            "project_name": "Mobile Grocery Price Comparison App",
            "meeting_summary": (
                "Build iOS mobile app for comparing grocery prices and natural food labels"
            ),
            "tasks": _sample_tasks(),
            "summary": _sample_summary(),
        },
        "model": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "tokens_used": {"input_tokens": 3577, "output_tokens": 2295, "total_tokens": 5872},
        "cost_breakdown": {
            "input_cost_usd": 0.000011,
            "output_cost_usd": 0.000034,
            "total_cost_usd": 0.000045,
        },
    }


def _modified_estimation(section: str, key: str, value: Any) -> dict[str, Any]:
    """Return a deep copy of the sample estimation with one summary-level field changed."""
    estimation = json.loads(json.dumps(_sample_response()["estimation"]))
    estimation[section][key] = value
    return estimation


# Test fixtures and tests
class TestEstimationValidation:
    """Test suite for estimation validation."""

    def sample_response(self) -> dict[str, Any]:
        """Sample valid estimation response."""
        return _sample_response()

    def run_all_tests(self) -> list[tuple[str, bool, list[str]]]:
        """Run all tests and return summary."""
        sample = self.sample_response()
        estimation = sample["estimation"]
        validator = EstimationValidator

        complete = validator.validate_complete(sample)
        wrong_total = validator.validate_business_logic(
            _modified_estimation("summary", "total_hours", 9999)
        )
        wrong_rate = validator.validate_reasonableness(
            _modified_estimation("summary", "hourly_rate", 500)
        )
        invalid_schema = validator.validate_schema({"model": "claude-haiku-4-5-20251001"})

        schema = validator.validate_schema(sample)
        business = validator.validate_business_logic(estimation)
        reasonableness = validator.validate_reasonableness(estimation)
        quality = validator.validate_content_quality(estimation)
        consistency = validator.validate_consistency(estimation)

        return [
            ("Schema Validation", len(schema) == 0, schema),
            ("Business Logic Validation", len(business) == 0, business),
            ("Reasonableness Validation", len(reasonableness) == 0, reasonableness),
            ("Quality Validation", len(quality) == 0, quality),
            ("Consistency Validation", len(consistency) == 0, consistency),
            ("Complete Validation", complete["valid"], complete.get("all_errors", [])),
            ("Invalid Schema Detection", len(invalid_schema) > 0, []),
            ("Calculation Error Detection", len(wrong_total) > 0, []),
            ("Reasonableness Bounds", len(wrong_rate) > 0, []),
        ]


def _emit(line: str = "") -> None:
    """Write one line of script output (application code must not use print)."""
    sys.stdout.write(f"{line}\n")


def _emit_result(test_name: str, success: bool, errors: list[str]) -> None:
    """Write one result line plus up to two error lines for a failure."""
    _emit(f"{'✅ PASS' if success else '❌ FAIL'} | {test_name}")
    if success:
        return
    for error in errors[:2]:  # Show first 2 errors
        _emit(f"       └─ {error}")
    if len(errors) > 2:
        _emit(f"       └─ ... and {len(errors) - 2} more errors")


def main() -> int:
    """Run the verification pipeline and return the process exit code."""
    _emit("=" * 70)
    _emit("CAG ESTIMATE - VERIFICATION PIPELINE")
    _emit("=" * 70)
    _emit()

    results = TestEstimationValidation().run_all_tests()
    for test_name, success, errors in results:
        _emit_result(test_name, success, errors)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    _emit()
    _emit("=" * 70)
    _emit(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
    _emit("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
