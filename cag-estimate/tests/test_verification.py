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
from typing import Any, Dict, List

try:
    import pytest
except ImportError:
    pytest = None


class EstimationValidator:
    """Validates project estimation responses."""

    # Complexity ranges (hours) - flexible ranges for real-world scenarios
    COMPLEXITY_RANGES = {
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
    def validate_schema(response: Dict[str, Any]) -> List[str]:
        """
        Validate response schema structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check top-level keys
        required_keys = ["estimation", "model", "provider", "tokens_used", "cost_breakdown"]
        for key in required_keys:
            if key not in response:
                errors.append(f"Missing required key: {key}")

        # Validate estimation structure
        if "estimation" in response:
            estimation = response["estimation"]
            estimation_keys = ["project_name", "meeting_summary", "tasks", "summary"]
            for key in estimation_keys:
                if key not in estimation:
                    errors.append(f"Missing estimation.{key}")

            # Validate tasks structure
            if "tasks" in estimation:
                for i, task in enumerate(estimation["tasks"]):
                    task_keys = [
                        "task_id",
                        "name",
                        "description",
                        "estimated_hours",
                        "estimated_cost_usd",
                        "complexity",
                        "includes",
                    ]
                    for key in task_keys:
                        if key not in task:
                            errors.append(f"Task {i} missing {key}")

            # Validate summary structure
            if "summary" in estimation:
                summary = estimation["summary"]
                summary_keys = [
                    "total_hours",
                    "total_cost_usd",
                    "team_size",
                    "estimated_duration_weeks",
                    "hourly_rate",
                ]
                for key in summary_keys:
                    if key not in summary:
                        errors.append(f"Missing summary.{key}")

        # Validate tokens_used
        if "tokens_used" in response:
            tokens = response["tokens_used"]
            for key in ["input_tokens", "output_tokens", "total_tokens"]:
                if key not in tokens:
                    errors.append(f"Missing tokens_used.{key}")

        # Validate cost_breakdown
        if "cost_breakdown" in response:
            costs = response["cost_breakdown"]
            for key in ["input_cost_usd", "output_cost_usd", "total_cost_usd"]:
                if key not in costs:
                    errors.append(f"Missing cost_breakdown.{key}")

        return errors

    @staticmethod
    def validate_business_logic(estimation: Dict[str, Any]) -> List[str]:
        """
        Validate business logic and calculations.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if "tasks" not in estimation or "summary" not in estimation:
            return errors

        tasks = estimation["tasks"]
        summary = estimation["summary"]

        # Validate total hours calculation (allow 15% tolerance for LLM variations)
        calculated_hours = sum(task.get("estimated_hours", 0) for task in tasks)
        expected_hours = summary.get("total_hours", 0)
        hours_tolerance = max(int(expected_hours * 0.15), 10)  # 15% or minimum 10 hours
        if abs(calculated_hours - expected_hours) > hours_tolerance:
            errors.append(
                f"Total hours mismatch: calculated={calculated_hours}, "
                f"expected={expected_hours} (tolerance: ±{hours_tolerance})"
            )

        # Validate total cost calculation (allow 5% tolerance)
        calculated_cost = sum(task.get("estimated_cost_usd", 0) for task in tasks)
        expected_cost = summary.get("total_cost_usd", 0)
        cost_tolerance = max(expected_cost * 0.05, 50)  # 5% or minimum $50
        if abs(calculated_cost - expected_cost) > cost_tolerance:
            errors.append(
                f"Total cost mismatch: calculated=${calculated_cost:,.2f}, "
                f"expected=${expected_cost:,.2f} (tolerance: ±${cost_tolerance:,.2f})"
            )

        # Validate cost = hours * hourly_rate (approximately)
        hourly_rate = summary.get("hourly_rate", 0)
        if hourly_rate > 0:
            expected_cost_from_hours = calculated_hours * hourly_rate
            if abs(expected_cost_from_hours - calculated_cost) > hourly_rate:
                errors.append(
                    f"Cost calculation inconsistent: hours*rate={expected_cost_from_hours}, "
                    f"sum(costs)={calculated_cost}"
                )

        # Validate duration calculation
        duration_weeks = summary.get("estimated_duration_weeks", 0)
        team_size_str = summary.get("team_size", "")
        # Extract number from "X developers" or similar
        try:
            team_count = int(team_size_str.split()[0])
            # Rough check: duration_weeks should be >= total_hours / (40 hours/week * team_count)
            min_duration = calculated_hours / (40 * team_count)
            if duration_weeks < min_duration - 1:  # Allow 1 week tolerance
                errors.append(
                    f"Duration too short: {duration_weeks} weeks for {calculated_hours} hours "
                    f"with {team_count} people (minimum ~{min_duration:.1f} weeks)"
                )
        except (ValueError, IndexError):
            pass  # Skip if team_size format is unexpected

        return errors

    @staticmethod
    def validate_reasonableness(estimation: Dict[str, Any]) -> List[str]:
        """
        Validate values are within reasonable ranges.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if "tasks" not in estimation or "summary" not in estimation:
            return errors

        tasks = estimation["tasks"]
        summary = estimation["summary"]

        # Validate task count
        if len(tasks) < EstimationValidator.MIN_TASKS:
            errors.append(
                f"Too few tasks: {len(tasks)} " f"(minimum {EstimationValidator.MIN_TASKS})"
            )
        if len(tasks) > EstimationValidator.MAX_TASKS:
            errors.append(
                f"Too many tasks: {len(tasks)} " f"(maximum {EstimationValidator.MAX_TASKS})"
            )

        # Validate individual task hours
        for i, task in enumerate(tasks):
            hours = task.get("estimated_hours", 0)
            complexity = task.get("complexity", "")

            if complexity in EstimationValidator.COMPLEXITY_RANGES:
                min_hours, max_hours = EstimationValidator.COMPLEXITY_RANGES[complexity]
                if not (min_hours <= hours <= max_hours * 2):  # Allow 2x buffer for complex tasks
                    errors.append(
                        f"Task {i} ({complexity}) hours inconsistent: "
                        f"{hours} hours (expected {min_hours}-{max_hours} for {complexity})"
                    )

        # Validate total project hours
        total_hours = summary.get("total_hours", 0)
        if total_hours < EstimationValidator.MIN_PROJECT_HOURS:
            errors.append(f"Project too small: {total_hours} hours " f"(minimum {EstimationValidator.MIN_PROJECT_HOURS})")
        if total_hours > EstimationValidator.MAX_PROJECT_HOURS:
            errors.append(f"Project too large: {total_hours} hours " f"(maximum {EstimationValidator.MAX_PROJECT_HOURS})")

        # Validate hourly rate
        hourly_rate = summary.get("hourly_rate", 0)
        if hourly_rate < EstimationValidator.MIN_HOURLY_RATE:
            errors.append(f"Hourly rate too low: ${hourly_rate} " f"(minimum ${EstimationValidator.MIN_HOURLY_RATE})")
        if hourly_rate > EstimationValidator.MAX_HOURLY_RATE:
            errors.append(f"Hourly rate too high: ${hourly_rate} " f"(maximum ${EstimationValidator.MAX_HOURLY_RATE})")

        return errors

    @staticmethod
    def validate_content_quality(estimation: Dict[str, Any]) -> List[str]:
        """
        Validate content quality and meaningfulness.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if "tasks" not in estimation or "summary" not in estimation:
            return errors

        tasks = estimation["tasks"]
        summary = estimation["summary"]

        # Validate task descriptions
        for i, task in enumerate(tasks):
            description = task.get("description", "").strip()
            if len(description) < 20:
                errors.append(f"Task {i} description too short: '{description}'")
            if len(description) > 500:
                errors.append(f"Task {i} description too long: {len(description)} chars")

            # Validate includes list
            includes = task.get("includes", [])
            if len(includes) < 2:
                errors.append(f"Task {i} includes too few items: {len(includes)} (minimum 2)")
            if len(includes) > 15:
                errors.append(f"Task {i} includes too many items: {len(includes)} (maximum 15)")

            # Check for meaningful includes
            for include in includes:
                if len(include) < 3:
                    errors.append(f"Task {i} has very short include: '{include}'")

        # Validate project name
        project_name = estimation.get("project_name", "").strip()
        if len(project_name) < 5:
            errors.append(f"Project name too short: '{project_name}'")
        if len(project_name) > 100:
            errors.append(f"Project name too long: {len(project_name)} chars")

        # Validate meeting summary
        meeting_summary = estimation.get("meeting_summary", "").strip()
        if len(meeting_summary) < 50:
            errors.append(f"Meeting summary too short: {len(meeting_summary)} chars (minimum 50)")
        if len(meeting_summary) > 2000:
            errors.append(f"Meeting summary too long: {len(meeting_summary)} chars (maximum 2000)")

        # Validate assumptions
        assumptions = summary.get("assumptions", [])
        if len(assumptions) < 3:
            errors.append(f"Too few assumptions: {len(assumptions)} (minimum 3)")
        if len(assumptions) > 20:
            errors.append(f"Too many assumptions: {len(assumptions)} (maximum 20)")

        for assumption in assumptions:
            if len(assumption) < 10:
                errors.append(f"Assumption too short: '{assumption}'")

        return errors

    @staticmethod
    def validate_consistency(estimation: Dict[str, Any]) -> List[str]:
        """
        Validate logical consistency between related fields.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if "tasks" not in estimation or "summary" not in estimation:
            return errors

        tasks = estimation["tasks"]
        summary = estimation["summary"]

        # Check that complexity roughly matches hours (lenient for real-world scenarios)
        for i, task in enumerate(tasks):
            hours = task.get("estimated_hours", 0)
            complexity = task.get("complexity", "").lower()

            if "simple" in complexity and hours > 16:
                errors.append(f"Task {i}: marked Simple but has {hours} hours (expected <16)")
            elif "high" in complexity and hours < 14:
                errors.append(
                    f"Task {i}: marked High but has only {hours} hours (expected >14 for High)"
                )

        # Check that team size matches hours
        team_size_str = summary.get("team_size", "")
        total_hours = summary.get("total_hours", 0)
        try:
            team_count = int(team_size_str.split()[0])
            # A team of 1 working on a 200-hour project would take 5 weeks
            # A team of 2 working on a 200-hour project would take 2.5 weeks
            hours_per_person_per_week = 40
            expected_min_duration = total_hours / (team_count * hours_per_person_per_week)
            actual_duration = summary.get("estimated_duration_weeks", 0)

            # Allow some buffer for meetings, communication, integration
            if actual_duration < expected_min_duration * 0.8:
                errors.append(
                    f"Duration too optimistic: {actual_duration} weeks vs "
                    f"calculated minimum {expected_min_duration:.1f} weeks"
                )
        except (ValueError, IndexError):
            pass

        # Check task_id uniqueness and sequence
        task_ids = [task.get("task_id") for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Duplicate task IDs found")

        return errors

    @classmethod
    def validate_complete(cls, response: Dict[str, Any]) -> Dict[str, Any]:
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

        all_errors = (
            business_errors + reasonableness_errors + quality_errors + consistency_errors
        )

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


# Test fixtures and tests
class TestEstimationValidation:
    """Test suite for estimation validation."""

    def sample_response(self) -> Dict[str, Any]:
        """Sample valid estimation response."""
        return {
            "estimation": {
                "project_name": "Mobile Grocery Price Comparison App",
                "meeting_summary": "Build iOS mobile app for comparing grocery prices and natural food labels",
                "tasks": [
                    {
                        "task_id": 1,
                        "name": "User Authentication",
                        "description": "Implement user registration, login, and profile management",
                        "estimated_hours": 16,
                        "estimated_cost_usd": 640,
                        "complexity": "Medium",
                        "includes": ["Registration flow", "Login", "Profile management"],
                    },
                    {
                        "task_id": 2,
                        "name": "Location Services",
                        "description": "Integrate location services to find nearby grocery shops",
                        "estimated_hours": 20,
                        "estimated_cost_usd": 800,
                        "complexity": "Medium",
                        "includes": ["GPS integration", "Store database", "Maps API"],
                    },
                    {
                        "task_id": 3,
                        "name": "Price Comparison Engine",
                        "description": "Build algorithm to compare product prices across stores",
                        "estimated_hours": 32,
                        "estimated_cost_usd": 1280,
                        "complexity": "High",
                        "includes": [
                            "Price scraping",
                            "Data normalization",
                            "Comparison logic",
                            "Caching",
                        ],
                    },
                    {
                        "task_id": 4,
                        "name": "Food Labels & Ingredients",
                        "description": "Implement food label analysis and ingredient filtering",
                        "estimated_hours": 28,
                        "estimated_cost_usd": 1120,
                        "complexity": "High",
                        "includes": [
                            "Label OCR",
                            "Ingredient database",
                            "Organic filter",
                            "Natural food ranking",
                        ],
                    },
                    {
                        "task_id": 5,
                        "name": "AI Shopping Cart Agent",
                        "description": "Create AI chat agent for shopping cart creation and recommendations",
                        "estimated_hours": 36,
                        "estimated_cost_usd": 1440,
                        "complexity": "High",
                        "includes": [
                            "NLP integration",
                            "Recommendation engine",
                            "Chat UI",
                            "Cart optimization",
                        ],
                    },
                    {
                        "task_id": 6,
                        "name": "Testing & QA",
                        "description": "Comprehensive testing including unit, integration, and E2E tests",
                        "estimated_hours": 24,
                        "estimated_cost_usd": 960,
                        "complexity": "Medium",
                        "includes": ["Unit tests", "Integration tests", "E2E tests", "Bug fixes"],
                    },
                ],
                "summary": {
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
                },
            },
            "model": "claude-haiku-4-5-20251001",
            "provider": "anthropic",
            "tokens_used": {
                "input_tokens": 3577,
                "output_tokens": 2295,
                "total_tokens": 5872,
            },
            "cost_breakdown": {
                "input_cost_usd": 0.000011,
                "output_cost_usd": 0.000034,
                "total_cost_usd": 0.000045,
            },
        }

    def run_all_tests(self):
        """Run all tests and return summary."""
        results = []
        sample = self.sample_response()

        # Test 1: Schema validation
        errors = EstimationValidator.validate_schema(sample)
        results.append(("Schema Validation", len(errors) == 0, errors))

        # Test 2: Business logic validation
        estimation = sample["estimation"]
        errors = EstimationValidator.validate_business_logic(estimation)
        results.append(("Business Logic Validation", len(errors) == 0, errors))

        # Test 3: Reasonableness validation
        errors = EstimationValidator.validate_reasonableness(estimation)
        results.append(("Reasonableness Validation", len(errors) == 0, errors))

        # Test 4: Quality validation
        errors = EstimationValidator.validate_content_quality(estimation)
        results.append(("Quality Validation", len(errors) == 0, errors))

        # Test 5: Consistency validation
        errors = EstimationValidator.validate_consistency(estimation)
        results.append(("Consistency Validation", len(errors) == 0, errors))

        # Test 6: Complete validation
        result = EstimationValidator.validate_complete(sample)
        results.append(("Complete Validation", result["valid"], result.get("all_errors", [])))

        # Test 7: Invalid schema detection
        invalid_response = {"model": "claude-haiku-4-5-20251001"}
        errors = EstimationValidator.validate_schema(invalid_response)
        results.append(("Invalid Schema Detection", len(errors) > 0, []))

        # Test 8: Calculation error detection
        estimation_copy = json.loads(json.dumps(sample["estimation"]))
        estimation_copy["summary"]["total_hours"] = 9999
        errors = EstimationValidator.validate_business_logic(estimation_copy)
        results.append(("Calculation Error Detection", len(errors) > 0, []))

        # Test 9: Reasonableness bounds
        estimation_copy = json.loads(json.dumps(sample["estimation"]))
        estimation_copy["summary"]["hourly_rate"] = 500
        errors = EstimationValidator.validate_reasonableness(estimation_copy)
        results.append(("Reasonableness Bounds", len(errors) > 0, []))

        return results


if __name__ == "__main__":
    # Run verification pipeline
    import sys

    print("=" * 70)
    print("CAG ESTIMATE - VERIFICATION PIPELINE")
    print("=" * 70)
    print()

    test_suite = TestEstimationValidation()
    results = test_suite.run_all_tests()

    passed = 0
    failed = 0

    for test_name, success, errors in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
        if errors and not success:
            for error in errors[:2]:  # Show first 2 errors
                print(f"       └─ {error}")
            if len(errors) > 2:
                print(f"       └─ ... and {len(errors) - 2} more errors")
        if success:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
