#!/usr/bin/env python3
"""
API Verification Script - Test CAG Estimate API with real examples.

This script tests the actual API endpoint with example transcriptions
and validates the response through the complete verification pipeline.
"""

import sys
from typing import Any

import requests
from test_verification import EstimationValidator

ERROR_SECTIONS = (
    ("Business Logic Errors", "business_logic_errors"),
    ("Reasonableness Errors", "reasonableness_errors"),
    ("Quality Errors", "quality_errors"),
    ("Consistency Errors", "consistency_errors"),
)


class APIError(RuntimeError):
    """The API answered with a non-200 status."""


class APIVerifier:
    """Verifies CAG Estimate API responses."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001") -> None:
        self.base_url = base_url
        self.validator = EstimationValidator()

    def check_api_health(self) -> bool:
        """Check if API is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def estimate_project(self, transcription: str, hourly_rate: int = 40) -> dict[str, Any]:
        """Call the API estimation endpoint."""
        payload = {"transcription": transcription, "hourly_rate": hourly_rate}

        response = requests.post(
            f"{self.base_url}/api/v1/estimate", json=payload, timeout=60
        )

        if response.status_code != 200:
            raise APIError(f"API error: {response.status_code} - {response.text}")

        return response.json()

    def verify_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Verify API response through complete validation pipeline."""
        return self.validator.validate_complete(response)

    def print_estimation_summary(self, estimation: dict[str, Any]) -> None:
        """Pretty print estimation summary."""
        summary = estimation.get("summary", {})
        print()
        print("📊 ESTIMATION SUMMARY")
        print("-" * 70)
        print(f"Project:      {estimation.get('project_name', 'N/A')}")
        print(f"Total Hours:  {summary.get('total_hours', 0)}")
        print(f"Total Cost:   ${summary.get('total_cost_usd', 0):,.2f}")
        print(f"Team Size:    {summary.get('team_size', 'N/A')}")
        print(f"Duration:     {summary.get('estimated_duration_weeks', 0)} weeks")
        print(f"Hourly Rate:  ${summary.get('hourly_rate', 0)}/hour")
        print()

        # Task breakdown
        print("📋 TASK BREAKDOWN")
        print("-" * 70)
        for task in estimation.get("tasks", [])[:5]:  # Show first 5 tasks
            print(f"  {task.get('task_id')}. {task.get('name')} ({task.get('complexity')})")
            print(f"     Hours: {task.get('estimated_hours')} | Cost: ${task.get('estimated_cost_usd'):,}")

        if len(estimation.get("tasks", [])) > 5:
            print(f"  ... and {len(estimation['tasks']) - 5} more tasks")
        print()

    def print_token_costs(self, response: dict[str, Any]) -> None:
        """Print token usage and costs."""
        tokens = response.get("tokens_used", {})
        costs = response.get("cost_breakdown", {})

        print("💰 TOKEN USAGE & COSTS")
        print("-" * 70)
        print(f"Input Tokens:   {tokens.get('input_tokens', 0):,}")
        print(f"Output Tokens:  {tokens.get('output_tokens', 0):,}")
        print(f"Total Tokens:   {tokens.get('total_tokens', 0):,}")
        print()
        print(f"Input Cost:     ${costs.get('input_cost_usd', 0):.6f}")
        print(f"Output Cost:    ${costs.get('output_cost_usd', 0):.6f}")
        print(f"Total API Cost: ${costs.get('total_cost_usd', 0):.6f}")
        print()

    def print_verification_result(self, result: dict[str, Any]) -> None:
        """Print verification results."""
        print("✅ VERIFICATION RESULTS")
        print("-" * 70)

        if result["valid"]:
            print("Status: ✅ ALL VALIDATIONS PASSED")
        else:
            print("Status: ❌ VALIDATION FAILED")
            for label, key in ERROR_SECTIONS:
                if result.get(key):
                    print(f"\n{label}:")
                    for error in result[key]:
                        print(f"  - {error}")

        print(f"\nTotal Errors: {result.get('total_errors', 0)}")
        print()


CASES = (
    (
        "TEST 1: Grocery Price Comparison App (iOS)",
        (
            "we have to estimate a new feature for the mobile app in iOS only, "
            "than most include a new chat with an agent than support the grocery shopping "
            "carts creation base on the grocery shops around where you live as food is quite "
            "becoming expensive we most save money on food so we most copare the prices of "
            "the produces than i want to eat, so we most compare also the labels with the "
            "ingredientes to filter out the most natural food and organic so estimate this project"
        ),
        40,
    ),
    (
        "TEST 2: Simple Feature - User Dashboard",
        (
            "We need to build a user dashboard for our SaaS product. "
            "Features include: user profile management, analytics widgets showing "
            "key metrics, data export to CSV, and real-time notifications. "
            "We need both web and mobile responsive design."
        ),
        50,
    ),
)


def run_case(verifier: APIVerifier, title: str, transcription: str, hourly_rate: int) -> bool:
    """Run one estimation request through the verification pipeline."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    try:
        print("\n📤 Sending estimation request...")
        response = verifier.estimate_project(transcription, hourly_rate=hourly_rate)
        print("✅ Received response from API")
        verifier.print_estimation_summary(response["estimation"])
        verifier.print_token_costs(response)

        print("🔐 Running verification pipeline...")
        verification_result = verifier.verify_response(response)
        verifier.print_verification_result(verification_result)
        return bool(verification_result["valid"])
    except Exception as e:  # noqa: BLE001 - CLI boundary: report any failure as exit code 1
        print(f"❌ Error: {e!s}")
        return False


def main() -> int:
    """Main verification flow."""
    print("\n" + "=" * 70)
    print("CAG ESTIMATE - API VERIFICATION SCRIPT")
    print("=" * 70)

    verifier = APIVerifier()

    # Check API health
    print("\n🔍 Checking API health...")
    if not verifier.check_api_health():
        print(
            "❌ API is not running. Start with:\n"
            "  .venv/bin/python -m uvicorn cag_estimate.main:app --host 127.0.0.1 --port 8001"
        )
        return 1

    print("✅ API is running and healthy")

    for title, transcription, rate in CASES:
        if not run_case(verifier, title, transcription, rate):
            return 1

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED SUCCESSFULLY")
    print("=" * 70)
    print("\nThe CAG Estimate API is working correctly and producing valid estimations!")
    print("Managers can trust the pipeline to verify project estimates reliably.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
