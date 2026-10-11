"""Offline tests for the evals CLI: dry run, exit codes, output file and cost banner."""

import json
from pathlib import Path

import pytest

from cag_estimate.evals.dataset import load_dataset
from cag_estimate.evals.run import main
from tests.eval_fakes import ScriptedClient, ok_reply, rejection_reply

GUARDRAIL = "guardrail-prompt-injection"


def test_dry_run_lists_cases_and_counts_calls_without_sending_anything(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = ScriptedClient()

    code = main(["--dry-run"], client=client)

    out = capsys.readouterr().out
    cases = load_dataset()
    calls = sum(len(case.turns) for case in cases)
    assert code == 0 and client.calls == []
    assert all(case.id in out for case in cases)
    assert f"{calls} estimate calls" in out


def test_dry_run_in_acb_mode_reports_the_extra_critic_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = ScriptedClient()

    code = main(["--dry-run", "--mode", "acb", "--case", "memory-bookstore-growth"], client=client)

    out = capsys.readouterr().out
    assert code == 0 and client.calls == []
    assert "3 estimate calls" in out and "critic" in out and "6" in out


def test_limit_and_case_select_the_cases(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--dry-run", "--limit", "2"], client=ScriptedClient())
    limited = capsys.readouterr().out
    main(
        ["--dry-run", "--case", GUARDRAIL, "--case", "guardrail-pii-email"], client=ScriptedClient()
    )
    chosen = capsys.readouterr().out

    first_two = [case.id for case in load_dataset()[:2]]
    assert all(case_id in limited for case_id in first_two) and "guardrail-pii-email" not in limited
    assert (
        GUARDRAIL in chosen and "guardrail-pii-email" in chosen and "single-bookstore" not in chosen
    )


@pytest.mark.parametrize(
    "argv",
    [["--case", "no-such-case"], ["--limit", "0"], ["--mode", "turbo"]],
    ids=["unknown_case", "zero_limit", "bad_mode"],
)
def test_usage_errors_exit_with_2_before_sending_anything(argv: list[str]) -> None:
    client = ScriptedClient()
    with pytest.raises(SystemExit) as exit_info:
        main([*argv, "--dry-run"], client=client)
    assert exit_info.value.code == 2 and client.calls == []


def test_an_unhealthy_api_exits_with_2_and_sends_no_estimate() -> None:
    client = ScriptedClient(health_status=503)
    assert main(["--case", GUARDRAIL], client=client) == 2
    assert client.urls("POST") == []


def test_an_unreachable_api_exits_with_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--case", GUARDRAIL, "--base-url", "http://127.0.0.1:1"])
    assert code == 2
    assert "cannot reach" in capsys.readouterr().out.lower()


def test_exit_0_when_every_case_passes_and_a_cost_banner_comes_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = ScriptedClient([rejection_reply("prompt_injection")])

    code = main(["--case", GUARDRAIL, "--base-url", "http://api.test"], client=client)

    out = capsys.readouterr().out
    assert code == 0
    assert out.index("real LLM") < out.index(GUARDRAIL, out.index("real LLM"))
    assert "1 cases, 1 estimate calls" in out and "http://api.test" in out
    assert "PASS" in out and "FAIL" not in out and "1/1" in out


def test_exit_1_when_a_case_fails(capsys: pytest.CaptureFixture[str]) -> None:
    client = ScriptedClient([ok_reply()])

    code = main(["--case", GUARDRAIL], client=client)

    out = capsys.readouterr().out
    assert code == 1 and "FAIL" in out and "0/1" in out


def test_exit_1_when_a_case_errors(capsys: pytest.CaptureFixture[str]) -> None:
    client = ScriptedClient([ConnectionError("refused")])
    assert main(["--case", GUARDRAIL], client=client) == 1
    assert "ERROR" in capsys.readouterr().out


def test_output_writes_the_json_report(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    client = ScriptedClient([rejection_reply("prompt_injection")])

    code = main(["--case", GUARDRAIL, "--output", str(target)], client=client)

    data = json.loads(target.read_text(encoding="utf-8"))
    assert code == 0
    assert {"mode", "base_url", "total_cases", "passed_cases", "failed_cases"} <= set(data)
    assert {"metric_pass_counts", "average_turn_latency_ms", "cases"} <= set(data)
    assert data["cases"][0]["id"] == GUARDRAIL and data["passed_cases"] == 1


def test_acb_mode_calls_the_reviewed_endpoint() -> None:
    client = ScriptedClient([rejection_reply("prompt_injection")])
    main(["--case", GUARDRAIL, "--mode", "acb"], client=client)
    assert client.urls("POST")[1].endswith("/estimate-acb")
