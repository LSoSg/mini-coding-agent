"""CLI status and layered-verification output tests."""

import sys
from types import SimpleNamespace

import pytest

import main as cli
from agent import (
    AgentResult,
    AgentStatus,
    VerificationEvidence,
    VerificationTier,
)
from working_memory import WorkingMemory
from verifier import VerifierReview, VerifierVerdict


def result_for(status: AgentStatus) -> AgentResult:
    return AgentResult(
        status=status,
        final_answer="finished",
        verification_evidence=[],
        verification_level=None,
        plan_history=[],
        steps=1,
        messages=[],
        error=None if status is AgentStatus.COMPLETED else "not fully verified",
    )


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        (AgentStatus.COMPLETED, 0),
        (AgentStatus.FATAL_ERROR, 1),
        (AgentStatus.ORIGINAL_TESTS_FAILED, 2),
        (AgentStatus.PLAN_FAILED, 2),
        (AgentStatus.VERIFIER_FAILED, 2),
    ],
)
def test_cli_exit_codes(
    status: AgentStatus,
    expected_exit: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "task"])
    settings = SimpleNamespace(
        builder_model="qwen-plus", verifier_model="deepseek-v4-flash"
    )
    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(cli, "LLMClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "CodingAgent",
        lambda _client, **_kwargs: SimpleNamespace(
            run=lambda _task: result_for(status)
        ),
    )

    assert cli.main() == expected_exit


def test_cli_prints_original_test_output(capsys: pytest.CaptureFixture[str]) -> None:
    result = result_for(AgentStatus.ORIGINAL_TESTS_FAILED)
    result.verification_evidence = [
        VerificationEvidence(
            tier=VerificationTier.ORIGINAL,
            command="python -m pytest <original tests>",
            success=False,
            output="original assertion failed",
            error="exit status 1",
            step=3,
            workspace_revision=1,
        )
    ]

    cli.print_result(result)
    output = capsys.readouterr().out

    assert "[Verification level]" not in output
    assert "[ORIGINAL]" in output
    assert "original assertion failed" in output


def test_cli_prints_compact_working_memory(capsys: pytest.CaptureFixture[str]) -> None:
    result = result_for(AgentStatus.COMPLETED)
    memory = WorkingMemory.from_task(
        "Inspect a.py. Do not modify it.", verification_required=False
    )
    memory.record_read("a.py", purpose="Inspect the target")
    result.working_memory = memory

    cli.print_result(result)
    output = capsys.readouterr().out

    assert "[Working Memory]" in output
    assert "Files read:" in output
    assert "a.py (revision 0, reads 1)" in output
    assert "Do not modify it" in output


def test_cli_prints_independent_verifier_review(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = result_for(AgentStatus.COMPLETED)
    result.verifier_review = VerifierReview(
        verdict=VerifierVerdict.FAIL,
        summary="A boundary case fails.",
        requirement_checks=[],
        counterexamples=["Empty input"],
        unresolved_assumptions=["Input is never empty"],
    )

    cli.print_result(result)
    output = capsys.readouterr().out

    assert "[Independent Verifier Advice]" in output
    assert "Verdict: FAIL" in output
    assert "Empty input" in output
    assert "Input is never empty" in output
