"""Unit tests for the v0.4 structured planning protocol."""

import json
from typing import Any

import pytest

from planning import (
    MAX_PLAN_STEPS,
    PlanValidationError,
    PlanStep,
    match_plan_step,
    parse_plan,
)


def step(
    step_id: str = "step_1",
    tool: str = "read_file",
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "description": "Perform one necessary action",
        "tool": tool,
        "argument_constraints": constraints or {"path": "app.py"},
        "rationale": "This action directly supports the goal",
    }


def encoded_plan(steps: list[dict[str, Any]], **updates: Any) -> str:
    data = {
        "goal": "Understand or update the requested file",
        "success_criteria": ["The requested outcome is achieved"],
        "steps": steps,
    }
    data.update(updates)
    return json.dumps(data)


def test_valid_plan_is_parsed_with_revision() -> None:
    plan = parse_plan(
        encoded_plan([step()]), revision=3, verification_required=False
    )
    assert plan.revision == 3
    assert plan.steps[0].tool == "read_file"


def test_information_plan_may_have_no_tool_steps() -> None:
    plan = parse_plan(encoded_plan([]), revision=0, verification_required=False)
    assert plan.steps == []


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"goal": "x", "success_criteria": ["x"]}),
        encoded_plan([step(tool="unknown_tool")]),
        encoded_plan([step("same"), step("same")]),
        encoded_plan([step(constraints={"path": "app.py", "surprise": True})]),
        encoded_plan([step(constraints={"path": "../outside.py"})]),
        encoded_plan([step(tool="write_file", constraints={"path": "a.py", "content": "x"})]),
        encoded_plan(
            [step(str(index), constraints={"path": f"{index}.py"})
             for index in range(MAX_PLAN_STEPS + 1)]
        ),
    ],
)
def test_invalid_plans_are_rejected(raw: str) -> None:
    with pytest.raises(PlanValidationError):
        parse_plan(raw, revision=0, verification_required=False)


def test_change_plan_requires_verification_after_last_write() -> None:
    missing = encoded_plan([
        step("write", "write_file", {"path": "a.py"}),
    ])
    wrong_order = encoded_plan([
        step("test", "execute_command", {"command": ["python", "-m", "pytest"]}),
        step("write", "write_file", {"path": "a.py"}),
    ])
    for raw in (missing, wrong_order):
        with pytest.raises(PlanValidationError):
            parse_plan(raw, revision=0, verification_required=True)


def test_only_allowlisted_commands_may_be_planned() -> None:
    raw = encoded_plan([
        step("command", "execute_command", {"command": ["pytest", "-v"]})
    ])
    with pytest.raises(PlanValidationError, match="allowlist"):
        parse_plan(raw, revision=0, verification_required=False)


def test_plan_matcher_normalizes_defaults_and_paths() -> None:
    planned = PlanStep(
        id="search",
        description="Search the project",
        tool="search_files",
        argument_constraints={"keyword": "needle", "path": "src\\pkg"},
        rationale="Find the relevant definition",
    )
    matched, error = match_plan_step(
        planned,
        "search_files",
        {"keyword": "needle", "path": "src/pkg", "max_results": 20},
    )
    assert matched
    assert error is None


def test_plan_matcher_ignores_write_content_but_enforces_path() -> None:
    planned = PlanStep(
        id="write",
        description="Create target",
        tool="write_file",
        argument_constraints={"path": "target.py"},
        rationale="Create the requested file",
    )
    assert match_plan_step(
        planned, "write_file", {"path": "target.py", "content": "print('ok')"}
    )[0]
    matched, error = match_plan_step(
        planned, "write_file", {"path": "other.py", "content": "x"}
    )
    assert not matched
    assert "path" in (error or "")
