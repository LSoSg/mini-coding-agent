"""Structured planning protocol and plan-to-tool-call validation."""

import inspect
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from tools import TOOL_REGISTRY


MAX_PLANNING_ATTEMPTS = 2
MAX_REPLANS = 2
MAX_PLAN_STEPS = 12
MAX_PLAN_COMPLETION_REMINDERS = 3

PLAN_TOP_LEVEL_FIELDS = {"goal", "success_criteria", "steps"}
PLAN_STEP_FIELDS = {
    "id",
    "description",
    "tool",
    "argument_constraints",
    "rationale",
}
PATH_ARGUMENTS = {
    "list_files": ("path",),
    "search_files": ("path",),
    "read_file": ("path",),
    "write_file": ("path",),
}
REQUIRED_PLAN_ARGUMENTS = {
    "list_files": {"path"},
    "search_files": {"keyword", "path"},
    "read_file": {"path"},
    "write_file": {"path"},
    "execute_command": {"command"},
}

PLANNING_PROMPT = """Create a minimal executable plan before using any local tools.
Return JSON only, with exactly this shape:
{
  "goal": "non-empty goal",
  "success_criteria": ["non-empty criterion"],
  "steps": [
    {
      "id": "unique_step_id",
      "description": "one atomic tool action",
      "tool": "one registered tool name",
      "argument_constraints": {"only stable arguments used to match the call"},
      "rationale": "why this action is necessary for the user goal"
    }
  ]
}

Rules:
- Use the fewest relevant files and actions needed for the task.
- For greenfield creation, do not read unrelated existing source files merely to gather context.
- Each step must correspond to exactly one tool call and must be ordered for execution.
- list_files requires path; search_files requires keyword and path; read_file and write_file require path; execute_command requires command.
- For write_file, constrain path only; never include the full content in the plan.
- Commands must match exactly one supported command: ["pytest"], ["python", "-m", "pytest"], ["python", "<workspace script.py>"], ["git", "status"], or ["git", "diff"].
- For pytest, the command MUST be exactly ["pytest"] or ["python", "-m", "pytest"]. Never append a test filename, -q, -v, or any other argument.
- Code creation or modification plans must include a verification command after the last write.
- Do not wrap the JSON in Markdown fences and do not include commentary outside JSON."""


class PlanStepStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"


@dataclass
class PlanStep:
    id: str
    description: str
    tool: str
    argument_constraints: dict[str, Any]
    rationale: str
    status: PlanStepStatus = PlanStepStatus.PENDING


@dataclass
class TaskPlan:
    goal: str
    success_criteria: list[str]
    steps: list[PlanStep]
    revision: int


class PlanValidationError(ValueError):
    """Raised when a model-generated plan violates the planning protocol."""


def is_verification_command(command: Any) -> bool:
    if not isinstance(command, list) or any(
        not isinstance(argument, str) for argument in command
    ):
        return False
    if command in (["pytest"], ["python", "-m", "pytest"]):
        return True
    return (
        len(command) == 2
        and command[0] == "python"
        and _is_safe_relative_path(command[1])
        and command[1].casefold().endswith(".py")
    )


def is_allowed_command(command: Any) -> bool:
    return is_verification_command(command) or command in (
        ["git", "status"],
        ["git", "diff"],
    )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_safe_relative_path(value: Any) -> bool:
    if not _is_non_empty_string(value):
        return False
    path = Path(value)
    return not path.is_absolute() and not path.drive and ".." not in path.parts


def _validate_constraints(tool: str, constraints: Any) -> dict[str, Any]:
    if not isinstance(constraints, dict):
        raise PlanValidationError("argument_constraints must be an object.")

    signature = inspect.signature(TOOL_REGISTRY[tool])
    unknown = set(constraints) - set(signature.parameters)
    if unknown:
        raise PlanValidationError(
            f"Plan step for '{tool}' has unknown arguments: {sorted(unknown)}."
        )
    missing = REQUIRED_PLAN_ARGUMENTS[tool] - set(constraints)
    if missing:
        raise PlanValidationError(
            f"Plan step for '{tool}' is missing constraints: {sorted(missing)}."
        )
    if tool == "write_file" and "content" in constraints:
        raise PlanValidationError("write_file plans must not include content.")

    for argument in PATH_ARGUMENTS.get(tool, ()):
        if not _is_safe_relative_path(constraints.get(argument)):
            raise PlanValidationError(
                f"Plan step for '{tool}' has an unsafe or invalid {argument}."
            )
    if tool == "search_files" and not _is_non_empty_string(
        constraints.get("keyword")
    ):
        raise PlanValidationError("search_files keyword must be non-empty.")
    if tool == "search_files" and "max_results" in constraints:
        maximum = constraints["max_results"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 50:
            raise PlanValidationError("search_files max_results must be from 1 to 50.")
    if tool == "execute_command" and not is_allowed_command(
        constraints.get("command")
    ):
        raise PlanValidationError(
            "execute_command is not in the supported allowlist. Use exactly "
            '["pytest"], ["python", "-m", "pytest"], '
            '["python", "<workspace script.py>"], ["git", "status"], or '
            '["git", "diff"]. Do not append test paths or pytest flags.'
        )
    return dict(constraints)


def parse_plan(
    raw_plan: str,
    *,
    revision: int,
    verification_required: bool,
    max_steps: int = MAX_PLAN_STEPS,
) -> TaskPlan:
    if not isinstance(raw_plan, str):
        raise PlanValidationError("Plan response must be a JSON string.")
    try:
        data = json.loads(raw_plan)
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"Plan is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanValidationError("Plan JSON must be an object.")
    if set(data) != PLAN_TOP_LEVEL_FIELDS:
        raise PlanValidationError(
            f"Plan fields must be exactly {sorted(PLAN_TOP_LEVEL_FIELDS)}."
        )
    if not _is_non_empty_string(data["goal"]):
        raise PlanValidationError("Plan goal must be a non-empty string.")

    criteria = data["success_criteria"]
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(not _is_non_empty_string(item) for item in criteria)
    ):
        raise PlanValidationError(
            "success_criteria must be a non-empty list of non-empty strings."
        )

    raw_steps = data["steps"]
    if not isinstance(raw_steps, list):
        raise PlanValidationError("Plan steps must be a list.")
    if len(raw_steps) > max_steps:
        raise PlanValidationError(f"Plan cannot exceed {max_steps} steps.")

    steps: list[PlanStep] = []
    seen_ids: set[str] = set()
    for position, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict) or set(raw_step) != PLAN_STEP_FIELDS:
            raise PlanValidationError(
                f"Plan step {position} fields must be exactly {sorted(PLAN_STEP_FIELDS)}."
            )
        step_id = raw_step["id"]
        if not _is_non_empty_string(step_id):
            raise PlanValidationError(f"Plan step {position} id must be non-empty.")
        if step_id in seen_ids:
            raise PlanValidationError(f"Duplicate plan step id: {step_id}.")
        seen_ids.add(step_id)

        tool = raw_step["tool"]
        if not isinstance(tool, str) or tool not in TOOL_REGISTRY:
            raise PlanValidationError(f"Unknown planned tool: {tool}.")
        if not _is_non_empty_string(raw_step["description"]):
            raise PlanValidationError(f"Plan step {step_id} needs a description.")
        if not _is_non_empty_string(raw_step["rationale"]):
            raise PlanValidationError(f"Plan step {step_id} needs a rationale.")
        constraints = _validate_constraints(tool, raw_step["argument_constraints"])
        steps.append(
            PlanStep(
                id=step_id,
                description=raw_step["description"].strip(),
                tool=tool,
                argument_constraints=constraints,
                rationale=raw_step["rationale"].strip(),
            )
        )

    verification_indexes = [
        index
        for index, step in enumerate(steps)
        if step.tool == "execute_command"
        and is_verification_command(step.argument_constraints.get("command"))
    ]
    write_indexes = [
        index for index, step in enumerate(steps) if step.tool == "write_file"
    ]
    if verification_required or write_indexes:
        if not verification_indexes:
            raise PlanValidationError(
                "This task requires a planned verification command."
            )
        if write_indexes and max(verification_indexes) < max(write_indexes):
            raise PlanValidationError(
                "Verification must be planned after the final write_file step."
            )

    return TaskPlan(
        goal=data["goal"].strip(),
        success_criteria=[item.strip() for item in criteria],
        steps=steps,
        revision=revision,
    )


def _canonical_argument(name: str, value: Any) -> Any:
    if name == "path" and isinstance(value, str):
        return Path(value).as_posix()
    return value


def normalized_tool_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(TOOL_REGISTRY[tool])
    try:
        bound = signature.bind_partial(**arguments)
    except TypeError:
        return dict(arguments)
    bound.apply_defaults()
    return dict(bound.arguments)


def match_plan_step(
    step: PlanStep,
    tool: str,
    arguments: dict[str, Any],
) -> tuple[bool, str | None]:
    if tool != step.tool:
        return False, f"expected tool '{step.tool}', received '{tool}'."
    actual = normalized_tool_arguments(tool, arguments)
    for name, expected in step.argument_constraints.items():
        if name not in actual:
            return False, f"missing planned argument '{name}'."
        if _canonical_argument(name, actual[name]) != _canonical_argument(name, expected):
            return False, (
                f"argument '{name}' does not match the current plan step "
                f"({actual[name]!r} != {expected!r})."
            )
    return True, None


def plan_as_dict(plan: TaskPlan) -> dict[str, Any]:
    return asdict(plan)


def plan_as_json(plan: TaskPlan) -> str:
    return json.dumps(plan_as_dict(plan), ensure_ascii=False)
