"""Plan-constrained coding-agent loop for v0.4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

import tools as tool_module

from planning import (
    MAX_PLAN_COMPLETION_REMINDERS,
    MAX_PLANNING_ATTEMPTS,
    MAX_REPLANS,
    PLANNING_PROMPT,
    PlanStepStatus,
    PlanValidationError,
    TaskPlan,
    is_verification_command,
    match_plan_step,
    parse_plan,
    plan_as_json,
)
from tool_schemas import TOOLS
from tools import ToolResult, execute_tool
from workspace_snapshot import WorkspaceSnapshot

MAX_STEPS = 20
MAX_VERIFICATION_REQUESTS = 3
MAX_TOOL_LOG_ARGUMENT_CHARS = 240

SYSTEM_PROMPT = """You are a coding agent operating inside a controlled workspace.
An accepted structured plan is a hard execution contract. Follow its pending steps in
order, use only the tool named by the current step, and obey every argument constraint.
Do not inspect unrelated files or read every file merely to establish context. Each tool
call must be an atomic action tied to the user's goal. If the plan no longer fits, do not
improvise: a plan deviation will trigger a bounded replanning phase.

Use only the supplied tools for local actions. Never claim that a file was changed or a
command passed without a successful tool result. For tasks that create or modify files,
finish with a real verification command such as `python -m pytest` or an allowed Python
script. Return a concise final answer only after the accepted plan is complete.
"""


class LLMChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Any: ...


class AgentStatus(str, Enum):
    COMPLETED = "COMPLETED"
    SELF_VERIFIED = "SELF_VERIFIED"
    ORIGINAL_TESTS_FAILED = "ORIGINAL_TESTS_FAILED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    FATAL_ERROR = "FATAL_ERROR"
    PLAN_FAILED = "PLAN_FAILED"


class VerificationTier(str, Enum):
    SELF = "SELF"
    ORIGINAL = "ORIGINAL"


@dataclass
class VerificationEvidence:
    tier: VerificationTier
    command: str
    success: bool
    output: str
    error: str | None
    step: int
    workspace_revision: int


@dataclass
class AgentResult:
    status: AgentStatus
    final_answer: str
    verification_evidence: list[VerificationEvidence]
    verification_level: VerificationTier | None
    plan_history: list[TaskPlan]
    steps: int
    messages: list[dict[str, Any]]
    error: str | None = None


def task_requires_verification(task: str) -> bool:
    """Conservatively identify tasks likely to change the workspace."""

    lowered = task.lower()
    change_terms = (
        "create", "write", "modify", "update", "edit", "fix", "implement",
        "add", "delete", "remove", "refactor", "生成", "创建", "编写",
        "写一个", "修改", "更新", "修复", "实现", "增加", "添加", "删除", "重构",
    )
    return any(term in lowered for term in change_terms)


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        result = dict(message)
    else:
        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(exclude_none=True)
            if not isinstance(dumped, dict):
                raise ValueError("Assistant message model_dump() did not return an object.")
            result = dumped
        else:
            tool_calls = getattr(message, "tool_calls", None)
            result = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
            }
            if tool_calls:
                result["tool_calls"] = [
                    call.model_dump(exclude_none=True)
                    if callable(getattr(call, "model_dump", None)) else call
                    for call in tool_calls
                ]

    if result.get("role", "assistant") != "assistant":
        raise ValueError("Model response role must be 'assistant'.")
    tool_calls = result.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, (list, tuple)):
        raise ValueError("Assistant tool_calls must be a list.")
    for tool_call in tool_calls or []:
        call_id, name, _arguments = _function_call_parts(tool_call)
        if not call_id or not name:
            raise ValueError("Each tool call must include a non-empty id and function name.")
    return result


def _function_call_parts(tool_call: Any) -> tuple[str, str, Any]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        return (
            str(tool_call.get("id", "")),
            str(function.get("name", "")),
            function.get("arguments", "{}"),
        )
    function = getattr(tool_call, "function", None)
    return (
        str(getattr(tool_call, "id", "")),
        str(getattr(function, "name", "")),
        getattr(function, "arguments", "{}"),
    )


def _parse_arguments(raw_arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw_arguments, dict):
        return raw_arguments, None
    if not isinstance(raw_arguments, str):
        return None, "Tool arguments must be a JSON object."
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON arguments: {exc.msg}."
    if not isinstance(arguments, dict):
        return None, "Tool arguments must decode to a JSON object."
    return arguments, None


def _tool_result_payload(result: ToolResult) -> str:
    return json.dumps(
        {"success": result.success, "output": result.output, "error": result.error},
        ensure_ascii=False,
    )


def _verification_failure_reason(command: list[str], result: ToolResult) -> str:
    """Build actionable replanning context from a failed verification result."""

    command_text = " ".join(command)
    diagnostics = f"{result.error or ''}\n{result.output or ''}".casefold()
    is_pytest = command in (["pytest"], ["python", "-m", "pytest"])
    no_tests_collected = any(
        marker in diagnostics
        for marker in (
            "exit status 5",
            "non-zero status 5",
            "exit_code: 5",
            "collected 0 items",
            "no tests collected",
            "no tests ran",
        )
    )
    if is_pytest and no_tests_collected:
        return (
            f"The planned verification command {command_text} failed because pytest "
            "collected no tests (exit code 5). Before running pytest again, create a "
            "separate discoverable test module whose filename matches `test_*.py` "
            "(for example, `test_floyd.py`). Merely adding `test_*` functions to a "
            "normal source module such as `floyd.py` will not make pytest collect it."
        )

    return (
        f"The planned verification command {command_text} failed: "
        f"{result.error or 'see the tool result for diagnostics'}. Replace that failed "
        "pending step with a plan that performs any necessary diagnosis or file repairs "
        "before running verification again."
    )


def _safe_tool_call_summary(name: str, arguments: dict[str, Any]) -> str:
    safe_arguments = dict(arguments)
    if name == "write_file" and "content" in safe_arguments:
        content = safe_arguments["content"]
        safe_arguments["content"] = (
            f"<{len(content) if isinstance(content, str) else '?'} characters>"
        )
    rendered = json.dumps(safe_arguments, ensure_ascii=False)
    if len(rendered) > MAX_TOOL_LOG_ARGUMENT_CHARS:
        rendered = rendered[:MAX_TOOL_LOG_ARGUMENT_CHARS] + "..."
    return rendered


class CodingAgent:
    """Run mandatory planning followed by a plan-constrained tool loop."""

    def __init__(
        self,
        llm_client: LLMChatClient,
        *,
        tool_executor: Callable[[str, dict[str, Any]], ToolResult] = execute_tool,
        max_steps: int = MAX_STEPS,
        max_verification_requests: int = MAX_VERIFICATION_REQUESTS,
        max_planning_attempts: int = MAX_PLANNING_ATTEMPTS,
        max_replans: int = MAX_REPLANS,
        max_plan_completion_reminders: int = MAX_PLAN_COMPLETION_REMINDERS,
        snapshot_factory: Callable[[], WorkspaceSnapshot] | None = None,
        verbose: bool = True,
        output: Callable[[str], None] = print,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if max_verification_requests < 0:
            raise ValueError("max_verification_requests cannot be negative.")
        if max_planning_attempts < 1:
            raise ValueError("max_planning_attempts must be at least 1.")
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative.")
        if max_plan_completion_reminders < 0:
            raise ValueError("max_plan_completion_reminders cannot be negative.")
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.max_verification_requests = max_verification_requests
        self.max_planning_attempts = max_planning_attempts
        self.max_replans = max_replans
        self.max_plan_completion_reminders = max_plan_completion_reminders
        self.snapshot_factory = snapshot_factory or (
            lambda: WorkspaceSnapshot(tool_module.WORKSPACE_ROOT)
        )
        self.verbose = verbose
        self.output = output
        self.logger = logger or (output if verbose else lambda _message: None)

    def _result(
        self,
        status: AgentStatus,
        final_answer: str,
        evidence: list[VerificationEvidence],
        plans: list[TaskPlan],
        steps: int,
        messages: list[dict[str, Any]],
        error: str | None = None,
    ) -> AgentResult:
        verification_level: VerificationTier | None = None
        if any(
            item.success and item.tier is VerificationTier.ORIGINAL
            for item in evidence
        ):
            verification_level = VerificationTier.ORIGINAL
        elif any(
            item.success and item.tier is VerificationTier.SELF
            for item in evidence
        ):
            verification_level = VerificationTier.SELF
        return AgentResult(
            status=status,
            final_answer=final_answer,
            verification_evidence=evidence,
            verification_level=verification_level,
            plan_history=plans,
            steps=steps,
            messages=messages,
            error=error,
        )

    def _fatal(self, reason: str, evidence: list[VerificationEvidence], plans: list[TaskPlan],
               steps: int, messages: list[dict[str, Any]]) -> AgentResult:
        self.logger("[Status] FATAL_ERROR")
        return self._result(
            AgentStatus.FATAL_ERROR,
            "Agent stopped because of a fatal error.",
            evidence, plans, steps, messages, reason,
        )

    def _plan_failed(self, reason: str, evidence: list[VerificationEvidence],
                     plans: list[TaskPlan], steps: int,
                     messages: list[dict[str, Any]]) -> AgentResult:
        self.logger("[Status] PLAN_FAILED")
        return self._result(
            AgentStatus.PLAN_FAILED,
            "Agent stopped because it could not obtain or follow a valid task plan.",
            evidence, plans, steps, messages, reason,
        )

    def _log_plan(self, plan: TaskPlan, *, replan: bool = False) -> None:
        label = "Replan" if replan else "Plan"
        self.logger(f"[{label} revision {plan.revision}]")
        if not plan.steps:
            self.logger("(no local tool steps)")
        for index, step in enumerate(plan.steps, start=1):
            constraints = json.dumps(step.argument_constraints, ensure_ascii=False)
            self.logger(f"{index}. {step.tool} {constraints}")

    def _request_plan(
        self,
        messages: list[dict[str, Any]],
        *,
        revision: int,
        verification_required: bool,
    ) -> tuple[TaskPlan | None, str | None, str | None]:
        last_error = "The model did not return a valid plan."
        for attempt in range(1, self.max_planning_attempts + 1):
            try:
                raw_plan = self.llm_client.chat(messages=messages)
            except Exception as exc:
                return None, None, f"Planning API request failed: {exc}"
            messages.append({"role": "assistant", "content": raw_plan})
            try:
                plan = parse_plan(
                    raw_plan,
                    revision=revision,
                    verification_required=verification_required,
                )
            except PlanValidationError as exc:
                last_error = str(exc)
                self.logger(f"[Planning attempt {attempt}] invalid plan: {last_error}")
                if attempt < self.max_planning_attempts:
                    messages.append({
                        "role": "system",
                        "content": (
                            f"The proposed plan is invalid: {last_error} "
                            "Return a corrected strict JSON plan only."
                        ),
                    })
                continue
            return plan, None, None
        return None, last_error, None

    def _replan(
        self,
        *,
        task: str,
        current_plan: TaskPlan,
        reason: str,
        revision: int,
        verification_required: bool,
        messages: list[dict[str, Any]],
    ) -> tuple[TaskPlan | None, str | None, str | None]:
        messages.append({
            "role": "system",
            "content": (
                f"{PLANNING_PROMPT}\n\nREPLANNING CONTEXT\n"
                f"Original task: {task}\n"
                f"Previous plan and completed statuses:\n{plan_as_json(current_plan)}\n"
                f"Deviation reason: {reason}\n"
                "Return a complete plan for the remaining work only. Do not repeat completed "
                "steps. Preserve the original goal and use only necessary local actions. "
                "If a tool or verification command failed, do not blindly repeat the failed "
                "step before the actions needed to diagnose or repair it."
            ),
        })
        return self._request_plan(
            messages, revision=revision, verification_required=verification_required
        )

    def run(self, task: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        evidence: list[VerificationEvidence] = []
        plan_history: list[TaskPlan] = []
        if not isinstance(task, str) or not task.strip():
            return self._fatal(
                "Task must be a non-empty string.", evidence, plan_history, 0, messages
            )

        self.logger("[Workspace snapshot]")
        try:
            snapshot = self.snapshot_factory()
            snapshot.capture()
        except Exception as exc:
            return self._fatal(
                f"Workspace snapshot failed: {exc}",
                evidence,
                plan_history,
                0,
                messages,
            )
        self.logger(
            "[Original test discovery] "
            f"available={snapshot.discovery.available} "
            f"files={len(snapshot.discovery.test_files)}"
        )
        try:
            return self._run_with_snapshot(task, snapshot)
        finally:
            snapshot.cleanup()

    def _complete_with_original_tests(
        self,
        *,
        final_answer: str,
        snapshot: WorkspaceSnapshot,
        evidence: list[VerificationEvidence],
        plans: list[TaskPlan],
        step_number: int,
        workspace_revision: int,
        messages: list[dict[str, Any]],
        external_verification_required: bool,
    ) -> AgentResult:
        if not external_verification_required:
            self.logger("[Status] COMPLETED")
            return self._result(
                AgentStatus.COMPLETED,
                final_answer.strip(),
                evidence,
                plans,
                step_number,
                messages,
            )

        discovery = snapshot.discovery
        if not discovery.has_tests:
            reason = discovery.error or (
                "The initial workspace snapshot contained no collectable original tests."
            )
            self.logger("[Status] SELF_VERIFIED")
            return self._result(
                AgentStatus.SELF_VERIFIED,
                final_answer.strip(),
                evidence,
                plans,
                step_number,
                messages,
                reason,
            )

        self.logger("[Original test regression]")
        try:
            original_result = snapshot.run_original_tests()
        except Exception as exc:
            return self._fatal(
                str(exc), evidence, plans, step_number, messages
            )

        evidence.append(
            VerificationEvidence(
                tier=VerificationTier.ORIGINAL,
                command=original_result.command,
                success=original_result.success,
                output=original_result.output,
                error=original_result.error,
                step=step_number,
                workspace_revision=workspace_revision,
            )
        )
        self.logger(f"[Original tests] success={original_result.success}")
        if not original_result.success:
            self.logger("[Status] ORIGINAL_TESTS_FAILED")
            return self._result(
                AgentStatus.ORIGINAL_TESTS_FAILED,
                final_answer.strip(),
                evidence,
                plans,
                step_number,
                messages,
                original_result.error or "Original tests failed.",
            )

        self.logger("[Status] COMPLETED")
        return self._result(
            AgentStatus.COMPLETED,
            final_answer.strip(),
            evidence,
            plans,
            step_number,
            messages,
        )

    def _run_with_snapshot(
        self, task: str, snapshot: WorkspaceSnapshot
    ) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        evidence: list[VerificationEvidence] = []
        plan_history: list[TaskPlan] = []
        if not isinstance(task, str) or not task.strip():
            return self._fatal(
                "Task must be a non-empty string.", evidence, plan_history, 0, messages
            )
        verification_required = task_requires_verification(task)
        workspace_revision = 0
        last_verified_revision: int | None = None
        verification_requests = 0
        plan_completion_reminders = 0
        replans = 0

        self.logger("[Planning inventory]")
        try:
            inventory = self.tool_executor("list_files", {"path": "."})
        except Exception as exc:
            inventory = ToolResult(False, error=f"Workspace inventory failed: {exc}")
        if not isinstance(inventory, ToolResult):
            inventory = ToolResult(False, error="Workspace inventory returned an invalid result.")
        self.logger(f"[Tool result] success={inventory.success}")
        if not inventory.success:
            return self._plan_failed(
                inventory.error or "Unable to inspect the controlled workspace root.",
                evidence, plan_history, 0, messages,
            )

        messages.extend([
            {
                "role": "system",
                "content": (
                    f"{PLANNING_PROMPT}\n\n"
                    f"Task requires verification: {verification_required}.\n"
                    "Controlled workspace root inventory (names and types only):\n"
                    f"{inventory.output}"
                ),
            },
            {"role": "user", "content": task},
        ])
        current_plan, planning_error, fatal_error = self._request_plan(
            messages, revision=0, verification_required=verification_required
        )
        if fatal_error:
            return self._fatal(fatal_error, evidence, plan_history, 0, messages)
        if current_plan is None:
            return self._plan_failed(
                planning_error or "Unable to obtain a valid initial plan.",
                evidence, plan_history, 0, messages,
            )
        plan_history.append(current_plan)
        self._log_plan(current_plan)
        messages.append({
            "role": "system",
            "content": (
                "The following plan is accepted and mandatory. Execute pending steps in order:\n"
                f"{plan_as_json(current_plan)}"
            ),
        })
        current_step_index = 0

        for step_number in range(1, self.max_steps + 1):
            self.logger(f"[Step {step_number}]")
            try:
                raw_message = self.llm_client.chat(messages=messages, tools=TOOLS)
                assistant_message = _assistant_message_to_dict(raw_message)
            except Exception as exc:
                return self._fatal(
                    f"LLM request failed: {exc}", evidence, plan_history,
                    step_number, messages,
                )
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []

            if not tool_calls:
                final_answer = assistant_message.get("content")
                if not isinstance(final_answer, str) or not final_answer.strip():
                    return self._fatal(
                        "Model returned neither tool calls nor usable final text.",
                        evidence, plan_history, step_number, messages,
                    )
                if current_step_index < len(current_plan.steps):
                    plan_completion_reminders += 1
                    if plan_completion_reminders > self.max_plan_completion_reminders:
                        return self._plan_failed(
                            "The model repeatedly returned a final answer before completing the plan.",
                            evidence, plan_history, step_number, messages,
                        )
                    pending = current_plan.steps[current_step_index]
                    self.logger(
                        f"[Plan incomplete] pending={pending.id} reminder="
                        f"{plan_completion_reminders}/{self.max_plan_completion_reminders}"
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "Do not answer yet. Continue the accepted plan. The next pending step "
                            f"is {pending.id}: {pending.description}, using {pending.tool}."
                        ),
                    })
                    continue

                verified_current_revision = last_verified_revision == workspace_revision
                if verification_required and not verified_current_revision:
                    verification_requests += 1
                    if verification_requests > self.max_verification_requests:
                        self.logger("[Status] VERIFICATION_REQUIRED")
                        return self._result(
                            AgentStatus.VERIFICATION_REQUIRED, final_answer, evidence,
                            plan_history, step_number, messages,
                            "The current workspace revision has no successful verification evidence.",
                        )
                    messages.append({
                        "role": "system",
                        "content": (
                            "The plan is complete, but the current workspace revision still requires "
                            "successful verification. Use an allowed verification command."
                        ),
                    })
                    continue

                return self._complete_with_original_tests(
                    final_answer=final_answer,
                    snapshot=snapshot,
                    evidence=evidence,
                    plans=plan_history,
                    step_number=step_number,
                    workspace_revision=workspace_revision,
                    messages=messages,
                    external_verification_required=(
                        verification_required
                        or any(item.tier is VerificationTier.SELF for item in evidence)
                    ),
                )

            parsed_calls: list[tuple[str, str, dict[str, Any] | None, str | None]] = []
            for tool_call in tool_calls:
                call_id, name, raw_arguments = _function_call_parts(tool_call)
                arguments, argument_error = _parse_arguments(raw_arguments)
                parsed_calls.append((call_id, name, arguments, argument_error))

            if any(item[3] is not None for item in parsed_calls):
                for call_id, name, _arguments, argument_error in parsed_calls:
                    error = argument_error or (
                        "Tool batch was not executed because another call had invalid arguments."
                    )
                    messages.append({
                        "role": "tool", "tool_call_id": call_id, "name": name,
                        "content": _tool_result_payload(ToolResult(False, error=error)),
                    })
                continue

            remaining = current_plan.steps[current_step_index:]
            deviations: list[str] = []
            if len(parsed_calls) > len(remaining):
                deviations.append("The batch contains more calls than remaining plan steps.")
            else:
                for offset, (_call_id, name, arguments, _error) in enumerate(parsed_calls):
                    matched, match_error = match_plan_step(
                        remaining[offset], name, arguments or {}
                    )
                    if not matched and match_error:
                        deviations.append(match_error)

            if deviations:
                reason = " ".join(deviations)
                self.logger(f"[Plan deviation] {reason}")
                for call_id, name, _arguments, _error in parsed_calls:
                    messages.append({
                        "role": "tool", "tool_call_id": call_id, "name": name,
                        "content": _tool_result_payload(
                            ToolResult(False, error=f"Plan deviation: {reason}")
                        ),
                    })
                if replans >= self.max_replans:
                    return self._plan_failed(
                        "The maximum number of replans was exceeded.", evidence,
                        plan_history, step_number, messages,
                    )
                replans += 1
                new_plan, planning_error, fatal_error = self._replan(
                    task=task, current_plan=current_plan, reason=reason,
                    revision=replans, verification_required=verification_required,
                    messages=messages,
                )
                if fatal_error:
                    return self._fatal(
                        fatal_error, evidence, plan_history, step_number, messages
                    )
                if new_plan is None:
                    return self._plan_failed(
                        planning_error or "Unable to obtain a valid revised plan.",
                        evidence, plan_history, step_number, messages,
                    )
                current_plan = new_plan
                plan_history.append(current_plan)
                current_step_index = 0
                plan_completion_reminders = 0
                self._log_plan(current_plan, replan=True)
                messages.append({
                    "role": "system",
                    "content": (
                        "The revised plan is accepted and mandatory. Execute pending steps in order:\n"
                        f"{plan_as_json(current_plan)}"
                    ),
                })
                continue

            batch_blocked = False
            failed_verification_reason: str | None = None
            for offset, (call_id, name, arguments, _error) in enumerate(parsed_calls):
                planned_step = remaining[offset]
                call_was_executed = not batch_blocked
                if batch_blocked:
                    result = ToolResult(
                        False, error="Not executed because an earlier call in this batch failed."
                    )
                else:
                    safe_arguments = arguments or {}
                    self.logger(f"Tool call: {name}")
                    self.logger(f"Arguments: {_safe_tool_call_summary(name, safe_arguments)}")
                    try:
                        result = self.tool_executor(name, safe_arguments)
                    except Exception as exc:
                        result = ToolResult(False, error=f"Tool execution failed: {exc}")
                    if not isinstance(result, ToolResult):
                        result = ToolResult(False, error="Tool returned an invalid result.")
                messages.append({
                    "role": "tool", "tool_call_id": call_id, "name": name,
                    "content": _tool_result_payload(result),
                })
                self.logger(f"[Tool result] success={result.success}")

                command = (arguments or {}).get("command")
                if (
                    call_was_executed
                    and name == "execute_command"
                    and is_verification_command(command)
                ):
                    item = VerificationEvidence(
                        tier=VerificationTier.SELF,
                        step=step_number,
                        command=" ".join(command),
                        success=result.success,
                        output=result.output,
                        error=result.error,
                        workspace_revision=workspace_revision,
                    )
                    evidence.append(item)
                    if result.success:
                        last_verified_revision = workspace_revision
                    self.logger(
                        f"[Verification evidence] {' '.join(command)}: "
                        f"{'passed' if result.success else 'failed'}"
                    )
                    if not result.success:
                        failed_verification_reason = _verification_failure_reason(
                            command, result
                        )
                if not result.success:
                    batch_blocked = True
                    continue

                planned_step.status = PlanStepStatus.COMPLETED
                current_step_index += 1
                plan_completion_reminders = 0
                command = (arguments or {}).get("command")
                if name == "write_file":
                    verification_required = True
                    workspace_revision += 1

            if failed_verification_reason is not None and step_number < self.max_steps:
                self.logger("[Verification failed] replanning recovery work")
                if replans >= self.max_replans:
                    return self._plan_failed(
                        "The maximum number of replans was exceeded after verification failures.",
                        evidence,
                        plan_history,
                        step_number,
                        messages,
                    )
                replans += 1
                new_plan, planning_error, fatal_error = self._replan(
                    task=task,
                    current_plan=current_plan,
                    reason=failed_verification_reason,
                    revision=replans,
                    verification_required=True,
                    messages=messages,
                )
                if fatal_error:
                    return self._fatal(
                        fatal_error, evidence, plan_history, step_number, messages
                    )
                if new_plan is None:
                    return self._plan_failed(
                        planning_error or "Unable to plan recovery from failed verification.",
                        evidence,
                        plan_history,
                        step_number,
                        messages,
                    )
                current_plan = new_plan
                plan_history.append(current_plan)
                current_step_index = 0
                plan_completion_reminders = 0
                self._log_plan(current_plan, replan=True)
                messages.append({
                    "role": "system",
                    "content": (
                        "Verification failed, so the revised recovery plan is accepted and "
                        "mandatory. Execute its pending repair and verification steps in order:\n"
                        f"{plan_as_json(current_plan)}"
                    ),
                })
                continue

        self.logger("[Status] MAX_STEPS_REACHED")
        return self._result(
            AgentStatus.MAX_STEPS_REACHED,
            "Agent stopped after reaching the maximum number of steps.",
            evidence, plan_history, self.max_steps, messages,
            "The agent did not finish within the configured step limit.",
        )
