"""Minimal tool-calling agent with verification-aware termination."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

from tool_schemas import TOOLS
from tools import ToolResult, execute_tool


MAX_STEPS = 20
MAX_VERIFICATION_REQUESTS = 3

SYSTEM_PROMPT = """You are a coding agent working only in the local workspace.
Use the provided tools to inspect, create, and modify code. Inspect relevant files before changing them, and never assume file contents when a tool can provide the facts.
Treat every ToolResult as authoritative. If a tool fails, use its error to correct the request or explain why the task cannot be completed.
For code changes, run an applicable allowed test or execution command after the latest modification. Only claim completion after a real execute_command ToolResult shows successful verification. If verification fails, fix the problem and verify again.
When finished, give a concise summary and state which verification command was run. Do not claim that tests passed unless the tool result actually succeeded.
Commands must be passed as JSON arrays and must match the execute_command tool description exactly."""

MODIFICATION_TERMS = {
    "create",
    "implement",
    "add",
    "modify",
    "update",
    "change",
    "fix",
    "repair",
    "refactor",
    "rewrite",
    "write",
    "remove",
    "delete",
}
VERIFICATION_TERMS = {
    "test",
    "tests",
    "pytest",
    "unittest",
    "verify",
    "validate",
    "check",
    "build",
}
INFORMATION_TERMS = {
    "inspect",
    "explain",
    "describe",
    "show",
    "list",
    "read",
    "search",
    "what",
    "why",
    "how",
    "tell",
}
MODIFICATION_PHRASES = (
    "创建",
    "实现",
    "添加",
    "修改",
    "更新",
    "修复",
    "重构",
    "编写",
    "删除",
)
VERIFICATION_PHRASES = ("测试", "验证", "检查", "构建", "运行测试")
INFORMATION_PHRASES = ("查看", "列出", "读取", "搜索", "解释", "说明", "是什么", "为什么", "如何")


class AgentStatus(str, Enum):
    COMPLETED = "COMPLETED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    FATAL_ERROR = "FATAL_ERROR"


@dataclass(frozen=True)
class VerificationEvidence:
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
    steps: int
    messages: list[dict[str, Any]]
    error: str | None = None


class LLMChatClient(Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> Any: ...


class AgentProtocolError(RuntimeError):
    """Raised when an assistant response cannot form a valid message history."""


def _words(text: str) -> set[str]:
    normalized = "".join(character if character.isalnum() else " " for character in text)
    return set(normalized.casefold().split())


def task_requires_verification(task: str) -> bool:
    """Classify tasks with conservative, deterministic keyword rules."""
    words = _words(task)
    folded = task.casefold()
    if words & (MODIFICATION_TERMS | VERIFICATION_TERMS):
        return True
    if any(term in task for term in MODIFICATION_PHRASES + VERIFICATION_PHRASES):
        return True
    if words & INFORMATION_TERMS:
        return False
    if any(term in task for term in INFORMATION_PHRASES):
        return False
    # Ambiguous tasks default to verification rather than unverified completion.
    return True


def is_verification_command(command: Any) -> bool:
    """Return whether an allowed command represents test or runtime evidence."""
    if not isinstance(command, list) or any(
        not isinstance(argument, str) for argument in command
    ):
        return False
    if command in (["pytest"], ["python", "-m", "pytest"]):
        return True
    return (
        len(command) == 2
        and command[0] == "python"
        and command[1].casefold().endswith(".py")
    )


def serialize_tool_result(result: ToolResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False)


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_tool_call(raw_call: Any) -> dict[str, Any]:
    call_id = _get_value(raw_call, "id")
    if not isinstance(call_id, str) or not call_id:
        raise AgentProtocolError("Assistant tool call is missing a valid id.")

    function = _get_value(raw_call, "function")
    if function is None:
        raise AgentProtocolError(f"Tool call '{call_id}' has no function payload.")
    name = _get_value(function, "name", "")
    arguments = _get_value(function, "arguments", "{}")
    if not isinstance(name, str):
        name = ""
    if not isinstance(arguments, str):
        arguments = ""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _normalize_assistant_message(response: Any) -> dict[str, Any]:
    if isinstance(response, str):
        return {"role": "assistant", "content": response}
    if isinstance(response, Mapping):
        data = dict(response)
    elif hasattr(response, "model_dump"):
        data = response.model_dump(exclude_none=True)
    else:
        data = {
            "content": _get_value(response, "content"),
            "tool_calls": _get_value(response, "tool_calls"),
        }

    raw_calls = data.get("tool_calls")
    if raw_calls is not None and not isinstance(raw_calls, (list, tuple)):
        raise AgentProtocolError("Assistant tool_calls must be a list.")
    tool_calls = [_normalize_tool_call(call) for call in (raw_calls or [])]
    content = data.get("content")
    if not tool_calls and (not isinstance(content, str) or not content.strip()):
        raise AgentProtocolError(
            "Assistant response contains neither tool calls nor final text."
        )

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _parse_arguments(raw_arguments: Any) -> tuple[dict[str, Any] | None, ToolResult | None]:
    if not isinstance(raw_arguments, str):
        return None, ToolResult(False, error="Tool arguments must be a JSON string.")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return None, ToolResult(False, error=f"Invalid tool arguments JSON: {exc}")
    if not isinstance(arguments, dict):
        return None, ToolResult(False, error="Tool arguments must decode to an object.")
    return arguments, None


def _command_text(command: list[str]) -> str:
    return " ".join(command)


class CodingAgent:
    def __init__(
        self,
        llm_client: LLMChatClient,
        *,
        tool_executor: Callable[[str, Mapping[str, Any]], ToolResult] = execute_tool,
        max_steps: int = MAX_STEPS,
        max_verification_requests: int = MAX_VERIFICATION_REQUESTS,
        verbose: bool = True,
        output: Callable[[str], None] = print,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if max_verification_requests < 0:
            raise ValueError("max_verification_requests cannot be negative.")
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.max_verification_requests = max_verification_requests
        self.verbose = verbose
        self.output = output

    def _emit(self, message: str) -> None:
        if self.verbose:
            self.output(message)

    def _log_arguments(self, arguments: dict[str, Any] | None) -> str:
        if arguments is None:
            return "<invalid JSON>"
        visible = dict(arguments)
        content = visible.get("content")
        if isinstance(content, str):
            visible["content"] = f"<{len(content)} characters>"
        rendered = json.dumps(visible, ensure_ascii=False)
        return rendered if len(rendered) <= 500 else rendered[:480] + "...[truncated]"

    def _fatal(
        self,
        message: str,
        step: int,
        messages: list[dict[str, Any]],
        evidence: list[VerificationEvidence],
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FATAL_ERROR,
            final_answer="Agent stopped because of a fatal error.",
            verification_evidence=evidence,
            steps=step,
            messages=messages,
            error=message,
        )

    def run(self, task: str) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            return self._fatal("Task must be a non-empty string.", 0, [], [])

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        evidence: list[VerificationEvidence] = []
        verification_required = task_requires_verification(task)
        verification_requests = 0
        workspace_revision = 0

        for step in range(1, self.max_steps + 1):
            self._emit(f"[Step {step}]")
            try:
                response = self.llm_client.chat(messages=messages, tools=TOOLS)
                assistant_message = _normalize_assistant_message(response)
            except Exception as exc:
                return self._fatal(
                    f"LLM communication or message error: {exc}",
                    step,
                    messages,
                    evidence,
                )

            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    raw_arguments = tool_call["function"]["arguments"]
                    arguments, parse_error = _parse_arguments(raw_arguments)
                    self._emit(f"Tool call: {name or '<missing name>'}")
                    self._emit(f"Arguments: {self._log_arguments(arguments)}")

                    if parse_error is not None:
                        result = parse_error
                    else:
                        try:
                            result = self.tool_executor(name, arguments or {})
                            if not isinstance(result, ToolResult):
                                result = ToolResult(
                                    False, error="Tool returned an invalid result type."
                                )
                        except Exception as exc:
                            result = ToolResult(
                                False, error=f"Unexpected tool execution error: {exc}"
                            )

                    if name == "write_file" and result.success:
                        workspace_revision += 1
                        verification_required = True

                    command = (
                        arguments.get("command")
                        if name == "execute_command" and arguments is not None
                        else None
                    )
                    if is_verification_command(command):
                        verification = VerificationEvidence(
                            command=_command_text(command),
                            success=result.success,
                            output=result.output,
                            error=result.error,
                            step=step,
                            workspace_revision=workspace_revision,
                        )
                        evidence.append(verification)
                        state = "passed" if result.success else "failed"
                        self._emit(
                            f"[Verification evidence] {verification.command}: {state}"
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": serialize_tool_result(result),
                        }
                    )
                    self._emit(f"[Tool result] success={result.success}")
                continue

            final_text = assistant_message["content"].strip()
            if not verification_required:
                return AgentResult(
                    AgentStatus.COMPLETED,
                    final_text,
                    evidence,
                    step,
                    messages,
                )

            current_verification = bool(
                evidence
                and evidence[-1].success
                and evidence[-1].workspace_revision == workspace_revision
            )
            if current_verification:
                return AgentResult(
                    AgentStatus.COMPLETED,
                    final_text,
                    evidence,
                    step,
                    messages,
                )

            if verification_requests >= self.max_verification_requests:
                return AgentResult(
                    AgentStatus.VERIFICATION_REQUIRED,
                    (
                        "Task was not marked completed because no successful "
                        "verification evidence was obtained for the latest changes."
                    ),
                    evidence,
                    step,
                    messages,
                    error="Successful current verification is required.",
                )

            verification_requests += 1
            if evidence and not evidence[-1].success:
                reason = (
                    "The latest verification command failed. Fix the problem and run "
                    "an allowed verification command again before giving a final answer."
                )
            elif evidence:
                reason = (
                    "The workspace changed after the last successful verification. "
                    "Run an allowed verification command for the current code before "
                    "giving a final answer."
                )
            else:
                reason = (
                    "This task requires verification, but no successful verification "
                    "evidence exists. Run an applicable allowed test or Python command "
                    "with execute_command before giving a final answer."
                )
            messages.append({"role": "system", "content": reason})
            self._emit(f"[Verification required] request {verification_requests}")

        missing = verification_required and not (
            evidence
            and evidence[-1].success
            and evidence[-1].workspace_revision == workspace_revision
        )
        final_answer = "Maximum agent steps reached before the task completed."
        if missing:
            final_answer += " No successful current verification evidence was obtained."
        return AgentResult(
            AgentStatus.MAX_STEPS_REACHED,
            final_answer,
            evidence,
            self.max_steps,
            messages,
            error="Maximum agent steps reached.",
        )
