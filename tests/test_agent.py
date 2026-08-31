"""Unit tests for the v0.3 agent and verified termination loop."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from agent import (
    AgentStatus,
    CodingAgent,
    is_verification_command,
    task_requires_verification,
)
from tools import ToolResult, execute_tool


def tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": encoded},
    }


def tool_response(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def final_response(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> Any:
        self.calls.append(
            {"messages": deepcopy(list(messages)), "tools": deepcopy(list(tools))}
        )
        if not self.responses:
            raise AssertionError("FakeLLM has no response left")
        return self.responses.pop(0)


class FakeExecutor:
    def __init__(self, results: list[ToolResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        if not self.results:
            raise AssertionError("FakeExecutor has no result left")
        return self.results.pop(0)


def run_agent(
    llm: FakeLLM,
    executor=execute_tool,
    **kwargs: Any,
):
    return CodingAgent(
        llm,
        tool_executor=executor,
        verbose=False,
        **kwargs,
    ).run


def test_information_task_direct_answer_completes() -> None:
    llm = FakeLLM([final_response("It is a calculator project.")])

    result = run_agent(llm)("Inspect the workspace and explain what it does.")

    assert result.status is AgentStatus.COMPLETED
    assert result.final_answer == "It is a calculator project."
    assert result.verification_evidence == []


def test_single_tool_call_returns_observation_to_llm() -> None:
    llm = FakeLLM(
        [
            tool_response(tool_call("call-read", "read_file", {"path": "a.py"})),
            final_response("The file prints hello."),
        ]
    )
    executor = FakeExecutor([ToolResult(True, "print('hello')")])

    result = run_agent(llm, executor)("Read and explain a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls == [("read_file", {"path": "a.py"})]
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-read"
    assert json.loads(tool_message["content"]) == {
        "success": True,
        "output": "print('hello')",
        "error": None,
    }


def test_code_change_requires_successful_verification() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call(
                    "call-write",
                    "write_file",
                    {"path": "a.py", "content": "print('ok')"},
                )
            ),
            tool_response(
                tool_call("call-test", "execute_command", {"command": ["pytest"]})
            ),
            final_response("Implemented and tested."),
        ]
    )
    executor = FakeExecutor(
        [
            ToolResult(True, "Wrote file."),
            ToolResult(True, "exit_code: 0\n1 passed"),
        ]
    )

    result = run_agent(llm, executor)("Implement the requested code change.")

    assert result.status is AgentStatus.COMPLETED
    assert len(result.verification_evidence) == 1
    evidence = result.verification_evidence[0]
    assert evidence.command == "pytest"
    assert evidence.success
    assert evidence.output == "exit_code: 0\n1 passed"
    assert evidence.error is None
    assert evidence.step == 2
    assert evidence.workspace_revision == 1


def test_unverified_change_prompts_model_then_accepts_real_verification() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call(
                    "write",
                    "write_file",
                    {"path": "a.py", "content": "value = 1"},
                )
            ),
            final_response("Done without running tests."),
            tool_response(
                tool_call(
                    "verify", "execute_command", {"command": ["python", "a.py"]}
                )
            ),
            final_response("Implemented and verified."),
        ]
    )
    executor = FakeExecutor(
        [ToolResult(True, "Wrote file."), ToolResult(True, "exit_code: 0")]
    )

    result = run_agent(llm, executor)("Create a.py.")

    assert result.status is AgentStatus.COMPLETED
    reminder_messages = [
        message
        for message in llm.calls[2]["messages"]
        if message["role"] == "system" and "requires verification" in message["content"]
    ]
    assert reminder_messages


def test_failed_verification_can_be_fixed_and_reverified() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call("w1", "write_file", {"path": "a.py", "content": "bad"})
            ),
            tool_response(
                tool_call("t1", "execute_command", {"command": ["pytest"]})
            ),
            tool_response(
                tool_call("w2", "write_file", {"path": "a.py", "content": "good"})
            ),
            tool_response(
                tool_call("t2", "execute_command", {"command": ["pytest"]})
            ),
            final_response("Fixed and verified."),
        ]
    )
    executor = FakeExecutor(
        [
            ToolResult(True, "wrote bad"),
            ToolResult(False, "exit_code: 1", "tests failed"),
            ToolResult(True, "wrote good"),
            ToolResult(True, "exit_code: 0\n1 passed"),
        ]
    )

    result = run_agent(llm, executor)("Fix the failing code.")

    assert result.status is AgentStatus.COMPLETED
    assert [item.success for item in result.verification_evidence] == [False, True]
    assert result.verification_evidence[-1].workspace_revision == 2


def test_failed_verification_then_final_is_not_completed() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call("w", "write_file", {"path": "a.py", "content": "bad"})
            ),
            tool_response(
                tool_call("t", "execute_command", {"command": ["pytest"]})
            ),
            final_response("Everything is complete."),
            final_response("Still complete."),
        ]
    )
    executor = FakeExecutor(
        [
            ToolResult(True, "wrote"),
            ToolResult(False, "exit_code: 1", "failed"),
        ]
    )

    result = run_agent(llm, executor, max_verification_requests=1)("Fix a.py.")

    assert result.status is AgentStatus.VERIFICATION_REQUIRED
    assert result.status is not AgentStatus.COMPLETED
    assert result.verification_evidence[-1].success is False


def test_multiple_tool_calls_all_execute_and_keep_ids() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call("one", "list_files", {"path": "."}),
                tool_call("two", "read_file", {"path": "a.py"}),
            ),
            final_response("Inspected both results."),
        ]
    )
    executor = FakeExecutor(
        [ToolResult(True, "a.py"), ToolResult(True, "content")]
    )

    result = run_agent(llm, executor)("Inspect and explain the workspace.")

    assert result.status is AgentStatus.COMPLETED
    assert [call[0] for call in executor.calls] == ["list_files", "read_file"]
    tool_messages = [
        message for message in llm.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == ["one", "two"]


def test_invalid_json_is_observed_without_execution() -> None:
    llm = FakeLLM(
        [
            tool_response(tool_call("bad-json", "read_file", "{not json")),
            final_response("The tool arguments were invalid."),
        ]
    )
    executor = FakeExecutor([])

    result = run_agent(llm, executor)("Inspect and explain a file.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls == []
    observation = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert observation["success"] is False
    assert "Invalid tool arguments JSON" in observation["error"]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        {},
        {"path": "a.py", "unexpected": True},
    ],
)
def test_non_object_missing_and_extra_arguments_are_observed(arguments: Any) -> None:
    llm = FakeLLM(
        [
            tool_response(tool_call("bad-arguments", "read_file", arguments)),
            final_response("The arguments were rejected."),
        ]
    )

    result = run_agent(llm)("Inspect and explain a file.")

    assert result.status is AgentStatus.COMPLETED
    observation = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert observation["success"] is False
    assert observation["error"]


def test_unknown_tool_is_an_observation() -> None:
    llm = FakeLLM(
        [
            tool_response(tool_call("unknown", "delete_everything", {})),
            final_response("That tool is unavailable."),
        ]
    )

    result = run_agent(llm)("Inspect and explain the workspace.")

    assert result.status is AgentStatus.COMPLETED
    observation = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert observation["success"] is False
    assert "Unknown tool" in observation["error"]


def test_ordinary_tool_failure_returns_to_model() -> None:
    llm = FakeLLM(
        [
            tool_response(tool_call("missing", "read_file", {"path": "missing.py"})),
            final_response("The requested file does not exist."),
        ]
    )
    executor = FakeExecutor([ToolResult(False, error="File does not exist")])

    result = run_agent(llm, executor)("Read and explain missing.py.")

    assert result.status is AgentStatus.COMPLETED
    assert json.loads(llm.calls[1]["messages"][-1]["content"])["success"] is False


def test_successful_information_command_is_not_verification() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call("status", "execute_command", {"command": ["git", "status"]})
            ),
            final_response("The code is fixed."),
        ]
    )
    executor = FakeExecutor([ToolResult(True, "exit_code: 0")])

    result = run_agent(llm, executor, max_verification_requests=0)("Fix the code.")

    assert result.status is AgentStatus.VERIFICATION_REQUIRED
    assert result.verification_evidence == []


def test_failed_execute_command_is_recorded_and_returned() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call("test", "execute_command", {"command": ["pytest"]})
            ),
            final_response("Tests passed."),
        ]
    )
    executor = FakeExecutor([ToolResult(False, "exit_code: 1", "failure")])

    result = run_agent(llm, executor, max_verification_requests=0)("Run tests.")

    assert result.status is AgentStatus.VERIFICATION_REQUIRED
    assert len(result.verification_evidence) == 1
    assert not result.verification_evidence[0].success
    observation = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert observation["error"] == "failure"


def test_max_verification_requests_prevents_loop() -> None:
    llm = FakeLLM(
        [
            final_response("Done."),
            final_response("Done again."),
            final_response("Still done."),
        ]
    )

    result = run_agent(llm, max_verification_requests=2)("Implement a feature.")

    assert result.status is AgentStatus.VERIFICATION_REQUIRED
    assert result.steps == 3


def test_max_steps_prevents_infinite_tool_loop() -> None:
    llm = FakeLLM(
        [
            tool_response(tool_call("one", "list_files", {"path": "."})),
            tool_response(tool_call("two", "list_files", {"path": "."})),
        ]
    )
    executor = FakeExecutor([ToolResult(True, "files"), ToolResult(True, "files")])

    result = run_agent(llm, executor, max_steps=2)("Inspect this unusual project task.")

    assert result.status is AgentStatus.MAX_STEPS_REACHED
    assert result.steps == 2


def test_successful_verification_becomes_stale_after_later_write() -> None:
    llm = FakeLLM(
        [
            tool_response(
                tool_call("test", "execute_command", {"command": ["pytest"]}),
                tool_call("write", "write_file", {"path": "a.py", "content": "new"}),
            ),
            final_response("Done."),
        ]
    )
    executor = FakeExecutor(
        [ToolResult(True, "exit_code: 0"), ToolResult(True, "wrote")]
    )

    result = run_agent(llm, executor, max_verification_requests=0)("Fix a.py.")

    assert result.status is AgentStatus.VERIFICATION_REQUIRED
    assert result.verification_evidence[0].workspace_revision == 0


def test_llm_failure_is_fatal() -> None:
    llm = FakeLLM([])

    result = run_agent(llm)("Inspect and explain the workspace.")

    assert result.status is AgentStatus.FATAL_ERROR
    assert "LLM communication" in result.error


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Explain what calculator.py does.", False),
        ("Fix the calculator bug.", True),
        ("Create a calculator module.", True),
        ("Run the tests and verify the result.", True),
        ("Help with this repository.", True),
        ("解释这个项目的作用。", False),
        ("修复这个项目。", True),
    ],
)
def test_task_requires_verification_rules(task: str, expected: bool) -> None:
    assert task_requires_verification(task) is expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["pytest"], True),
        (["python", "-m", "pytest"], True),
        (["python", "smoke.py"], True),
        (["git", "status"], False),
        (["git", "diff"], False),
        ("pytest", False),
    ],
)
def test_verification_command_rules(command: Any, expected: bool) -> None:
    assert is_verification_command(command) is expected
