"""Unit tests for the v0.4 plan-constrained agent loop."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from agent import (
    AgentStatus,
    CodingAgent,
    VerificationTier,
    task_requires_verification,
)
from planning import (
    MAX_PLAN_STEPS,
    PlanStep,
    PlanStepStatus,
    TaskPlan,
    is_verification_command,
)
from tools import ToolResult
from workspace_snapshot import OriginalTestDiscovery, OriginalTestRun
from verifier import VerifierVerdict


def planned_step(
    step_id: str,
    tool: str,
    constraints: dict[str, Any],
    description: str = "Perform an atomic action",
) -> dict[str, Any]:
    return {
        "id": step_id,
        "description": description,
        "tool": tool,
        "argument_constraints": constraints,
        "rationale": "This action directly supports the user goal",
    }


def plan_response(
    steps: list[dict[str, Any]],
    goal: str = "Complete the requested task",
) -> str:
    return json.dumps({
        "goal": goal,
        "success_criteria": ["The requested outcome is complete"],
        "steps": steps,
    })


def tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": encoded},
    }


def tool_response(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def final_response(content: str = "Done.") -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def verifier_response(verdict: str = "PASS") -> str:
    return json.dumps({
        "verdict": verdict,
        "summary": (
            "Independent review passed."
            if verdict == "PASS" else "The implementation has a counterexample."
        ),
        "requirement_checks": [{
            "requirement": "The requested implementation is correct",
            "satisfied": verdict == "PASS",
            "reason": "Reviewed independently from Builder tests",
        }],
        "counterexamples": [] if verdict == "PASS" else ["Empty input fails"],
        "unresolved_assumptions": [],
    })


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({
            "messages": deepcopy(list(messages)),
            "tools": deepcopy(list(tools)) if tools is not None else None,
        })
        if not self.responses:
            raise AssertionError("FakeLLM has no response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeExecutor:
    def __init__(
        self,
        results: list[ToolResult] | None = None,
        inventory: ToolResult | None = None,
    ) -> None:
        self.results = list(results or [])
        self.inventory = inventory or ToolResult(True, "[FILE] calculator.py")
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        if len(self.calls) == 1 and name == "list_files" and arguments == {"path": "."}:
            return self.inventory
        if not self.results:
            raise AssertionError("FakeExecutor has no result left")
        return self.results.pop(0)


class FakeSnapshot:
    def __init__(
        self,
        *,
        discovery: OriginalTestDiscovery | None = None,
        original_result: OriginalTestRun | None = None,
        capture_error: Exception | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.discovery = discovery or OriginalTestDiscovery(
            available=True, test_files=("test_original.py",)
        )
        self.original_result = original_result or OriginalTestRun(
            success=True,
            command="python -m pytest <original tests>",
            output="1 passed",
        )
        self.capture_error = capture_error
        self.run_error = run_error
        self.events: list[str] = []

    def capture(self) -> None:
        self.events.append("capture")
        if self.capture_error:
            raise self.capture_error

    def run_original_tests(self) -> OriginalTestRun:
        self.events.append("run_original_tests")
        if self.run_error:
            raise self.run_error
        return self.original_result

    def cleanup(self) -> None:
        self.events.append("cleanup")


def run_agent(
    llm: FakeLLM,
    executor: FakeExecutor | None = None,
    **kwargs: Any,
):
    snapshot = kwargs.pop("snapshot", None) or FakeSnapshot()
    return CodingAgent(
        llm,
        tool_executor=executor or FakeExecutor(),
        logger=lambda _message: None,
        snapshot_factory=lambda: snapshot,
        **kwargs,
    ).run


def test_planning_precedes_tools_and_receives_root_inventory() -> None:
    llm = FakeLLM([plan_response([]), final_response("A calculator project.")])
    executor = FakeExecutor()

    result = run_agent(llm, executor)("Explain this project.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls == [("list_files", {"path": "."})]
    assert llm.calls[0]["tools"] is None
    planning_text = "\n".join(
        str(message.get("content", "")) for message in llm.calls[0]["messages"]
    )
    assert "[FILE] calculator.py" in planning_text
    assert llm.calls[1]["tools"] is not None


def test_working_memory_is_injected_into_every_llm_call() -> None:
    llm = FakeLLM([plan_response([]), final_response("Done.")])

    result = run_agent(llm)("Explain this project. You must only inspect it.")

    assert result.status is AgentStatus.COMPLETED
    assert result.working_memory is not None
    assert result.working_memory.constraints == [
        "Explain this project. You must only inspect it."
    ]
    for call in llm.calls:
        memory_messages = [
            message for message in call["messages"]
            if str(message.get("content", "")).startswith("WORKING MEMORY (")
        ]
        assert len(memory_messages) == 1


def test_duplicate_read_in_same_revision_is_rejected_and_replanned() -> None:
    repeated_reads = [
        planned_step("read_once", "read_file", {"path": "a.py"}),
        planned_step("read_again", "read_file", {"path": "a.py"}),
    ]
    llm = FakeLLM([
        plan_response(repeated_reads),
        tool_response(tool_call("first", "read_file", {"path": "a.py"})),
        tool_response(tool_call("duplicate", "read_file", {"path": "a.py"})),
        plan_response([]),
        final_response("Inspected without rereading."),
    ])
    executor = FakeExecutor(results=[ToolResult(True, "source")])

    result = run_agent(llm, executor)("Inspect a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls[1:] == [("read_file", {"path": "a.py"})]
    assert len(result.plan_history) == 2
    assert result.working_memory is not None
    assert result.working_memory.read_files["a.py"].reads == 1
    duplicate_observation = next(
        message for message in llm.calls[3]["messages"]
        if message.get("tool_call_id") == "duplicate"
    )
    assert "Duplicate read_file refused" in duplicate_observation["content"]


def test_duplicate_paths_in_one_batch_are_normalized_and_batch_is_atomic() -> None:
    steps = [
        planned_step("read_one", "read_file", {"path": "a.py"}),
        planned_step("read_alias", "read_file", {"path": "./a.py"}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(
            tool_call("one", "read_file", {"path": "a.py"}),
            tool_call("alias", "read_file", {"path": "./a.py"}),
        ),
        plan_response([]),
        final_response("Done."),
    ])
    executor = FakeExecutor()

    result = run_agent(llm, executor)("Inspect a.py once.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls == [("list_files", {"path": "."})]


def test_write_revision_allows_a_needed_reread() -> None:
    steps = [
        planned_step("read", "read_file", {"path": "a.py"}),
        planned_step("write", "write_file", {"path": "a.py"}),
        planned_step("reread", "read_file", {"path": "a.py"}),
        planned_step(
            "verify", "execute_command", {"command": ["python", "a.py"]}
        ),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("read", "read_file", {"path": "a.py"})),
        tool_response(
            tool_call("write", "write_file", {"path": "a.py", "content": "ok"})
        ),
        tool_response(tool_call("reread", "read_file", {"path": "a.py"})),
        tool_response(
            tool_call("verify", "execute_command", {"command": ["python", "a.py"]})
        ),
        final_response("Done."),
    ])
    executor = FakeExecutor(results=[
        ToolResult(True, "old"),
        ToolResult(True, "written"),
        ToolResult(True, "new"),
        ToolResult(True, "passed"),
    ])

    result = run_agent(llm, executor)("Modify a.py and verify it.")

    assert result.status is AgentStatus.COMPLETED
    assert result.working_memory is not None
    assert result.working_memory.workspace_revision == 1
    assert result.working_memory.read_files["a.py"].reads == 2
    assert result.working_memory.read_files["a.py"].workspace_revision == 1
    assert result.working_memory.modified_files["a.py"].writes == 1


def test_independent_verifier_uses_fresh_context_and_excludes_builder_tests() -> None:
    steps = [
        planned_step("source", "write_file", {"path": "feature.py"}),
        planned_step("tests", "write_file", {"path": "test_feature.py"}),
        planned_step("verify", "execute_command", {"command": ["pytest"]}),
    ]
    builder = FakeLLM([
        plan_response(steps),
        tool_response(tool_call(
            "source", "write_file", {"path": "feature.py", "content": "def f(): return 1"}
        )),
        tool_response(tool_call(
            "tests", "write_file", {"path": "test_feature.py", "content": "def test_f(): pass"}
        )),
        tool_response(tool_call("verify", "execute_command", {"command": ["pytest"]})),
        final_response("Implemented."),
    ])
    verifier = FakeLLM([verifier_response()])
    executor = FakeExecutor(results=[
        ToolResult(True, "wrote source"),
        ToolResult(True, "wrote tests"),
        ToolResult(True, "1 passed"),
        ToolResult(True, "def f(): return 1"),
    ])

    result = run_agent(
        builder, executor, verifier_client=verifier
    )("Implement feature.py and test it.")

    assert result.status is AgentStatus.COMPLETED
    assert result.verifier_review is not None
    assert result.verifier_review.verdict is VerifierVerdict.PASS
    assert len(verifier.calls) == 1
    assert verifier.calls[0]["tools"] is None
    verifier_messages = verifier.calls[0]["messages"]
    assert [message["role"] for message in verifier_messages] == ["system", "user"]
    context_text = verifier_messages[1]["content"]
    assert "def f(): return 1" in context_text
    assert "def test_f(): pass" not in context_text
    assert all("tool_calls" not in message for message in verifier_messages)


def test_verifier_failure_is_advice_and_does_not_block_original_regression() -> None:
    steps = [
        planned_step("source", "write_file", {"path": "feature.py"}),
        planned_step("verify", "execute_command", {"command": ["pytest"]}),
    ]
    builder = FakeLLM([
        plan_response(steps),
        tool_response(tool_call(
            "source", "write_file", {"path": "feature.py", "content": "def f(): return 0"}
        )),
        tool_response(tool_call("verify", "execute_command", {"command": ["pytest"]})),
        final_response("Implemented."),
    ])
    verifier = FakeLLM([verifier_response("FAIL")])
    executor = FakeExecutor(results=[
        ToolResult(True, "wrote"),
        ToolResult(True, "1 passed"),
        ToolResult(True, "def f(): return 0"),
    ])
    snapshot = FakeSnapshot()

    result = run_agent(
        builder, executor, verifier_client=verifier, snapshot=snapshot
    )("Implement feature.py.")

    assert result.status is AgentStatus.COMPLETED
    assert result.verifier_review is not None
    assert result.verifier_review.counterexamples == ["Empty input fails"]
    assert "run_original_tests" in snapshot.events


def test_invalid_verifier_protocol_retries_then_returns_verifier_failed() -> None:
    steps = [
        planned_step("source", "write_file", {"path": "feature.py"}),
        planned_step("verify", "execute_command", {"command": ["pytest"]}),
    ]
    builder = FakeLLM([
        plan_response(steps),
        tool_response(tool_call(
            "source", "write_file", {"path": "feature.py", "content": "x = 1"}
        )),
        tool_response(tool_call("verify", "execute_command", {"command": ["pytest"]})),
        final_response(),
    ])
    verifier = FakeLLM(["not-json", "still-not-json"])
    executor = FakeExecutor(results=[
        ToolResult(True, "wrote"),
        ToolResult(True, "passed"),
        ToolResult(True, "x = 1"),
    ])

    result = run_agent(
        builder, executor, verifier_client=verifier
    )("Implement feature.py.")

    assert result.status is AgentStatus.VERIFIER_FAILED
    assert len(verifier.calls) == 2


def test_information_task_does_not_require_independent_code_verifier() -> None:
    builder = FakeLLM([plan_response([]), final_response("An example project.")])
    verifier = FakeLLM([])

    result = run_agent(
        builder, verifier_client=verifier
    )("Explain this project.")

    assert result.status is AgentStatus.COMPLETED
    assert verifier.calls == []


def test_inventory_failure_returns_plan_failed_without_llm_call() -> None:
    llm = FakeLLM([])
    executor = FakeExecutor(inventory=ToolResult(False, error="cannot list root"))

    result = run_agent(llm, executor)("Explain the project.")

    assert result.status is AgentStatus.PLAN_FAILED
    assert llm.calls == []


def test_snapshot_is_captured_before_inventory_and_always_cleaned() -> None:
    snapshot = FakeSnapshot()
    llm = FakeLLM([plan_response([]), final_response("Explained.")])

    def ordered_executor(name: str, arguments: Mapping[str, Any]) -> ToolResult:
        assert snapshot.events == ["capture"]
        assert name == "list_files"
        return ToolResult(True, "[empty directory]")

    result = CodingAgent(
        llm,
        tool_executor=ordered_executor,
        snapshot_factory=lambda: snapshot,
        logger=lambda _message: None,
    ).run("Explain the project.")

    assert result.status is AgentStatus.COMPLETED
    assert snapshot.events == ["capture", "cleanup"]
    assert "run_original_tests" not in snapshot.events


def test_snapshot_failure_is_fatal_before_llm_or_inventory() -> None:
    snapshot = FakeSnapshot(capture_error=RuntimeError("copy failed"))
    llm = FakeLLM([])
    executor = FakeExecutor()

    result = run_agent(llm, executor, snapshot=snapshot)("Fix the code.")

    assert result.status is AgentStatus.FATAL_ERROR
    assert llm.calls == []
    assert executor.calls == []


def test_legal_write_and_verification_plan_completes_in_order() -> None:
    steps = [
        planned_step("write", "write_file", {"path": "a.py"}),
        planned_step("verify", "execute_command", {"command": ["python", "-m", "pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("w", "write_file", {"path": "a.py", "content": "x = 1"})),
        tool_response(tool_call("t", "execute_command", {"command": ["python", "-m", "pytest"]})),
        final_response("Implemented and verified."),
    ])
    executor = FakeExecutor([ToolResult(True, "wrote"), ToolResult(True, "1 passed")])

    result = run_agent(llm, executor)("Create a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert [name for name, _ in executor.calls] == ["list_files", "write_file", "execute_command"]
    assert all(step.status is PlanStepStatus.COMPLETED for step in result.plan_history[0].steps)
    assert result.verification_evidence[-1].workspace_revision == 1
    assert result.verification_level is VerificationTier.ORIGINAL
    assert [item.tier for item in result.verification_evidence] == [
        VerificationTier.ORIGINAL,
    ]


def test_write_only_initial_plan_does_not_create_a_verification_plan() -> None:
    initial = [planned_step("write", "write_file", {"path": "a.py"})]
    llm = FakeLLM([
        plan_response(initial),
        tool_response(tool_call("write", "write_file", {
            "path": "a.py", "content": "x = 1"
        })),
        final_response("Implemented."),
    ])
    verifier = FakeLLM([verifier_response()])
    executor = FakeExecutor([
        ToolResult(True, "wrote"),
        ToolResult(True, "x = 1"),
    ])
    snapshot = FakeSnapshot(
        discovery=OriginalTestDiscovery(available=True, test_files=())
    )

    result = run_agent(
        llm,
        executor,
        verifier_client=verifier,
        snapshot=snapshot,
    )("Create a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert [plan.revision for plan in result.plan_history] == [0]
    assert [name for name, _arguments in executor.calls[1:]] == [
        "write_file", "read_file"
    ]
    assert result.working_memory is not None
    assert result.verifier_review is not None
    assert [call["tools"] is None for call in llm.calls] == [True, False, False]


def test_web_only_write_completes_after_independent_review() -> None:
    initial = [planned_step("write", "write_file", {"path": "index.html"})]
    builder = FakeLLM([
        plan_response(initial),
        tool_response(tool_call("write", "write_file", {
            "path": "index.html", "content": "<canvas id='game'></canvas>"
        })),
        final_response("The game page is ready."),
    ])
    verifier = FakeLLM([verifier_response()])
    executor = FakeExecutor([
        ToolResult(True, "wrote"),
        ToolResult(True, "<canvas id='game'></canvas>"),
    ])
    snapshot = FakeSnapshot(
        discovery=OriginalTestDiscovery(available=True, test_files=())
    )

    result = run_agent(
        builder,
        executor,
        verifier_client=verifier,
        snapshot=snapshot,
    )("设计一个网页版小游戏。")

    assert result.status is AgentStatus.COMPLETED
    assert result.working_memory is not None
    assert result.working_memory.verification_state.value == "NOT_REQUIRED"
    assert result.verifier_review is not None


def test_optional_builder_test_does_not_become_completion_evidence() -> None:
    steps = [
        planned_step("write", "write_file", {"path": "a.py"}),
        planned_step("verify", "execute_command", {"command": ["pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("w", "write_file", {"path": "a.py", "content": "x"})),
        tool_response(tool_call("v", "execute_command", {"command": ["pytest"]})),
        final_response("Self-tested."),
    ])
    snapshot = FakeSnapshot(
        discovery=OriginalTestDiscovery(available=True, test_files=())
    )

    result = run_agent(
        llm,
        FakeExecutor([ToolResult(True, "wrote"), ToolResult(True, "passed")]),
        snapshot=snapshot,
    )("Create a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert result.verification_level is None
    assert result.verification_evidence == []
    assert snapshot.events == ["capture", "cleanup"]


def test_original_test_failure_is_terminal_without_another_llm_call() -> None:
    steps = [
        planned_step("write", "write_file", {"path": "a.py"}),
        planned_step("verify", "execute_command", {"command": ["pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("w", "write_file", {"path": "a.py", "content": "bad"})),
        tool_response(tool_call("v", "execute_command", {"command": ["pytest"]})),
        final_response("Done."),
    ])
    snapshot = FakeSnapshot(original_result=OriginalTestRun(
        success=False,
        command="python -m pytest <original tests>",
        output="assert -1 == 5",
        error="Original tests exited with status 1.",
    ))

    result = run_agent(
        llm,
        FakeExecutor([ToolResult(True, "wrote"), ToolResult(True, "passed")]),
        snapshot=snapshot,
    )("Fix a.py.")

    assert result.status is AgentStatus.ORIGINAL_TESTS_FAILED
    assert result.verification_level is None
    assert result.verification_evidence[-1].tier is VerificationTier.ORIGINAL
    assert result.verification_evidence[-1].success is False
    assert len(llm.calls) == 4
    assert snapshot.events == ["capture", "run_original_tests", "cleanup"]


def test_original_collection_failure_can_complete_after_independent_review() -> None:
    snapshot = FakeSnapshot(discovery=OriginalTestDiscovery(
        available=False,
        error="Original test collection failed with exit code 2.",
    ))
    steps = [planned_step("write", "write_file", {"path": "a.py"})]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("w", "write_file", {"path": "a.py", "content": "x"})),
        final_response(),
    ])
    verifier = FakeLLM([verifier_response()])

    result = run_agent(
        llm,
        FakeExecutor([ToolResult(True, "wrote"), ToolResult(True, "x")]),
        snapshot=snapshot,
        verifier_client=verifier,
    )("Create a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert result.verifier_review is not None


def test_original_runner_protocol_failure_is_fatal() -> None:
    steps = [
        planned_step("write", "write_file", {"path": "a.py"}),
        planned_step("verify", "execute_command", {"command": ["pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("w", "write_file", {"path": "a.py", "content": "x"})),
        tool_response(tool_call("v", "execute_command", {"command": ["pytest"]})),
        final_response(),
    ])
    snapshot = FakeSnapshot(run_error=RuntimeError("runner protocol failed"))

    result = run_agent(
        llm,
        FakeExecutor([ToolResult(True, "wrote"), ToolResult(True, "passed")]),
        snapshot=snapshot,
    )("Create a.py.")

    assert result.status is AgentStatus.FATAL_ERROR
    assert "runner protocol failed" in (result.error or "")


def test_dijkstra_greenfield_plan_does_not_read_calculator_files() -> None:
    steps = [
        planned_step("implementation", "write_file", {"path": "dijkstra.py"}),
        planned_step("tests", "write_file", {"path": "test_dijkstra.py"}),
        planned_step("verify", "execute_command", {"command": ["python", "-m", "pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(steps, "Implement and verify Dijkstra"),
        tool_response(
            tool_call("w1", "write_file", {"path": "dijkstra.py", "content": "code"}),
            tool_call("w2", "write_file", {"path": "test_dijkstra.py", "content": "tests"}),
            tool_call("v", "execute_command", {"command": ["python", "-m", "pytest"]}),
        ),
        final_response("Dijkstra is implemented and tested."),
    ])
    executor = FakeExecutor([
        ToolResult(True, "wrote"), ToolResult(True, "wrote"), ToolResult(True, "passed")
    ])

    result = run_agent(llm, executor)("请实现标准 Dijkstra 算法")

    assert result.status is AgentStatus.COMPLETED
    assert all(name != "read_file" for name, _ in executor.calls)
    assert "calculator.py" not in plan_response(steps)


def test_unplanned_call_is_rejected_with_call_id_then_replanned() -> None:
    initial = [planned_step("read_a", "read_file", {"path": "a.py"})]
    revised = [planned_step("read_b", "read_file", {"path": "b.py"})]
    llm = FakeLLM([
        plan_response(initial),
        tool_response(tool_call("deviated-call", "read_file", {"path": "b.py"})),
        plan_response(revised),
        tool_response(tool_call("planned-call", "read_file", {"path": "b.py"})),
        final_response("Explained b.py."),
    ])
    executor = FakeExecutor([ToolResult(True, "contents")])

    result = run_agent(llm, executor)("Explain the requested file.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls[1:] == [("read_file", {"path": "b.py"})]
    deviation_messages = [
        message for message in llm.calls[2]["messages"]
        if message.get("role") == "tool"
    ]
    assert deviation_messages[-1]["tool_call_id"] == "deviated-call"
    assert "Plan deviation" in deviation_messages[-1]["content"]
    assert [plan.revision for plan in result.plan_history] == [0, 1]


def test_more_than_two_replans_returns_plan_failed() -> None:
    def read_plan(path: str) -> str:
        return plan_response([planned_step(f"read_{path}", "read_file", {"path": path})])

    llm = FakeLLM([
        read_plan("a.py"),
        tool_response(tool_call("d1", "read_file", {"path": "b.py"})),
        read_plan("b.py"),
        tool_response(tool_call("d2", "read_file", {"path": "c.py"})),
        read_plan("c.py"),
        tool_response(tool_call("d3", "read_file", {"path": "d.py"})),
    ])
    executor = FakeExecutor()

    result = run_agent(llm, executor)("Explain a file.")

    assert result.status is AgentStatus.PLAN_FAILED
    assert len(result.plan_history) == 3
    assert executor.calls == [("list_files", {"path": "."})]


@pytest.mark.parametrize(
    "invalid_plan",
    [
        "not-json",
        plan_response([planned_step("bad", "missing_tool", {})]),
        plan_response([
            planned_step("duplicate", "read_file", {"path": "a.py"}),
            planned_step("duplicate", "read_file", {"path": "b.py"}),
        ]),
        plan_response([planned_step("bad", "read_file", {"path": "a.py", "extra": 1})]),
        plan_response([
            planned_step(str(index), "read_file", {"path": f"{index}.py"})
            for index in range(MAX_PLAN_STEPS + 1)
        ]),
    ],
)
def test_invalid_plan_is_corrected_on_second_attempt(invalid_plan: str) -> None:
    llm = FakeLLM([invalid_plan, plan_response([]), final_response()])

    result = run_agent(llm)("Explain the project.")

    assert result.status is AgentStatus.COMPLETED
    assert llm.calls[0]["tools"] is None
    assert llm.calls[1]["tools"] is None
    assert any(
        "invalid" in str(message.get("content", "")).lower()
        for message in llm.calls[1]["messages"]
    )


def test_two_invalid_plans_return_plan_failed() -> None:
    llm = FakeLLM(["bad", "still bad"])
    result = run_agent(llm)("Explain the project.")
    assert result.status is AgentStatus.PLAN_FAILED
    assert result.plan_history == []


def test_matching_consecutive_batch_executes_all_calls() -> None:
    steps = [
        planned_step("list", "list_files", {"path": "."}),
        planned_step("read", "read_file", {"path": "a.py"}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(
            tool_call("one", "list_files", {"path": "."}),
            tool_call("two", "read_file", {"path": "a.py"}),
        ),
        final_response(),
    ])
    executor = FakeExecutor([ToolResult(True, "a.py"), ToolResult(True, "content")])

    result = run_agent(llm, executor)("Inspect the workspace.")

    assert result.status is AgentStatus.COMPLETED
    assert [name for name, _ in executor.calls[1:]] == ["list_files", "read_file"]


def test_one_batch_deviation_rejects_entire_batch() -> None:
    initial = [
        planned_step("list", "list_files", {"path": "."}),
        planned_step("read", "read_file", {"path": "a.py"}),
    ]
    llm = FakeLLM([
        plan_response(initial),
        tool_response(
            tool_call("one", "list_files", {"path": "."}),
            tool_call("two", "read_file", {"path": "wrong.py"}),
        ),
        plan_response([]),
        final_response(),
    ])
    executor = FakeExecutor()

    result = run_agent(llm, executor)("Inspect the workspace.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls == [("list_files", {"path": "."})]
    observations = [
        message for message in llm.calls[2]["messages"]
        if message.get("role") == "tool"
    ]
    assert {item["tool_call_id"] for item in observations[-2:]} == {"one", "two"}


def test_failed_tool_keeps_step_pending_until_successful_retry() -> None:
    plan = plan_response([planned_step("read", "read_file", {"path": "a.py"})])
    llm = FakeLLM([
        plan,
        tool_response(tool_call("first", "read_file", {"path": "a.py"})),
        tool_response(tool_call("retry", "read_file", {"path": "a.py"})),
        final_response(),
    ])
    executor = FakeExecutor([
        ToolResult(False, error="missing"), ToolResult(True, "content")
    ])

    result = run_agent(llm, executor)("Explain a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert result.plan_history[0].steps[0].status is PlanStepStatus.COMPLETED
    assert len(executor.calls[1:]) == 2


def test_four_early_final_answers_return_plan_failed_after_three_reminders() -> None:
    llm = FakeLLM([
        plan_response([planned_step("read", "read_file", {"path": "a.py"})]),
        final_response(), final_response(), final_response(), final_response(),
    ])

    result = run_agent(llm, max_plan_completion_reminders=3)("Explain a.py.")

    assert result.status is AgentStatus.PLAN_FAILED
    assert result.plan_history[0].steps[0].status is PlanStepStatus.PENDING


def test_verification_before_write_becomes_stale_until_final_verification() -> None:
    steps = [
        planned_step("verify_old", "execute_command", {"command": ["pytest"]}),
        planned_step("write", "write_file", {"path": "a.py"}),
        planned_step("verify_new", "execute_command", {"command": ["pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(
            tool_call("v1", "execute_command", {"command": ["pytest"]}),
            tool_call("w", "write_file", {"path": "a.py", "content": "x"}),
        ),
        final_response("Too early."),
        tool_response(tool_call("v2", "execute_command", {"command": ["pytest"]})),
        final_response("Verified."),
    ])
    executor = FakeExecutor([
        ToolResult(True, "passed"), ToolResult(True, "wrote"), ToolResult(True, "passed")
    ])

    result = run_agent(llm, executor)("Fix a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert [item.tier for item in result.verification_evidence] == [
        VerificationTier.ORIGINAL
    ]


def test_failed_builder_command_is_not_verification_evidence() -> None:
    steps = [planned_step("verify", "execute_command", {"command": ["pytest"]})]
    llm = FakeLLM([
        plan_response(steps),
        tool_response(tool_call("v", "execute_command", {"command": ["pytest"]})),
    ])
    executor = FakeExecutor([ToolResult(False, "exit_code: 1", "failed")])

    result = run_agent(llm, executor, max_steps=1)("Run the tests.")

    assert result.status is AgentStatus.MAX_STEPS_REACHED
    assert result.verification_evidence == []
    assert result.plan_history[0].steps[0].status is PlanStepStatus.PENDING


def test_failed_optional_builder_command_retries_without_special_replan() -> None:
    initial = [
        planned_step("implementation", "write_file", {"path": "floyd.py"}),
        planned_step("optional_check", "execute_command", {"command": ["pytest"]}),
    ]
    llm = FakeLLM([
        plan_response(initial),
        tool_response(
            tool_call("write-source", "write_file", {
                "path": "floyd.py", "content": "def floyd(): pass"
            }),
            tool_call("no-tests", "execute_command", {"command": ["pytest"]}),
        ),
        tool_response(tool_call("passing-tests", "execute_command", {"command": ["pytest"]})),
        final_response("Floyd is implemented and tested."),
    ])
    executor = FakeExecutor([
        ToolResult(True, "wrote source"),
        ToolResult(False, "collected 0 items", "exit status 5"),
        ToolResult(True, "1 passed"),
    ])

    result = run_agent(llm, executor)("Write a Floyd algorithm.")

    assert result.status is AgentStatus.COMPLETED
    assert [plan.revision for plan in result.plan_history] == [0]
    assert result.verification_evidence[-1].tier is VerificationTier.ORIGINAL
    assert [name for name, _arguments in executor.calls[1:]] == [
        "write_file", "execute_command", "execute_command"
    ]


def test_max_steps_and_fatal_errors_remain_distinct() -> None:
    llm = FakeLLM([
        plan_response([planned_step("read", "read_file", {"path": "a.py"})]),
        tool_response(tool_call("read", "read_file", {"path": "a.py"})),
    ])
    result = run_agent(llm, FakeExecutor([ToolResult(True, "content")]), max_steps=1)(
        "Explain a.py."
    )
    assert result.status is AgentStatus.MAX_STEPS_REACHED

    fatal = run_agent(FakeLLM([RuntimeError("network down")]))("Explain this project.")
    assert fatal.status is AgentStatus.FATAL_ERROR

    execution_fatal = run_agent(FakeLLM([
        plan_response([]), RuntimeError("protocol failure")
    ]))("Explain this project.")
    assert execution_fatal.status is AgentStatus.FATAL_ERROR

    malformed = run_agent(FakeLLM([
        plan_response([]), {"role": "user", "content": "wrong role"}
    ]))("Explain this project.")
    assert malformed.status is AgentStatus.FATAL_ERROR


def test_invalid_tool_json_is_observed_without_executor_call() -> None:
    plan = plan_response([planned_step("read", "read_file", {"path": "a.py"})])
    llm = FakeLLM([
        plan,
        tool_response(tool_call("bad", "read_file", "{not-json")),
        tool_response(tool_call("good", "read_file", {"path": "a.py"})),
        final_response(),
    ])
    executor = FakeExecutor([ToolResult(True, "content")])

    result = run_agent(llm, executor)("Explain a.py.")

    assert result.status is AgentStatus.COMPLETED
    assert executor.calls[1:] == [("read_file", {"path": "a.py"})]
    observation = json.loads(llm.calls[2]["messages"][-1]["content"])
    assert observation["success"] is False
    assert "Invalid JSON" in observation["error"]


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Explain calculator.py.", False),
        ("Fix the calculator bug.", True),
        ("Create a module.", True),
        ("解释这个项目。", False),
        ("实现最短路径算法。", True),
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
        (["pytest", "-v"], False),
        ("pytest", False),
    ],
)
def test_verification_command_rules(command: Any, expected: bool) -> None:
    assert is_verification_command(command) is expected
