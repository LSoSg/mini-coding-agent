"""Unit tests for the v0.6 program-maintained working memory."""

from planning import PlanStep, PlanStepStatus, TaskPlan
from working_memory import (
    MemoryVerificationState,
    WorkingMemory,
    extract_explicit_constraints,
    messages_with_working_memory,
)


def test_extracts_only_explicit_user_constraints() -> None:
    task = """修改 parser.py。
必须保留 parse() 的公开签名。
不要修改测试。
最后解释结果。
"""

    assert extract_explicit_constraints(task) == [
        "必须保留 parse() 的公开签名。",
        "不要修改测试。",
    ]


def test_tracks_plan_reads_writes_and_verification() -> None:
    memory = WorkingMemory.from_task("Modify a.py", verification_required=True)
    assert memory.verification_state is MemoryVerificationState.NOT_REQUIRED
    step = PlanStep(
        id="read_a",
        description="Read the target module",
        tool="read_file",
        argument_constraints={"path": "a.py"},
        rationale="Needed for the task",
    )
    plan = TaskPlan("Fix a.py", ["Tests pass"], [step], revision=0)

    memory.accept_plan(plan)
    memory.record_read("a.py", purpose=step.description)
    assert memory.was_read_in_current_revision("a.py")

    step.status = PlanStepStatus.COMPLETED
    memory.complete_step(step, plan)
    memory.record_write("a.py", workspace_revision=1)
    assert not memory.was_read_in_current_revision("a.py")
    assert memory.verification_state is MemoryVerificationState.NOT_REQUIRED


def test_memory_injection_does_not_mutate_or_hide_latest_observation() -> None:
    memory = WorkingMemory.from_task("Inspect a.py", verification_required=False)
    messages = [
        {"role": "system", "content": "base"},
        {"role": "tool", "tool_call_id": "call", "content": "observation"},
    ]

    injected = messages_with_working_memory(messages, memory)

    assert len(messages) == 2
    assert injected[1]["role"] == "system"
    assert "WORKING MEMORY" in injected[1]["content"]
    assert injected[-1] == messages[-1]
