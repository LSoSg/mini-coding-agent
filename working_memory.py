"""Small, program-maintained working memory for one agent run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from planning import PlanStep, PlanStepStatus, TaskPlan

MAX_MEMORY_CHARS = 8_000
MAX_CONSTRAINTS = 20
MAX_TRACKED_FILES = 50
MAX_TEXT_CHARS = 300


class MemoryVerificationState(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRED = "REQUIRED"
    SELF_FAILED = "SELF_FAILED"
    SELF_PASSED = "SELF_PASSED"
    ORIGINAL_FAILED = "ORIGINAL_FAILED"
    ORIGINAL_PASSED = "ORIGINAL_PASSED"


@dataclass
class FileReadRecord:
    path: str
    workspace_revision: int
    purpose: str
    reads: int = 1


@dataclass
class FileModificationRecord:
    path: str
    workspace_revision: int
    writes: int = 1


def _shorten(value: str, limit: int = MAX_TEXT_CHARS) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "..."


def normalize_workspace_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    return PurePosixPath(normalized).as_posix()


def extract_explicit_constraints(task: str) -> list[str]:
    """Keep explicit user constraints visible without asking the LLM to summarize them."""

    cues = (
        "must", "must not", "only", "do not", "don't", "never", "without",
        "必须", "只能", "不要", "不得", "禁止", "不能", "仅", "保留",
    )
    constraints: list[str] = []
    for raw_line in task.splitlines():
        line = raw_line.strip().lstrip("-*0123456789.、 ")
        lowered = line.casefold()
        if line and any(cue in lowered for cue in cues):
            item = _shorten(line)
            if item not in constraints:
                constraints.append(item)
        if len(constraints) >= MAX_CONSTRAINTS:
            break
    return constraints


@dataclass
class WorkingMemory:
    """Authoritative compact state derived from actual agent events."""

    original_task: str
    goal: str
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    plan_revision: int | None = None
    current_step_id: str | None = None
    current_step_description: str | None = None
    completed_step_ids: list[str] = field(default_factory=list)
    read_files: dict[str, FileReadRecord] = field(default_factory=dict)
    modified_files: dict[str, FileModificationRecord] = field(default_factory=dict)
    workspace_revision: int = 0
    verification_state: MemoryVerificationState = MemoryVerificationState.NOT_REQUIRED
    last_verification_command: str | None = None
    open_issues: list[str] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: str, *, verification_required: bool) -> WorkingMemory:
        return cls(
            original_task=_shorten(task, 2_000),
            goal=_shorten(task, 1_000),
            constraints=extract_explicit_constraints(task),
            verification_state=(
                MemoryVerificationState.REQUIRED
                if verification_required else MemoryVerificationState.NOT_REQUIRED
            ),
        )

    def accept_plan(self, plan: TaskPlan) -> None:
        self.goal = _shorten(plan.goal, 1_000)
        self.success_criteria = [_shorten(item) for item in plan.success_criteria]
        self.plan_revision = plan.revision
        self.completed_step_ids = []
        self.open_issues.clear()
        self._set_next_step(plan)

    def _set_next_step(self, plan: TaskPlan) -> None:
        pending = next(
            (step for step in plan.steps if step.status is PlanStepStatus.PENDING),
            None,
        )
        self.current_step_id = pending.id if pending else None
        self.current_step_description = pending.description if pending else None

    def complete_step(self, step: PlanStep, plan: TaskPlan) -> None:
        if step.id not in self.completed_step_ids:
            self.completed_step_ids.append(step.id)
        self._set_next_step(plan)

    def record_read(self, path: str, *, purpose: str) -> None:
        key = normalize_workspace_path(path)
        previous = self.read_files.get(key)
        reads = previous.reads + 1 if previous else 1
        if len(self.read_files) < MAX_TRACKED_FILES or key in self.read_files:
            self.read_files[key] = FileReadRecord(
                path=key,
                workspace_revision=self.workspace_revision,
                purpose=_shorten(purpose),
                reads=reads,
            )

    def was_read_in_current_revision(self, path: str) -> bool:
        record = self.read_files.get(normalize_workspace_path(path))
        return bool(record and record.workspace_revision == self.workspace_revision)

    def record_write(self, path: str, *, workspace_revision: int) -> None:
        self.workspace_revision = workspace_revision
        key = normalize_workspace_path(path)
        previous = self.modified_files.get(key)
        writes = previous.writes + 1 if previous else 1
        if len(self.modified_files) < MAX_TRACKED_FILES or key in self.modified_files:
            self.modified_files[key] = FileModificationRecord(
                path=key,
                workspace_revision=workspace_revision,
                writes=writes,
            )
        self.verification_state = MemoryVerificationState.REQUIRED

    def record_self_verification(self, command: str, *, success: bool) -> None:
        self.last_verification_command = command
        self.verification_state = (
            MemoryVerificationState.SELF_PASSED
            if success else MemoryVerificationState.SELF_FAILED
        )

    def record_original_verification(self, command: str, *, success: bool) -> None:
        self.last_verification_command = command
        self.verification_state = (
            MemoryVerificationState.ORIGINAL_PASSED
            if success else MemoryVerificationState.ORIGINAL_FAILED
        )

    def add_issue(self, issue: str) -> None:
        item = _shorten(issue, 600)
        if item and item not in self.open_issues:
            self.open_issues.append(item)
            self.open_issues[:] = self.open_issues[-5:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_task": self.original_task,
            "goal": self.goal,
            "success_criteria": self.success_criteria,
            "explicit_constraints": self.constraints,
            "plan_revision": self.plan_revision,
            "current_step": {
                "id": self.current_step_id,
                "description": self.current_step_description,
            },
            "completed_step_ids": self.completed_step_ids,
            "workspace_revision": self.workspace_revision,
            "files_read": [record.__dict__ for record in self.read_files.values()],
            "files_modified": [
                record.__dict__ for record in self.modified_files.values()
            ],
            "verification": {
                "state": self.verification_state.value,
                "last_command": self.last_verification_command,
            },
            "open_issues": self.open_issues,
        }

    def render(self) -> str:
        rendered = json.dumps(self.as_dict(), ensure_ascii=False, indent=2)
        if len(rendered) > MAX_MEMORY_CHARS:
            rendered = rendered[:MAX_MEMORY_CHARS] + "\n[working memory truncated]"
        return rendered


def messages_with_working_memory(
    messages: list[dict[str, Any]], memory: WorkingMemory
) -> list[dict[str, Any]]:
    """Inject one fresh memory view without growing the persistent message history."""

    memory_message = {
        "role": "system",
        "content": (
            "WORKING MEMORY (program-maintained, authoritative):\n"
            f"{memory.render()}\n"
            "Use this state to stay aligned with the global goal and constraints. "
            "Do not repeat a successful read_file in the same workspace revision; "
            "reuse its earlier observation unless a later write made rereading necessary."
        ),
    }
    if not messages:
        return [memory_message]
    return [messages[0], memory_message, *messages[1:]]
