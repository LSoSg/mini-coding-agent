"""Subprocess execution without command safety policy."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    """Raw result returned by a local subprocess execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    exception: str | None = None


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class LocalEnvironment:
    """Execute a pre-validated argument list and capture its raw result."""

    def run(
        self,
        command: list[str],
        cwd: Path,
        timeout: int | float,
    ) -> ExecutionResult:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                timed_out=True,
            )
        except Exception as exc:
            return ExecutionResult(exception=str(exc))

        return ExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )
