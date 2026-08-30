"""Safe local tools for operating inside one workspace."""

import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_environment import ExecutionResult, LocalEnvironment


WORKSPACE_ROOT = Path(__file__).resolve().parent / "workspace"

MAX_FILE_CHARS = 16_000
MAX_SEARCH_RESULTS = 50
MAX_PREVIEW_CHARS = 240
MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_ERROR_CHARS = 2_000
COMMAND_TIMEOUT = 30

SKIPPED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {"dockerfile", "makefile"}
SHELL_METACHARACTERS = ("|", ">", "<", ";", "&", "`", "$(", "\n", "\r")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: str = ""
    error: str | None = None


class ToolInputError(ValueError):
    """Raised internally when a tool request violates its contract."""


def _truncate(text: str, limit: int) -> str:
    marker = "\n[output truncated]"
    if len(text) <= limit:
        return text
    if limit <= len(marker):
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _success(output: str = "") -> ToolResult:
    return ToolResult(success=True, output=_truncate(output, MAX_TOOL_OUTPUT_CHARS))


def _failure(error: str, output: str = "") -> ToolResult:
    return ToolResult(
        success=False,
        output=_truncate(output, MAX_TOOL_OUTPUT_CHARS),
        error=_truncate(error, MAX_ERROR_CHARS),
    )


def _root() -> Path:
    try:
        root = WORKSPACE_ROOT.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolInputError(f"Workspace is unavailable: {exc}") from exc
    if not root.is_dir():
        raise ToolInputError("Workspace root is not a directory.")
    return root


def _contains_reserved_windows_part(path: Path) -> bool:
    for part in path.parts:
        if ":" in part:
            return True
        stem = part.rstrip(" .").split(".", maxsplit=1)[0].upper()
        if stem in WINDOWS_RESERVED_NAMES:
            return True
    return False


def _resolve_workspace_path(path: str | Path) -> tuple[Path, Path]:
    if not isinstance(path, (str, Path)):
        raise ToolInputError("Path must be a string or Path.")

    relative = Path(path)
    if relative.is_absolute() or relative.drive:
        raise ToolInputError("Absolute paths are not allowed.")
    if _contains_reserved_windows_part(relative):
        raise ToolInputError("Reserved system paths are not allowed.")
    if ".." in relative.parts:
        raise ToolInputError("Path traversal is not allowed.")

    root = _root()
    unresolved = root / relative

    current = root
    for part in relative.parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            raise ToolInputError("Symbolic links are not allowed in tool paths.")

    try:
        resolved = unresolved.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolInputError("Path must stay inside the workspace.") from exc

    return resolved, root


def _relative_display(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative or "."


def list_files(path: str = ".") -> ToolResult:
    """List one workspace directory with explicit file/directory labels."""
    try:
        directory, root = _resolve_workspace_path(path)
        if not directory.exists():
            return _failure(f"Path does not exist: {path}")
        if not directory.is_dir():
            return _failure(f"Path is not a directory: {path}")

        entries = sorted(
            directory.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
        lines = [f"Directory: {_relative_display(directory, root)}"]
        for entry in entries:
            if entry.is_symlink():
                kind = "SYMLINK"
                suffix = ""
            elif entry.is_dir():
                kind = "DIR"
                suffix = "/"
            elif entry.is_file():
                kind = "FILE"
                suffix = f" ({entry.stat().st_size} bytes)"
            else:
                kind = "OTHER"
                suffix = ""
            lines.append(f"[{kind}] {entry.name}{suffix}")

        if len(lines) == 1:
            lines.append("[empty directory]")
        return _success("\n".join(lines))
    except (ToolInputError, OSError) as exc:
        return _failure(str(exc))


def _is_searchable_file(path: Path) -> bool:
    return (
        path.suffix.casefold() in TEXT_SUFFIXES
        or path.name.casefold() in TEXT_FILENAMES
    )


def search_files(
    keyword: str,
    path: str = ".",
    max_results: int = MAX_SEARCH_RESULTS,
) -> ToolResult:
    """Recursively search common UTF-8 code and text files."""
    try:
        if not isinstance(keyword, str) or not keyword:
            raise ToolInputError("Keyword must be a non-empty string.")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= MAX_SEARCH_RESULTS
        ):
            raise ToolInputError(
                f"max_results must be between 1 and {MAX_SEARCH_RESULTS}."
            )

        search_root, workspace = _resolve_workspace_path(path)
        if not search_root.exists():
            return _failure(f"Path does not exist: {path}")
        if not search_root.is_dir():
            return _failure(f"Path is not a directory: {path}")

        matches: list[str] = []
        result_limit_reached = False
        folded_keyword = keyword.casefold()

        for current_dir, dirnames, filenames in os.walk(
            search_root, followlinks=False
        ):
            current = Path(current_dir)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name.casefold() not in SKIPPED_DIRECTORIES
                and not (current / name).is_symlink()
            )

            for filename in sorted(filenames):
                candidate = current / filename
                if candidate.is_symlink() or not _is_searchable_file(candidate):
                    continue

                file_matches: list[str] = []
                try:
                    with candidate.open("r", encoding="utf-8", errors="strict") as file:
                        for line_number, line in enumerate(file, start=1):
                            if folded_keyword not in line.casefold():
                                continue
                            if len(matches) + len(file_matches) >= max_results:
                                result_limit_reached = True
                                continue
                            preview = line.strip().replace("\t", " ")
                            if len(preview) > MAX_PREVIEW_CHARS:
                                preview = preview[: MAX_PREVIEW_CHARS - 1] + "…"
                            relative = candidate.relative_to(workspace).as_posix()
                            file_matches.append(
                                f"{relative}:{line_number}: {preview}"
                            )
                except (UnicodeDecodeError, OSError):
                    continue

                matches.extend(file_matches)
                if len(matches) >= max_results:
                    result_limit_reached = True
                    break

            if len(matches) >= max_results:
                break

        if not matches:
            return _success("No matches found.")
        if result_limit_reached:
            matches.append("[results truncated]")
        return _success("\n".join(matches))
    except (ToolInputError, OSError) as exc:
        return _failure(str(exc))


def read_file(path: str) -> ToolResult:
    """Read one UTF-8 regular file from the workspace."""
    try:
        file_path, _ = _resolve_workspace_path(path)
        if not file_path.exists():
            return _failure(f"File does not exist: {path}")
        if not file_path.is_file():
            return _failure(f"Path is not a regular file: {path}")

        with file_path.open("r", encoding="utf-8", errors="strict") as file:
            content = file.read(MAX_FILE_CHARS + 1)
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n[output truncated]"
        return _success(content)
    except UnicodeDecodeError:
        return _failure(f"File is not valid UTF-8 text: {path}")
    except (ToolInputError, OSError) as exc:
        return _failure(str(exc))


def write_file(path: str, content: str) -> ToolResult:
    """Write UTF-8 text to a regular workspace file, creating parents."""
    try:
        if not isinstance(content, str):
            raise ToolInputError("Content must be a string.")

        file_path, root = _resolve_workspace_path(path)
        if file_path.exists() and not file_path.is_file():
            return _failure(f"Path is not a regular file: {path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Re-resolve after directory creation to reduce symlink race exposure.
        file_path, root = _resolve_workspace_path(path)
        if file_path.exists() and not file_path.is_file():
            return _failure(f"Path is not a regular file: {path}")

        file_path.write_text(content, encoding="utf-8", errors="strict")
        byte_count = len(content.encode("utf-8"))
        relative = _relative_display(file_path, root)
        return _success(
            f"Wrote {len(content)} characters ({byte_count} bytes) to {relative}."
        )
    except (ToolInputError, OSError) as exc:
        return _failure(str(exc))


def _contains_shell_syntax(command: list[str]) -> bool:
    return any(
        token in argument
        for argument in command
        for token in SHELL_METACHARACTERS
    )


def _trusted_git_executable(root: Path) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise ToolInputError("git executable was not found.")
    resolved = Path(executable).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        return str(resolved)
    raise ToolInputError("Refusing to execute a git binary from the workspace.")


def _workspace_git_directory(root: Path) -> Path:
    git_directory = root / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise ToolInputError(
            "git commands require the workspace to contain its own .git directory."
        )
    return git_directory.resolve(strict=True)


def _validate_command(command: list[str]) -> tuple[list[str], Path]:
    if not isinstance(command, list) or not command:
        raise ToolInputError("Command must be a non-empty list of strings.")
    if any(not isinstance(argument, str) or not argument for argument in command):
        raise ToolInputError("Every command argument must be a non-empty string.")
    if _contains_shell_syntax(command):
        raise ToolInputError("Shell operators and command chaining are not allowed.")

    root = _root()
    if command[0] == "python":
        if command[1:] == ["-m", "pytest"]:
            return [sys.executable, "-m", "pytest"], root
        if len(command) != 2:
            raise ToolInputError(
                "Only 'python <workspace script.py>' or 'python -m pytest' is allowed."
            )
        script, _ = _resolve_workspace_path(command[1])
        if script.suffix.casefold() != ".py":
            raise ToolInputError("Python may only run a workspace .py file.")
        if not script.exists() or not script.is_file():
            raise ToolInputError(f"Python script does not exist: {command[1]}")
        return [sys.executable, str(script)], root

    if command == ["pytest"]:
        return [sys.executable, "-m", "pytest"], root
    if command in (["git", "status"], ["git", "diff"]):
        git = _trusted_git_executable(root)
        git_directory = _workspace_git_directory(root)
        safe_command = [
            git,
            "--no-pager",
            f"--git-dir={git_directory}",
            f"--work-tree={root}",
            command[1],
        ]
        if command[1] == "diff":
            safe_command.extend(["--no-ext-diff", "--no-textconv"])
        return safe_command, root

    raise ToolInputError("Command is not in the allowed command list.")


def _format_execution(result: ExecutionResult) -> str:
    sections = [f"exit_code: {result.exit_code}"]
    if result.stdout:
        sections.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        sections.append(f"stderr:\n{result.stderr.rstrip()}")
    return "\n".join(sections)


def execute_command(command: list[str]) -> ToolResult:
    """Validate and execute one command from the strict v0.2 allowlist."""
    try:
        safe_command, root = _validate_command(command)
        result = LocalEnvironment().run(
            command=safe_command,
            cwd=root,
            timeout=COMMAND_TIMEOUT,
        )
        output = _format_execution(result)
        if result.timed_out:
            return _failure(
                f"Command timed out after {COMMAND_TIMEOUT} seconds.", output
            )
        if result.exception is not None:
            return _failure(f"Command execution failed: {result.exception}", output)
        if result.exit_code != 0:
            return _failure(
                f"Command exited with non-zero status {result.exit_code}.", output
            )
        return _success(output)
    except (ToolInputError, OSError) as exc:
        return _failure(str(exc))


ToolFunction = Callable[..., ToolResult]

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "list_files": list_files,
    "search_files": search_files,
    "read_file": read_file,
    "write_file": write_file,
    "execute_command": execute_command,
}


def execute_tool(name: str, arguments: Mapping[str, Any]) -> ToolResult:
    """Dispatch one registered tool without allowing errors to escape."""
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return _failure(f"Unknown tool: {name}")
    if not isinstance(arguments, Mapping):
        return _failure("Tool arguments must be a mapping.")

    try:
        result = TOOL_REGISTRY[name](**dict(arguments))
        if not isinstance(result, ToolResult):
            return _failure("Tool returned an invalid result type.")
        return ToolResult(
            success=result.success,
            output=_truncate(result.output, MAX_TOOL_OUTPUT_CHARS),
            error=(
                _truncate(result.error, MAX_ERROR_CHARS)
                if result.error is not None
                else None
            ),
        )
    except TypeError as exc:
        return _failure(f"Invalid arguments for tool '{name}': {exc}")
    except (OSError, RuntimeError, ValueError) as exc:
        return _failure(f"Tool '{name}' failed: {exc}")
    except Exception as exc:  # Last-resort boundary for future registry entries.
        return _failure(f"Unexpected error in tool '{name}': {exc}")
