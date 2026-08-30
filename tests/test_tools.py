"""Unit tests for the v0.2 local tool layer."""

import shutil
import subprocess
from pathlib import Path

import pytest

import tools
from local_environment import ExecutionResult, LocalEnvironment


def test_default_workspace_is_dedicated_directory() -> None:
    expected = Path(tools.__file__).resolve().parent / "workspace"

    assert tools.WORKSPACE_ROOT == expected


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tools, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


def test_list_read_write_and_search(workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text(
        "print('needle')\n", encoding="utf-8"
    )

    listed = tools.list_files("src")
    assert listed.success
    assert "[FILE] app.py" in listed.output

    read = tools.read_file("src/app.py")
    assert read.success
    assert read.output == "print('needle')\n"

    written = tools.write_file("src/new.txt", "hello needle")
    assert written.success
    assert "Wrote 12 characters" in written.output
    assert (workspace / "src" / "new.txt").read_text(encoding="utf-8") == "hello needle"

    searched = tools.search_files("needle")
    assert searched.success
    assert "src/app.py:1:" in searched.output
    assert "src/new.txt:1:" in searched.output


@pytest.mark.parametrize(
    "tool_call",
    [
        lambda path: tools.list_files(path),
        lambda path: tools.search_files("needle", path),
        lambda path: tools.read_file(path),
        lambda path: tools.write_file(path, "data"),
    ],
)
def test_absolute_workspace_external_paths_are_rejected(
    workspace: Path, tool_call
) -> None:
    outside = workspace.parent / f"{workspace.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")

    result = tool_call(str(outside))

    assert not result.success
    assert "Absolute paths" in result.error


@pytest.mark.parametrize(
    "tool_call",
    [
        lambda: tools.list_files("../outside"),
        lambda: tools.search_files("needle", "../outside"),
        lambda: tools.read_file("../outside.txt"),
        lambda: tools.write_file("../outside.txt", "data"),
    ],
)
def test_path_traversal_is_rejected(workspace: Path, tool_call) -> None:
    result = tool_call()

    assert not result.success
    assert "traversal" in result.error


def test_external_symbolic_link_is_rejected(workspace: Path) -> None:
    outside = workspace.parent / f"{workspace.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable on this platform: {exc}")

    read = tools.read_file("link.txt")
    write = tools.write_file("link.txt", "replacement")

    assert not read.success
    assert not write.success
    assert "Symbolic links" in read.error
    assert "Symbolic links" in write.error
    assert outside.read_text(encoding="utf-8") == "secret"


def test_missing_paths_return_errors(workspace: Path) -> None:
    assert not tools.list_files("missing").success
    assert not tools.search_files("text", "missing").success
    assert not tools.read_file("missing.txt").success


def test_file_cannot_be_listed_or_searched_as_directory(workspace: Path) -> None:
    (workspace / "file.txt").write_text("text", encoding="utf-8")

    listed = tools.list_files("file.txt")
    searched = tools.search_files("text", "file.txt")

    assert not listed.success
    assert "not a directory" in listed.error
    assert not searched.success
    assert "not a directory" in searched.error


def test_invalid_utf8_read_fails_and_search_skips_file(workspace: Path) -> None:
    (workspace / "invalid.txt").write_bytes(b"needle\xff")
    (workspace / "valid.txt").write_text("needle", encoding="utf-8")

    read = tools.read_file("invalid.txt")
    searched = tools.search_files("needle")

    assert not read.success
    assert "UTF-8" in read.error
    assert searched.success
    assert "valid.txt:1:" in searched.output
    assert "invalid.txt" not in searched.output


def test_large_file_read_is_truncated(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "MAX_FILE_CHARS", 10)
    (workspace / "large.txt").write_text("x" * 20, encoding="utf-8")

    result = tools.read_file("large.txt")

    assert result.success
    assert result.output.startswith("x" * 10)
    assert result.output.endswith("[output truncated]")


def test_search_result_count_is_limited(workspace: Path) -> None:
    (workspace / "many.txt").write_text(
        "\n".join(f"needle {index}" for index in range(10)), encoding="utf-8"
    )

    result = tools.search_files("needle", max_results=3)

    assert result.success
    assert result.output.count("many.txt:") == 3
    assert "[results truncated]" in result.output


def test_search_output_length_is_limited(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "MAX_TOOL_OUTPUT_CHARS", 80)
    (workspace / "long.txt").write_text(
        "\n".join("needle " + "x" * 200 for _ in range(5)), encoding="utf-8"
    )

    result = tools.search_files("needle")

    assert result.success
    assert len(result.output) <= 80
    assert result.output.endswith("[output truncated]")


def test_search_skips_ignored_directories(workspace: Path) -> None:
    for directory in (".git", ".venv", "venv", "node_modules", "__pycache__"):
        ignored = workspace / directory
        ignored.mkdir()
        (ignored / "hidden.py").write_text("needle", encoding="utf-8")

    result = tools.search_files("needle")

    assert result.success
    assert result.output == "No matches found."


def test_write_file_creates_parent_directories(workspace: Path) -> None:
    result = tools.write_file("one/two/file.txt", "content")

    assert result.success
    assert (workspace / "one" / "two" / "file.txt").read_text(
        encoding="utf-8"
    ) == "content"


def test_write_file_rejects_directory_target(workspace: Path) -> None:
    (workspace / "directory").mkdir()

    result = tools.write_file("directory", "content")

    assert not result.success
    assert "not a regular file" in result.error


def test_python_script_executes_in_workspace(workspace: Path) -> None:
    (workspace / "hello.py").write_text(
        "from pathlib import Path\nprint(Path.cwd().name)\n", encoding="utf-8"
    )

    result = tools.execute_command(["python", "hello.py"])

    assert result.success
    assert "exit_code: 0" in result.output
    assert workspace.name in result.output


@pytest.mark.parametrize("command", [["python", "-m", "pytest"], ["pytest"]])
def test_pytest_commands_execute(workspace: Path, command: list[str]) -> None:
    (workspace / "test_sample.py").write_text(
        "def test_sample():\n    assert 2 + 2 == 4\n", encoding="utf-8"
    )

    result = tools.execute_command(command)

    assert result.success, result.error
    assert "1 passed" in result.output


def test_git_status_and_diff_execute(workspace: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    subprocess.run(
        [git, "init", "--quiet"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    status = tools.execute_command(["git", "status"])
    diff = tools.execute_command(["git", "diff"])

    assert status.success, status.error
    assert diff.success, diff.error


def test_git_does_not_discover_parent_repository(workspace: Path) -> None:
    result = tools.execute_command(["git", "status"])

    assert not result.success
    assert "its own .git directory" in result.error


def test_non_zero_exit_code_is_an_error(workspace: Path) -> None:
    (workspace / "fail.py").write_text(
        "import sys\nprint('failed')\nsys.exit(3)\n", encoding="utf-8"
    )

    result = tools.execute_command(["python", "fail.py"])

    assert not result.success
    assert "non-zero status 3" in result.error
    assert "exit_code: 3" in result.output
    assert "failed" in result.output


def test_command_timeout_is_an_error(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "COMMAND_TIMEOUT", 0.05)
    (workspace / "slow.py").write_text(
        "import time\ntime.sleep(1)\n", encoding="utf-8"
    )

    result = tools.execute_command(["python", "slow.py"])

    assert not result.success
    assert "timed out" in result.error


@pytest.mark.parametrize(
    "command",
    [
        ["rm", "file.txt"],
        ["python", "safe.py", ";", "rm"],
        ["git", "status&&git", "diff"],
        ["python", "-c", "print('unsafe')"],
    ],
)
def test_dangerous_commands_are_rejected(workspace: Path, command: list[str]) -> None:
    result = tools.execute_command(command)

    assert not result.success


def test_command_path_outside_workspace_is_rejected(workspace: Path) -> None:
    outside = workspace.parent / f"{workspace.name}-outside.py"
    outside.write_text("print('outside')", encoding="utf-8")

    absolute = tools.execute_command(["python", str(outside)])
    traversal = tools.execute_command(["python", "../outside.py"])

    assert not absolute.success
    assert "Absolute paths" in absolute.error
    assert not traversal.success
    assert "traversal" in traversal.error


def test_execution_exception_is_an_error(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "ok.py").write_text("print('ok')", encoding="utf-8")
    monkeypatch.setattr(
        LocalEnvironment,
        "run",
        lambda self, command, cwd, timeout: ExecutionResult(exception="boom"),
    )

    result = tools.execute_command(["python", "ok.py"])

    assert not result.success
    assert "execution failed" in result.error


def test_execute_tool_unknown_name(workspace: Path) -> None:
    result = tools.execute_tool("missing", {})

    assert not result.success
    assert "Unknown tool" in result.error


def test_execute_tool_invalid_arguments(workspace: Path) -> None:
    wrong_type = tools.execute_tool("list_files", [])
    wrong_name = tools.execute_tool("list_files", {"unexpected": True})

    assert not wrong_type.success
    assert "mapping" in wrong_type.error
    assert not wrong_name.success
    assert "Invalid arguments" in wrong_name.error


def test_execute_tool_contains_internal_exceptions(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_tool() -> tools.ToolResult:
        raise KeyError("internal bug")

    monkeypatch.setitem(tools.TOOL_REGISTRY, "broken", broken_tool)

    result = tools.execute_tool("broken", {})

    assert not result.success
    assert "Unexpected error" in result.error
