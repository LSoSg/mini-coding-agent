"""Tests for v0.5 workspace snapshots and original-test regression."""

from pathlib import Path

from workspace_snapshot import MAX_SNAPSHOT_OUTPUT_CHARS, WorkspaceSnapshot


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_snapshot_skips_caches_and_cleans_temporary_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "app.py", "VALUE = 1\n")
    write(workspace / "__pycache__" / "app.pyc", "cache")
    write(workspace / ".pytest_cache" / "state", "cache")
    write(workspace / "node_modules" / "package.js", "cache")
    snapshot = WorkspaceSnapshot(workspace)

    snapshot.capture()
    temporary_root = snapshot.temporary_root

    assert (snapshot.snapshot_root / "app.py").is_file()
    assert not (snapshot.snapshot_root / "__pycache__").exists()
    assert not (snapshot.snapshot_root / ".pytest_cache").exists()
    assert not (snapshot.snapshot_root / "node_modules").exists()

    snapshot.cleanup()
    assert not temporary_root.exists()


def test_collection_supports_root_tests_and_custom_pytest_rules(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(
        workspace / "pytest.ini",
        "[pytest]\ntestpaths = specs\npython_files = check_*.py\n",
    )
    write(workspace / "app.py", "VALUE = 1\n")
    write(
        workspace / "specs" / "check_app.py",
        "from app import VALUE\ndef test_value(): assert VALUE == 1\n",
    )
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        assert snapshot.discovery.available
        assert snapshot.discovery.test_files == ("specs/check_app.py",)
    finally:
        snapshot.cleanup()


def test_modified_assertion_is_ignored_by_original_regression(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "app.py", "def add(a, b): return a + b\n")
    write(
        workspace / "test_app.py",
        "from app import add\ndef test_add(): assert add(2, 3) == 5\n",
    )
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        write(workspace / "app.py", "def add(a, b): return a - b\n")
        write(workspace / "test_app.py", "def test_fake(): assert True\n")

        result = snapshot.run_original_tests()

        assert not result.success
        assert "test_add" in result.output
        assert "test_fake" not in result.output
        assert str(snapshot.temporary_root) not in result.output
    finally:
        snapshot.cleanup()


def test_deleted_original_test_is_restored_for_regression(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "app.py", "VALUE = 1\n")
    write(workspace / "test_app.py", "from app import VALUE\ndef test_value(): assert VALUE == 1\n")
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        (workspace / "test_app.py").unlink()
        result = snapshot.run_original_tests()
        assert result.success
        assert "1 passed" in result.output
    finally:
        snapshot.cleanup()


def test_test_directory_helpers_and_conftest_come_from_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "app.py", "def value(): return 5\n")
    write(workspace / "tests" / "helper.py", "EXPECTED = 5\n")
    write(workspace / "tests" / "conftest.py", "ORIGINAL_FIXTURE = True\n")
    write(
        workspace / "tests" / "test_app.py",
        "from app import value\nfrom tests.helper import EXPECTED\n"
        "def test_value(): assert value() == EXPECTED\n",
    )
    write(workspace / "tests" / "__init__.py", "")
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        write(workspace / "app.py", "def value(): return -1\n")
        write(workspace / "tests" / "helper.py", "EXPECTED = -1\n")
        write(workspace / "tests" / "test_app.py", "def test_fake(): assert True\n")

        result = snapshot.run_original_tests()

        assert not result.success
        assert "test_value" in result.output
        assert "test_fake" not in result.output
    finally:
        snapshot.cleanup()


def test_original_pytest_config_replaces_agent_modified_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "pytest.ini", "[pytest]\npython_files = test_*.py\n")
    write(workspace / "app.py", "VALUE = 1\n")
    write(workspace / "test_app.py", "from app import VALUE\ndef test_value(): assert VALUE == 1\n")
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        write(workspace / "pytest.ini", "[pytest]\npython_files = never_*.py\n")
        result = snapshot.run_original_tests()
        assert result.success
        assert "1 passed" in result.output
    finally:
        snapshot.cleanup()


def test_agent_created_tests_are_not_part_of_original_regression(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "app.py", "VALUE = 1\n")
    write(workspace / "test_app.py", "from app import VALUE\ndef test_value(): assert VALUE == 1\n")
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        write(workspace / "test_agent_created.py", "def test_new(): assert False\n")
        result = snapshot.run_original_tests()
        assert result.success
        assert "test_agent_created" not in result.output
        assert "1 passed" in result.output
    finally:
        snapshot.cleanup()


def test_original_runner_uses_its_own_pytest_basetemp(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(
        workspace / "test_temp.py",
        "def test_temp_directory(tmp_path):\n"
        "    (tmp_path / 'value.txt').write_text('ok')\n"
        "    assert (tmp_path / 'value.txt').read_text() == 'ok'\n",
    )
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        result = snapshot.run_original_tests()
        assert result.success
    finally:
        snapshot.cleanup()


def test_collection_error_is_recorded_without_raising(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(workspace / "pytest.ini", "not valid pytest configuration")
    write(workspace / "test_app.py", "def test_value(): assert True\n")
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        assert not snapshot.discovery.available
        assert snapshot.discovery.error
    finally:
        snapshot.cleanup()


def test_original_output_has_a_hard_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write(
        workspace / "test_large.py",
        "def test_large():\n    assert False, 'x' * 30000\n",
    )
    snapshot = WorkspaceSnapshot(workspace)

    try:
        snapshot.capture()
        result = snapshot.run_original_tests()
        assert not result.success
        assert len(result.output) <= MAX_SNAPSHOT_OUTPUT_CHARS
        assert result.output.endswith("[output truncated]")
    finally:
        snapshot.cleanup()
