"""Workspace snapshots and independent regression against original tests."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from local_environment import ExecutionResult, LocalEnvironment


SNAPSHOT_TIMEOUT = 30
MAX_SNAPSHOT_OUTPUT_CHARS = 20_000
PYTEST_CONFIG_NAMES = (
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


class SnapshotError(RuntimeError):
    """Raised when the snapshot or external runner cannot operate safely."""


@dataclass(frozen=True)
class OriginalTestDiscovery:
    available: bool
    test_files: tuple[str, ...] = ()
    collection_output: str = ""
    error: str | None = None

    @property
    def has_tests(self) -> bool:
        return self.available and bool(self.test_files)


@dataclass(frozen=True)
class OriginalTestRun:
    success: bool
    command: str
    output: str = ""
    error: str | None = None
    timed_out: bool = False


def _truncate(text: str, limit: int = MAX_SNAPSHOT_OUTPUT_CHARS) -> str:
    marker = "\n[output truncated]"
    if len(text) <= limit:
        return text
    return text[: limit - len(marker)] + marker


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name.casefold() in IGNORED_DIRECTORIES
        or name.casefold().endswith((".pyc", ".pyo"))
    }


def _copy_workspace(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=_copy_ignore,
        symlinks=True,
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _format_execution(result: ExecutionResult) -> str:
    sections: list[str] = []
    if result.exit_code is not None:
        sections.append(f"exit_code: {result.exit_code}")
    if result.stdout:
        sections.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        sections.append(f"stderr:\n{result.stderr.rstrip()}")
    return "\n".join(sections)


class WorkspaceSnapshot:
    """Keep an immutable copy of the initial workspace for one agent run."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        timeout: int = SNAPSHOT_TIMEOUT,
        environment_factory: Callable[[], LocalEnvironment] = LocalEnvironment,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.timeout = timeout
        self.environment_factory = environment_factory
        self.discovery = OriginalTestDiscovery(
            available=False, error="Workspace snapshot has not been captured."
        )
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._temporary_root: Path | None = None
        self._snapshot_root: Path | None = None

    @property
    def snapshot_root(self) -> Path:
        if self._snapshot_root is None:
            raise SnapshotError("Workspace snapshot is unavailable.")
        return self._snapshot_root

    @property
    def temporary_root(self) -> Path:
        if self._temporary_root is None:
            raise SnapshotError("Workspace snapshot temporary directory is unavailable.")
        return self._temporary_root

    def capture(self) -> None:
        try:
            root = self.workspace_root.resolve(strict=True)
            if not root.is_dir():
                raise SnapshotError("Workspace root is not a directory.")
            self._temporary = tempfile.TemporaryDirectory(prefix="coding-agent-snapshot-")
            self._temporary_root = Path(self._temporary.name)
            self._snapshot_root = self._temporary_root / "original"
            _copy_workspace(root, self._snapshot_root)
        except SnapshotError:
            self.cleanup()
            raise
        except (OSError, RuntimeError, shutil.Error) as exc:
            self.cleanup()
            raise SnapshotError(f"Unable to capture workspace snapshot: {exc}") from exc

        self.discovery = self._discover_original_tests()

    def cleanup(self) -> None:
        temporary = self._temporary
        self._temporary = None
        self._temporary_root = None
        self._snapshot_root = None
        if temporary is not None:
            try:
                temporary.cleanup()
            except OSError:
                pass

    def _sanitize_output(self, text: str) -> str:
        sanitized = text
        for path in (self._temporary_root, self._snapshot_root):
            if path is None:
                continue
            rendered = str(path)
            sanitized = sanitized.replace(rendered, "<snapshot-workspace>")
            sanitized = sanitized.replace(rendered.replace("\\", "/"), "<snapshot-workspace>")
        return _truncate(sanitized)

    def _worker_path(self) -> Path:
        worker = Path(__file__).resolve().with_name("pytest_snapshot_worker.py")
        if not worker.is_file():
            raise SnapshotError("Internal pytest snapshot worker is missing.")
        return worker

    def _discover_original_tests(self) -> OriginalTestDiscovery:
        result_path = self.temporary_root / "collection.json"
        command = [
            sys.executable,
            str(self._worker_path()),
            "collect",
            str(self.snapshot_root),
            str(result_path),
        ]
        result = self.environment_factory().run(
            command=command,
            cwd=self.snapshot_root,
            timeout=self.timeout,
        )
        output = self._sanitize_output(_format_execution(result))
        if result.timed_out:
            return OriginalTestDiscovery(
                available=False,
                collection_output=output,
                error=f"Original test collection timed out after {self.timeout} seconds.",
            )
        if result.exception is not None or result.exit_code != 0 or not result_path.is_file():
            return OriginalTestDiscovery(
                available=False,
                collection_output=output,
                error=result.exception or "Original test collection worker failed.",
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            collection_exit = payload["exit_code"]
            raw_files = payload["test_files"]
            if isinstance(collection_exit, bool) or not isinstance(collection_exit, int):
                raise ValueError("invalid collection exit code")
            if not isinstance(raw_files, list) or any(
                not isinstance(path, str) for path in raw_files
            ):
                raise ValueError("invalid collected file list")
            test_files = self._validated_test_paths(raw_files)
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            return OriginalTestDiscovery(
                available=False,
                collection_output=output,
                error=f"Original test collection result was invalid: {exc}",
            )

        if collection_exit not in (0, 5):
            return OriginalTestDiscovery(
                available=False,
                collection_output=output,
                error=f"Original test collection failed with exit code {collection_exit}.",
            )
        return OriginalTestDiscovery(
            available=True,
            test_files=tuple(test_files),
            collection_output=output,
        )

    def _validated_test_paths(self, raw_files: list[str]) -> list[str]:
        valid: set[str] = set()
        root = self.snapshot_root.resolve(strict=True)
        for value in raw_files:
            relative = Path(value)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("collected test path escaped the snapshot")
            candidate = (root / relative).resolve(strict=True)
            candidate.relative_to(root)
            if not candidate.is_file():
                raise ValueError("collected test path is not a file")
            valid.add(candidate.relative_to(root).as_posix())
        return sorted(valid)

    def run_original_tests(self) -> OriginalTestRun:
        if not self.discovery.has_tests:
            raise SnapshotError("No original tests are available to run.")
        try:
            final_root = self.workspace_root.resolve(strict=True)
            verification_root = self.temporary_root / "verification"
            if verification_root.exists():
                shutil.rmtree(verification_root)
            _copy_workspace(final_root, verification_root)
            self._restore_original_test_assets(verification_root)
        except (OSError, RuntimeError, ValueError, shutil.Error) as exc:
            raise SnapshotError(f"Unable to prepare original test regression: {exc}") from exc

        targets = [str(verification_root / path) for path in self.discovery.test_files]
        command = [
            sys.executable,
            str(self._worker_path()),
            "run",
            str(verification_root),
            *targets,
        ]
        result = self.environment_factory().run(
            command=command,
            cwd=verification_root,
            timeout=self.timeout,
        )
        output = self._sanitize_output(_format_execution(result))
        display_command = "python -m pytest <original tests>"
        if result.exception is not None:
            raise SnapshotError(f"Original test runner failed: {result.exception}")
        if result.timed_out:
            return OriginalTestRun(
                success=False,
                command=display_command,
                output=output,
                error=f"Original tests timed out after {self.timeout} seconds.",
                timed_out=True,
            )
        success = result.exit_code == 0
        return OriginalTestRun(
            success=success,
            command=display_command,
            output=output,
            error=None if success else f"Original tests exited with status {result.exit_code}.",
        )

    def _restore_original_test_assets(self, verification_root: Path) -> None:
        for config_name in PYTEST_CONFIG_NAMES:
            destination = verification_root / config_name
            if destination.exists() or destination.is_symlink():
                _remove_path(destination)
        for conftest in list(verification_root.rglob("conftest.py")):
            if conftest.is_file() or conftest.is_symlink():
                conftest.unlink()

        test_directories = self._original_test_directories()
        for relative_directory in sorted(test_directories, key=lambda path: len(path.parts)):
            source = self.snapshot_root / relative_directory
            destination = verification_root / relative_directory
            if destination.exists() or destination.is_symlink():
                _remove_path(destination)
            shutil.copytree(source, destination, symlinks=True)

        assets = self._individual_original_assets(test_directories)
        for relative_file in assets:
            source = self.snapshot_root / relative_file
            destination = verification_root / relative_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)

    def _original_test_directories(self) -> set[Path]:
        directories: set[Path] = set()
        for test_file in self.discovery.test_files:
            parts = Path(test_file).parts[:-1]
            for index, part in enumerate(parts):
                if part.casefold() in {"test", "tests"}:
                    directories.add(Path(*parts[: index + 1]))
                    break
        return directories

    def _individual_original_assets(self, restored_directories: set[Path]) -> set[Path]:
        assets: set[Path] = set()
        for config_name in PYTEST_CONFIG_NAMES:
            candidate = self.snapshot_root / config_name
            if candidate.is_file():
                assets.add(Path(config_name))

        for test_file_text in self.discovery.test_files:
            test_file = Path(test_file_text)
            parent = test_file.parent
            while True:
                for name in ("conftest.py", "__init__.py"):
                    candidate = self.snapshot_root / parent / name
                    if candidate.is_file():
                        assets.add(parent / name)
                if parent == Path("."):
                    break
                parent = parent.parent
            if any(test_file.is_relative_to(directory) for directory in restored_directories):
                continue
            assets.add(test_file)
        return assets
