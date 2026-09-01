"""Internal subprocess worker for collecting and running snapshot tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


class _CollectedFilePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: set[str] = set()

    def pytest_collection_finish(self, session: Any) -> None:
        for item in session.items:
            try:
                relative = Path(item.path).resolve().relative_to(self.root)
            except (AttributeError, OSError, RuntimeError, ValueError):
                continue
            self.files.add(relative.as_posix())


def _config_argument(root: Path) -> str:
    for name in (
        "pytest.ini",
        ".pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
    ):
        candidate = root / name
        if candidate.is_file():
            return str(candidate)
    return os.devnull


def _pytest_base_arguments(root: Path) -> list[str]:
    return [
        "--rootdir",
        str(root),
        "--confcutdir",
        str(root),
        "-c",
        _config_argument(root),
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(root.parent / ".agent_pytest_tmp"),
    ]


def _collect(root: Path, result_path: Path) -> int:
    plugin = _CollectedFilePlugin(root)
    exit_code = pytest.main(
        [*_pytest_base_arguments(root), "--collect-only", "-q"],
        plugins=[plugin],
    )
    result_path.write_text(
        json.dumps(
            {"exit_code": int(exit_code), "test_files": sorted(plugin.files)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


def _run(root: Path, targets: list[str]) -> int:
    return int(pytest.main([*_pytest_base_arguments(root), *targets]))


def main() -> int:
    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    root = Path(sys.argv[2]).resolve(strict=True)
    if not root.is_dir():
        return 2

    os.chdir(root)
    sys.path.insert(0, str(root))

    if mode == "collect" and len(sys.argv) == 4:
        return _collect(root, Path(sys.argv[3]))
    if mode == "run" and len(sys.argv) >= 4:
        return _run(root, sys.argv[3:])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
