from __future__ import annotations

from uuid import uuid4
from pathlib import Path

import _pytest.pathlib
import _pytest.tmpdir
import pytest


def pytest_configure() -> None:
    original_cleanup_dead_symlinks = _pytest.pathlib.cleanup_dead_symlinks

    def _safe_cleanup_dead_symlinks(root) -> None:
        try:
            original_cleanup_dead_symlinks(root)
        except PermissionError:
            # Windows sandbox runs can leave basetemp unreadable during teardown.
            pass

    _pytest.pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
    _pytest.tmpdir.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks


@pytest.fixture
def tmp_path() -> Path:
    tmp_root = Path(__file__).resolve().parents[1] / ".pytest_fixture_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    fixture_path = tmp_root / uuid4().hex
    fixture_path.mkdir(parents=True, exist_ok=False)
    yield fixture_path
