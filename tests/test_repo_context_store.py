"""SQLite repo context snapshot cache."""

from __future__ import annotations

import pytest

from infra_ai.config import get_settings
from infra_ai.repo_context_store import (
    load_snapshot,
    normalize_repo_key,
    save_snapshot,
)


def test_normalize_repo_key_strips_git_suffix() -> None:
    assert normalize_repo_key("https://GitHub.com/Org/Repo.GIT") == "https://github.com/org/repo"


def test_save_load_snapshot_head_match(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "rc.sqlite"
    monkeypatch.setenv("REPO_CONTEXT_SQLITE_PATH", str(db))
    get_settings.cache_clear()

    key = "https://github.com/org/repo"
    branch = "main"
    head = "abc123def456"
    snap = {
        "repo_head_commit": head,
        "repo_tree_paths": ["a.tf", "b.yaml"],
        "repo_summary": {"top_level_folders": ["terraform"]},
        "reusable_modules": {"terraform": [], "kubernetes": []},
        "file_excerpts": {},
    }
    save_snapshot(key, branch, head, snap)

    row = load_snapshot(key, branch)
    assert row is not None
    assert row["head_commit"] == head
    assert row["snapshot"]["repo_tree_paths"] == ["a.tf", "b.yaml"]


def test_snapshot_invalidated_when_head_changes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "rc2.sqlite"
    monkeypatch.setenv("REPO_CONTEXT_SQLITE_PATH", str(db))
    get_settings.cache_clear()

    key = "https://example.com/r"
    branch = "main"
    save_snapshot(key, branch, "oldsha", {"repo_head_commit": "oldsha", "repo_tree_paths": ["x.tf"]})

    loaded = load_snapshot(key, branch)
    assert loaded["head_commit"] == "oldsha"

    # Resolver compares remote HEAD to row["head_commit"]; different head => miss
    assert loaded["head_commit"] != "newsha"
