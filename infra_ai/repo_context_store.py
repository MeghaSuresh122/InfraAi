"""SQLite-backed canonical repo context snapshots (hybrid cache with Milvus)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from infra_ai.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "repo_context_cache.sqlite"


def _db_path() -> Path:
    import os

    p = os.environ.get("REPO_CONTEXT_SQLITE_PATH", "").strip()
    return Path(p) if p else _DEFAULT_DB


def _conn() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_repo_context_store() -> None:
    c = _conn()
    try:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS repo_context_snapshots (
                repo_key TEXT NOT NULL,
                base_branch TEXT NOT NULL,
                head_commit TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (repo_key, base_branch)
            )
            """
        )
        c.commit()
    finally:
        c.close()


def load_snapshot(repo_key: str, base_branch: str) -> dict[str, Any] | None:
    init_repo_context_store()
    c = _conn()
    try:
        row = c.execute(
            "SELECT head_commit, snapshot_json, updated_at FROM repo_context_snapshots "
            "WHERE repo_key = ? AND base_branch = ?",
            (repo_key, base_branch),
        ).fetchone()
    finally:
        c.close()
    if not row:
        return None
    try:
        snap = json.loads(row["snapshot_json"])
    except json.JSONDecodeError:
        logger.warning("Corrupt snapshot JSON for %s@%s", repo_key, base_branch)
        return None
    return {
        "head_commit": row["head_commit"],
        "snapshot": snap,
        "updated_at": row["updated_at"],
    }


def save_snapshot(repo_key: str, base_branch: str, head_commit: str, snapshot: dict[str, Any]) -> None:
    init_repo_context_store()
    payload = json.dumps(snapshot, ensure_ascii=False)
    now = time.time()
    c = _conn()
    try:
        c.execute(
            """
            INSERT INTO repo_context_snapshots (repo_key, base_branch, head_commit, snapshot_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repo_key, base_branch) DO UPDATE SET
                head_commit = excluded.head_commit,
                snapshot_json = excluded.snapshot_json,
                updated_at = excluded.updated_at
            """,
            (repo_key, base_branch, head_commit, payload, now),
        )
        c.commit()
    finally:
        c.close()


def normalize_repo_key(repo_url: str) -> str:
    u = (repo_url or "").strip().rstrip("/")
    if u.lower().endswith(".git"):
        u = u[:-4]
    return u.lower()
