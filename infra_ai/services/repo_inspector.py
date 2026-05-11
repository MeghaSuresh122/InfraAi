"""Build repo tree inventory and folder-level summaries from a local root path."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ignore heavy / irrelevant dirs when walking
_SKIP_DIR_NAMES = {
    ".git",
    ".terraform",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
}


def classify_path(rel: str) -> str:
    lower = rel.lower()
    if lower.endswith((".tf", ".tfvars")) or lower.endswith(".tf.json"):
        return "terraform"
    if lower.endswith((".yaml", ".yml")):
        return "kubernetes_yaml"
    if lower.endswith(("chart.yaml", "charts.yaml")):
        return "helm"
    if "kustomization" in os.path.basename(lower) and lower.endswith((".yaml", ".yml")):
        return "kustomize"
    return "other"


def build_tree_and_summary(repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    """
    Returns (repo_tree_paths sorted, repo_summary dict).

    repo_summary contains:
    - folder_file_counts: rel_dir -> {suffix -> count}
    - top_level_folders: list of immediate child dir names
    - file_types: global counts by classify_path category
    """
    repo_root = repo_root.resolve()
    paths: list[str] = []
    folder_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    file_types: dict[str, int] = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        rel_dir = os.path.relpath(dirpath, repo_root)
        if rel_dir == ".":
            rel_dir = ""
        for fn in filenames:
            full = Path(dirpath) / fn
            try:
                rel = str(full.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                continue
            paths.append(rel)
            cat = classify_path(rel)
            file_types[cat] += 1
            parent = str(Path(rel).parent).replace("\\", "/") if Path(rel).parent != Path(".") else ""
            folder_counts[parent or "."][cat] += 1

    paths.sort()
    top_level = sorted(
        {p.split("/")[0] for p in paths if "/" in p} | {d.name for d in repo_root.iterdir() if d.is_dir()}
    )

    summary: dict[str, Any] = {
        "folder_module_hints": _folder_hints(paths),
        "folder_file_counts": {k: dict(v) for k, v in sorted(folder_counts.items())},
        "top_level_folders": top_level,
        "file_types": dict(file_types),
        "root": str(repo_root),
    }
    return paths, summary


def _folder_hints(paths: list[str]) -> list[dict[str, Any]]:
    """Lightweight folder descriptions for LLM context."""
    hints: list[dict[str, Any]] = []
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        parent = str(Path(p).parent).replace("\\", "/")
        if parent == ".":
            parent = ""
        by_dir[parent].append(p)
    for folder in sorted(by_dir.keys())[:80]:
        files = by_dir[folder][:30]
        hints.append(
            {
                "folder": folder or ".",
                "sample_files": files,
                "count": len(by_dir[folder]),
            }
        )
    return hints
