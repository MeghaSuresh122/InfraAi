"""Repo context pipeline nodes: HEAD resolution, cache, analyze, persist, plan+RAG."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from infra_ai.milvus_store import delete_repo_context_for_branch, upsert_repo_context_chunks
from infra_ai.nodes.context_budget import build_bounded_repo_rag_text
from infra_ai.repo_context_store import (
    load_snapshot,
    normalize_repo_key,
    save_snapshot,
)
from infra_ai.services.change_planner import build_codegen_plan
from infra_ai.services.git_service import GitService, is_remote_git_url, resolve_local_repo_root
from infra_ai.services.module_analyzer import build_reusable_modules
from infra_ai.services.repo_chunks import chunk_text, summary_chunks
from infra_ai.services.repo_inspector import build_tree_and_summary
from infra_ai.services.repo_remote import fetch_github_branch_sha, shallow_clone_branch
from infra_ai.services.relevance_ranker import merge_milvus_ranks, rank_paths
from infra_ai.state import InfraGraphState

logger = logging.getLogger(__name__)


def _local_head(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=45,
            check=True,
        )
        return (out.stdout or "").strip()[:64]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "local"


def _read_snippet(repo_root: Path, rel: str, limit: int = 3500) -> str:
    p = repo_root / rel
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _collect_readmes(repo_root: Path, repo_folder: str, limit_per_file: int = 3000) -> dict[str, str]:
    """Return {relative_path: content} for README files found in:
    - repo root
    - repo_folder itself (if given)
    - every parent directory of repo_folder up to the repo root
    """
    readmes: dict[str, str] = {}

    def _try_dir(directory: Path) -> None:
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.name.upper().startswith("README"):
                rel = entry.relative_to(repo_root).as_posix()
                try:
                    readmes[rel] = entry.read_text(encoding="utf-8", errors="replace")[:limit_per_file]
                except OSError:
                    pass

    # Always check repo root
    if repo_root.is_dir():
        _try_dir(repo_root)

    if repo_folder:
        target = repo_root / repo_folder
        # Walk parents between root and target (exclusive of root, inclusive of target)
        parts = Path(repo_folder).parts
        for i in range(1, len(parts) + 1):
            candidate = repo_root / Path(*parts[:i])
            if candidate.is_dir() and candidate != repo_root:
                _try_dir(candidate)

    return readmes


def _folder_info(repo_root: Path, repo_folder: str) -> tuple[bool, list[str]]:
    """Return (exists, list_of_relative_file_paths_inside_folder)."""
    if not repo_folder:
        return False, []
    target = repo_root / repo_folder
    if not target.is_dir():
        return False, []
    files = [
        f.relative_to(repo_root).as_posix()
        for f in sorted(target.rglob("*"))
        if f.is_file() and "/.git/" not in f.as_posix().replace("\\", "/")
    ]
    return True, files


def repo_context_head_node(state: InfraGraphState) -> dict[str, Any]:
    repo_url = (state.get("repo_url") or "").strip()
    branch = state.get("target_branch") or "main"
    events: list[dict[str, Any]] = [{"node": "repo_context_head", "repo_url_set": bool(repo_url)}]
    if not repo_url:
        return {
            "repo_head_commit": "",
            "repo_context_status": "skipped_no_repo",
            "repo_context_events": events,
        }
    sha = ""
    if is_remote_git_url(repo_url):
        svc = GitService(
            repo_url=repo_url,
            default_branch=branch,
            remote_name=state.get("git_remote_name"),
        )
        sha = fetch_github_branch_sha(repo_url, branch, svc) or ""
    else:
        root = resolve_local_repo_root(repo_url)
        if root.is_dir():
            sha = _local_head(root)
    if not sha:
        return {
            "repo_head_commit": "",
            "repo_context_status": "error",
            "repo_context_events": events + [{"node": "repo_context_head", "error": "no_head_sha"}],
        }
    return {
        "repo_head_commit": sha,
        "repo_context_status": "ok",
        "repo_context_events": events + [{"node": "repo_context_head", "head": sha[:12]}],
    }


def route_after_repo_context_head(state: InfraGraphState) -> str:
    if not (state.get("repo_url") or "").strip():
        return "rc_plan_rag"
    if not state.get("repo_head_commit"):
        return "rc_plan_rag"
    return "rc_resolve"


def repo_context_resolve_node(state: InfraGraphState) -> dict[str, Any]:
    """Try state reuse or SQLite cache for same HEAD; set rc_skip_analyze flag via events only."""
    repo_url = (state.get("repo_url") or "").strip()
    branch = state.get("target_branch") or "main"
    head = state.get("repo_head_commit") or ""
    rc = state.get("repo_context") or {}
    events: list[dict[str, Any]] = []

    if rc and rc.get("repo_head_commit") == head and (rc.get("repo_tree_paths") or []):
        return {
            "repo_context_source": "state_reuse",
            "repo_context_events": events
            + [{"node": "repo_context_resolve", "source": "state_reuse", "head": head[:12]}],
        }

    key = normalize_repo_key(repo_url)
    row = load_snapshot(key, branch)
    if row and row["head_commit"] == head:
        snap = row["snapshot"]
        return {
            "repo_context": snap,
            "repo_context_source": "cache_sqlite",
            "repo_context_events": events
            + [{"node": "repo_context_resolve", "source": "cache_sqlite", "head": head[:12]}],
        }

    return {
        "repo_context_source": "pending_fresh",
        "repo_context_events": events + [{"node": "repo_context_resolve", "source": "needs_analyze"}],
    }


def route_after_repo_context_resolve(state: InfraGraphState) -> str:
    if not state.get("repo_head_commit"):
        return "rc_plan_rag"
    src = state.get("repo_context_source")
    if src in ("state_reuse", "cache_sqlite"):
        return "rc_plan_rag"
    if not (state.get("repo_url") or "").strip():
        return "rc_plan_rag"
    return "rc_analyze"


def repo_context_analyze_node(state: InfraGraphState) -> dict[str, Any]:
    repo_url = (state.get("repo_url") or "").strip()
    branch = state.get("target_branch") or "main"
    head = state.get("repo_head_commit") or ""
    events: list[dict[str, Any]] = []
    tmp: Path | None = None
    try:
        if is_remote_git_url(repo_url):
            svc = GitService(
                repo_url=repo_url,
                default_branch=branch,
                remote_name=state.get("git_remote_name"),
            )
            tmp = shallow_clone_branch(repo_url, branch, svc)
            root = tmp / "repo"
            source = "fresh_clone"
        else:
            root = resolve_local_repo_root(repo_url)
            source = "fresh_local"

        paths, summary = build_tree_and_summary(root)
        modules = build_reusable_modules(root, paths)

        excerpts: dict[str, str] = {}
        tf_yaml = [
            p
            for p in paths
            if p.endswith((".tf", ".tfvars", ".yaml", ".yml"))
            and "/.git/" not in p.replace("\\", "/")
        ][:70]
        for rel in tf_yaml:
            excerpts[rel] = _read_snippet(root, rel)

        # repo_folder awareness
        repo_folder = (state.get("repo_folder") or "").strip()
        readme_contents = _collect_readmes(root, repo_folder)
        folder_exists, folder_files = _folder_info(root, repo_folder)
        logger.info(
            "repo_folder='%s' exists=%s files=%d READMEs=%d",
            repo_folder or "<none>", folder_exists, len(folder_files), len(readme_contents),
        )

        snap = {
            "repo_head_commit": head,
            "repo_tree_paths": paths,
            "repo_summary": summary,
            "reusable_modules": modules,
            "file_excerpts": excerpts,
            # folder-aware fields
            "readme_contents": readme_contents,
            "repo_folder_exists": folder_exists,
            "repo_folder_files": folder_files,
        }
        return {
            "repo_context": snap,
            "repo_context_source": source,
            "repo_context_events": events
            + [{
                "node": "repo_context_analyze",
                "paths": len(paths),
                "source": source,
                "repo_folder": repo_folder or "<none>",
                "folder_exists": folder_exists,
                "folder_files": len(folder_files),
                "readmes": list(readme_contents.keys()),
            }],
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("repo_context_analyze failed")
        return {
            "repo_context_status": "error",
            "repo_context_events": events + [{"node": "repo_context_analyze", "error": str(e)}],
        }
    finally:
        if tmp and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def repo_context_persist_node(state: InfraGraphState) -> dict[str, Any]:
    repo_url = (state.get("repo_url") or "").strip()
    branch = state.get("target_branch") or "main"
    head = state.get("repo_head_commit") or ""
    rc = state.get("repo_context") or {}
    if not repo_url or not head or not rc.get("repo_tree_paths"):
        return {"repo_context_events": [{"node": "repo_context_persist", "skipped": True}]}

    if state.get("repo_context_source") in ("state_reuse", "cache_sqlite"):
        return {"repo_context_events": [{"node": "repo_context_persist", "skipped": True}]}

    key = normalize_repo_key(repo_url)
    save_snapshot(key, branch, head, rc)

    chunks: list[dict[str, Any]] = []
    chunks.extend(summary_chunks(rc.get("repo_summary") or {}, rc.get("reusable_modules") or {}))
    for path, ex in (rc.get("file_excerpts") or {}).items():
        chunks.extend(chunk_text(path, ex))
    delete_repo_context_for_branch(key, branch)
    n = upsert_repo_context_chunks(key, branch, head, chunks)

    return {
        "repo_context_events": [
            {"node": "repo_context_persist", "sqlite": True, "milvus_chunks": n}
        ],
    }


def repo_context_plan_rag_node(state: InfraGraphState) -> dict[str, Any]:
    item = state.get("current_config_item") or {}
    fields = state.get("config_fields_output") or {}
    artifact = item.get("type") or "k8s_deployment"
    repo_url = (state.get("repo_url") or "").strip()
    branch = state.get("target_branch") or "main"
    head = state.get("repo_head_commit") or ""
    rc = state.get("repo_context") or {}

    paths = rc.get("repo_tree_paths") or []
    summary = rc.get("repo_summary") or {}
    modules = rc.get("reusable_modules") or {}
    excerpts = rc.get("file_excerpts") or {}

    plan = build_codegen_plan(artifact, fields, paths, summary, modules)
    key = normalize_repo_key(repo_url) if repo_url else ""
    ranked = rank_paths(artifact, fields, paths, plan, top_k=30)
    if key and head:
        ranked = merge_milvus_ranks(key, branch, head, artifact, fields, ranked, top_k=18)

    repo_folder = (state.get("repo_folder") or "").strip()
    folder_exists: bool = rc.get("repo_folder_exists", False)
    folder_files: list[str] = rc.get("repo_folder_files") or []
    readme_contents: dict[str, str] = rc.get("readme_contents") or {}

    # --- README instructions section ---
    readme_parts: list[str] = []
    for readme_path, readme_text in readme_contents.items():
        readme_parts.append(f"#### {readme_path}\n{readme_text.strip()}")
    readme_section = "\n\n".join(readme_parts)

    # --- repo_folder context section ---
    if repo_folder:
        if folder_exists:
            folder_ctx_lines = [
                f"The target folder '{repo_folder}' EXISTS in the repository.",
                "Treat the files listed below as the current baseline — update or extend them as needed.",
                "Generate the FULL path for every file (starting with the folder path).",
                f"Current files inside '{repo_folder}':",
            ] + ([f"  - {f}" for f in folder_files] or ["  (no files yet)"])
        else:
            # Identify sibling folders to infer structural conventions
            parent_path = str(Path(repo_folder).parent).replace("\\", "/")
            sibling_dirs = sorted({
                str(Path(p).parent).replace("\\", "/")
                for p in paths
                if str(Path(p).parent).replace("\\", "/").startswith(parent_path)
                   and str(Path(p).parent).replace("\\", "/") != parent_path
                   and str(Path(p).parent).replace("\\", "/") != repo_folder
            })[:10]
            folder_ctx_lines = [
                f"The target folder '{repo_folder}' does NOT exist in the repository — create it.",
                "Model the new folder's structure on the sibling/peer folders shown below.",
                "Generate the FULL path for every file (starting with the folder path).",
                f"Sibling folders under '{parent_path}' (for structural reference):",
            ] + ([f"  - {d}" for d in sibling_dirs] or ["  (no peer folders found)"])
        folder_context_section = "\n".join(folder_ctx_lines)
    else:
        folder_context_section = (
            "No specific repo_folder was provided. "
            "Choose an appropriate path that fits the existing repository layout."
        )

    rag_sections: list[tuple[str, str]] = [
        ("codegen_plan", json.dumps(plan, indent=2, default=str)[:8000]),
        ("repo_summary", json.dumps(summary, indent=2, default=str)[:6000]),
        ("reusable_modules", json.dumps(modules, indent=2, default=str)[:6000]),
        (
            "relevant_files_ranked",
            json.dumps(ranked[:25], indent=2, default=str)[:6000],
        ),
    ]
    if readme_section:
        rag_sections.append(("readme_instructions", readme_section[:8000]))
    rag_sections.append(("repo_folder_context", folder_context_section[:3000]))

    snippet_parts: list[str] = []
    for row in ranked[:12]:
        p = row.get("path")
        if not p:
            continue
        body = excerpts.get(p) or ""
        if not body:
            continue
        snippet_parts.append(f"### {p} (score={row.get('score')})\n```\n{body}\n```")
    rag_sections.append(("file_snippets", "\n\n".join(snippet_parts)[:20000]))

    text, meta = build_bounded_repo_rag_text(rag_sections, max_tokens=2800)

    events = [
        {
            "node": "repo_context_plan_rag",
            "ranked_top": [r.get("path") for r in ranked[:8]],
            "rag_meta": meta,
        }
    ]
    return {
        "codegen_plan": plan,
        "relevant_repo_files": ranked,
        "repo_rag_context_text": text,
        "repo_context_status": state.get("repo_context_status") or "ok",
        "repo_context_events": events,
    }
