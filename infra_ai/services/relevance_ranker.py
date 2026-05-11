"""Lexical + optional Milvus scores for repo file relevance."""

from __future__ import annotations

import json
import re
from typing import Any

from infra_ai.milvus_store import query_repo_context_chunks


def _tokens(s: str) -> set[str]:
    s = re.sub(r"[^\w\./-]+", " ", (s or "").lower())
    return {t for t in s.split() if len(t) > 2}


def lexical_score(artifact: str, fields: dict[str, Any], rel_path: str) -> tuple[float, str]:
    blob = f"{artifact} {json.dumps(fields, default=str)}"
    keys = _tokens(blob)
    path_lower = rel_path.lower()
    score = 0.0
    reasons: list[str] = []

    if artifact.startswith("terraform_"):
        if path_lower.endswith(".tf") or path_lower.endswith(".tfvars"):
            score += 3.0
            reasons.append("terraform_file")
        if "module" in path_lower:
            score += 2.0
            reasons.append("path_module")
        if "eks" in path_lower and "eks" in keys:
            score += 2.0
            reasons.append("path_eks")
        if "vpc" in path_lower:
            score += 1.0
    else:
        if path_lower.endswith((".yaml", ".yml")):
            score += 3.0
            reasons.append("yaml")
        if any(x in path_lower for x in ("k8s", "kubernetes", "manifest", "deploy")):
            score += 2.0
            reasons.append("k8s_path")

    for k in keys:
        if k in path_lower:
            score += 0.5
            reasons.append(f"kw:{k}")

    return score, ",".join(reasons[:6]) or "baseline"


def rank_paths(
    artifact: str,
    fields: dict[str, Any],
    repo_tree_paths: list[str],
    codegen_plan: dict[str, Any],
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """Return sorted list of {path, score, reason}."""
    priority: set[str] = set()
    for key in ("create_paths", "update_paths"):
        for p in codegen_plan.get(key) or []:
            priority.add(p.replace("\\", "/"))

    scored: list[dict[str, Any]] = []
    for rel in repo_tree_paths:
        sc, reason = lexical_score(artifact, fields, rel)
        if rel in priority:
            sc += 5.0
            reason = "plan_target," + reason
        scored.append({"path": rel.replace("\\", "/"), "score": sc, "reason": reason})

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def merge_milvus_ranks(
    repo_key: str,
    base_branch: str,
    head_commit: str,
    artifact: str,
    fields: dict[str, Any],
    lexical_ranked: list[dict[str, Any]],
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Re-rank using Milvus chunk hits when enabled; else return lexical_ranked."""
    q = f"{artifact} {json.dumps(fields, default=str)}"[:4000]
    hits = query_repo_context_chunks(q, repo_key, base_branch, head_commit, top_k=top_k)
    if not hits:
        return lexical_ranked

    boost: dict[str, float] = {}
    for h in hits:
        p = (h.get("path") or "").replace("\\", "/")
        if not p:
            continue
        boost[p] = boost.get(p, 0.0) + float(h.get("score", 0.0))

    merged: list[dict[str, Any]] = []
    for row in lexical_ranked:
        p = row["path"]
        sc = row["score"] + boost.get(p, 0.0)
        reason = row["reason"]
        if p in boost:
            reason = reason + f",milvus+{boost[p]:.2f}"
        merged.append({"path": p, "score": sc, "reason": reason})
    merged.sort(key=lambda x: -x["score"])
    return merged[:top_k]
