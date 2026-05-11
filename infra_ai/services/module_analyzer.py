"""Detect Terraform and Kubernetes reusable modules/patterns in a repo root."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_MODULE_BLOCK = re.compile(
    r"module\s+\"([^\"]+)\"\s*\{([^}]*)\}",
    re.MULTILINE | re.DOTALL,
)
_SOURCE = re.compile(r"source\s*=\s*\"([^\"]+)\"")


def _read_text_limited(path: Path, max_bytes: int = 256_000) -> str:
    try:
        raw = path.read_bytes()[:max_bytes]
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return ""


def analyze_terraform_modules(repo_root: Path, paths: list[str]) -> list[dict[str, Any]]:
    """Terraform root modules (module blocks) and local module folders."""
    roots: list[dict[str, Any]] = []

    for rel in paths:
        if not rel.endswith(".tf"):
            continue
        p = repo_root / rel
        if not p.is_file():
            continue
        text = _read_text_limited(p)
        for m in _MODULE_BLOCK.finditer(text):
            name = m.group(1)
            block = m.group(2)
            sm = _SOURCE.search(block)
            source = sm.group(1) if sm else ""
            entry = {
                "kind": "terraform_module_call",
                "file": rel.replace("\\", "/"),
                "module_name": name,
                "source": source,
            }
            if source.startswith("./") or source.startswith("../") or source.startswith("./../"):
                entry["local_source"] = source
            roots.append(entry)

    seen_def: set[str] = set()
    mod_dir = repo_root / "modules"
    if mod_dir.is_dir():
        for child in sorted(mod_dir.iterdir()):
            if not child.is_dir():
                continue
            mod_name = child.name
            key = f"modules/{mod_name}"
            if mod_name in seen_def:
                continue
            if (child / "main.tf").is_file():
                seen_def.add(mod_name)
                roots.append(
                    {
                        "kind": "terraform_module_definition",
                        "path": key,
                        "module_name": mod_name,
                        "entry": f"modules/{mod_name}/main.tf",
                    }
                )

    return roots


def analyze_kubernetes_reuse(repo_root: Path, paths: list[str]) -> list[dict[str, Any]]:
    """Helm charts, Kustomize bases/overlays, shared manifest dirs."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for rel in paths:
        r = rel.replace("\\", "/")
        base = Path(r).name.lower()
        if base == "chart.yaml":
            chart_dir = str(Path(r).parent).replace("\\", "/")
            if chart_dir not in seen:
                seen.add(chart_dir)
                out.append({"kind": "helm_chart", "path": chart_dir, "chart_yaml": r})
        if base in ("kustomization.yaml", "kustomization.yml"):
            kdir = str(Path(r).parent).replace("\\", "/")
            if kdir not in seen:
                seen.add(kdir)
                text = _read_text_limited(repo_root / rel)
                bases = re.findall(r"^\s*bases?:\s*$", text, re.MULTILINE)
                has_resources = "resources:" in text
                out.append(
                    {
                        "kind": "kustomize",
                        "path": kdir,
                        "kustomization_yaml": r,
                        "has_bases": bool(bases),
                        "has_resources": has_resources,
                    }
                )

    hint_paths = [rel for rel in paths if "base" in Path(rel).parts or "bases" in Path(rel).parts]
    for rel in hint_paths[:40]:
        r = rel.replace("\\", "/")
        out.append({"kind": "k8s_path_hint", "path": r, "hint": "possible_kustomize_or_shared_base"})
    return out


def build_reusable_modules(repo_root: Path, paths: list[str]) -> dict[str, Any]:
    tf = analyze_terraform_modules(repo_root, paths)
    k8s = analyze_kubernetes_reuse(repo_root, paths)
    return {
        "terraform": tf,
        "kubernetes": k8s,
    }
