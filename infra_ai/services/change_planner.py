"""Deterministic change plan from artifact, fields, and repo snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _artifact_family(artifact: str) -> str:
    if artifact.startswith("terraform_"):
        return "terraform"
    return "kubernetes"


def build_codegen_plan(
    artifact: str,
    fields: dict[str, Any],
    repo_tree_paths: list[str],
    repo_summary: dict[str, Any],
    reusable_modules: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce codegen_plan: folders_to_touch, create_paths, update_paths,
    reuse_modules, notes.
    """
    family = _artifact_family(artifact)
    paths_set = set(repo_tree_paths)
    create_paths: list[str] = []
    update_paths: list[str] = []
    reuse: list[dict[str, Any]] = []
    notes: list[str] = []

    if family == "terraform":
        tf_roots = sorted({str(Path(p).parent).replace("\\", "/") for p in repo_tree_paths if p.endswith(".tf")})
        if not tf_roots:
            create_paths.append("terraform/main.tf")
            notes.append("No existing .tf files; propose terraform/ layout.")
        else:
            # Prefer environments/ or terraform/ roots
            preferred = [r for r in tf_roots if "env" in r or r in (".", "terraform", "infra")]
            target_root = preferred[0] if preferred else tf_roots[0]
            notes.append(f"Primary terraform folder candidate: {target_root}")
            main_tf = f"{target_root}/main.tf".replace("//", "/") if target_root != "." else "main.tf"
            if main_tf in paths_set or (target_root == "." and "main.tf" in paths_set):
                update_paths.append(main_tf if main_tf in paths_set else "main.tf")
            else:
                create_paths.append(main_tf if target_root != "." else "main.tf")

        for m in reusable_modules.get("terraform") or []:
            if m.get("kind") == "terraform_module_definition":
                reuse.append(
                    {
                        "type": "terraform_local_module",
                        "path": m.get("path"),
                        "how": f'module "x" {{ source = "./{m.get("path")}" }}',
                    }
                )
            if m.get("kind") == "terraform_module_call" and m.get("source", "").startswith("."):
                reuse.append(
                    {
                        "type": "terraform_existing_call_pattern",
                        "file": m.get("file"),
                        "module_name": m.get("module_name"),
                        "source": m.get("source"),
                    }
                )

    else:
        # Kubernetes
        yaml_paths = [p for p in repo_tree_paths if p.endswith((".yaml", ".yml"))]
        k8s_like = [p for p in yaml_paths if any(x in p.lower() for x in ("k8s", "manifest", "deploy", "helm", "charts"))]
        if k8s_like:
            target = str(Path(k8s_like[0]).parent).replace("\\", "/")
            notes.append(f"Kubernetes YAML cluster under: {target}")
            deploy_name = "deployment.yaml"
            cand = f"{target}/{deploy_name}" if target != "." else deploy_name
            if cand in paths_set:
                update_paths.append(cand)
            else:
                create_paths.append(cand if target == "." else f"{target}/{deploy_name}")
        else:
            create_paths.append("k8s/deployment.yaml")
            notes.append("No k8s-like paths; default k8s/deployment.yaml.")

        for m in reusable_modules.get("kubernetes") or []:
            if m.get("kind") == "helm_chart":
                reuse.append({"type": "helm_chart", "path": m.get("path"), "how": "extend Chart templates or add subchart"})
            if m.get("kind") == "kustomize":
                reuse.append(
                    {
                        "type": "kustomize",
                        "path": m.get("path"),
                        "how": "add resources/patches under this kustomization",
                    }
                )

    folders_to_touch = sorted(
        {str(Path(p).parent).replace("\\", "/") for p in create_paths + update_paths if "/" in p}
    )

    return {
        "artifact": artifact,
        "family": family,
        "folders_to_touch": folders_to_touch,
        "create_paths": create_paths,
        "update_paths": update_paths,
        "reuse_modules": reuse,
        "notes": notes,
        "fields_digest": json.dumps(fields, sort_keys=True, default=str)[:2000],
    }
