"""Repo inspector, module analyzer, change planner, relevance ranker, context budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from infra_ai.nodes.context_budget import approx_tokens, build_bounded_repo_rag_text
from infra_ai.services.change_planner import build_codegen_plan
from infra_ai.services.module_analyzer import build_reusable_modules
from infra_ai.services.repo_inspector import build_tree_and_summary, classify_path
from infra_ai.services.relevance_ranker import rank_paths


def test_classify_path() -> None:
    assert classify_path("m/main.tf") == "terraform"
    assert classify_path("k/svc.yaml") == "kubernetes_yaml"


def test_build_tree_and_summary(tmp_path: Path) -> None:
    (tmp_path / "terraform").mkdir()
    (tmp_path / "terraform" / "main.tf").write_text('module "x" {\nsource = "./modules/vpc"\n}\n', encoding="utf-8")
    (tmp_path / "modules" / "vpc").mkdir(parents=True)
    (tmp_path / "modules" / "vpc" / "main.tf").write_text("# vpc\n", encoding="utf-8")
    (tmp_path / "k8s").mkdir()
    (tmp_path / "k8s" / "deployment.yaml").write_text("apiVersion: apps/v1\nkind: Deployment\n", encoding="utf-8")

    paths, summary = build_tree_and_summary(tmp_path)
    assert "terraform/main.tf" in paths
    assert "k8s/deployment.yaml" in paths
    assert summary.get("file_types", {}).get("terraform", 0) >= 1


def test_module_analyzer_finds_tf_module_and_k8s(tmp_path: Path) -> None:
    (tmp_path / "root.tf").write_text(
        'module "net" {\n  source = "./modules/network"\n}\n',
        encoding="utf-8",
    )
    (tmp_path / "modules" / "network").mkdir(parents=True)
    (tmp_path / "modules" / "network" / "main.tf").write_text("resource \"null_resource\" \"x\" {}\n", encoding="utf-8")
    (tmp_path / "charts" / "app").mkdir(parents=True)
    (tmp_path / "charts" / "app" / "Chart.yaml").write_text("apiVersion: v2\nname: app\n", encoding="utf-8")
    (tmp_path / "overlay").mkdir(parents=True)
    (tmp_path / "overlay" / "kustomization.yaml").write_text("resources:\n- ../base\n", encoding="utf-8")

    paths = [str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.rglob("*") if p.is_file()]
    mods = build_reusable_modules(tmp_path, paths)
    kinds = {m.get("kind") for m in mods["terraform"]}
    assert "terraform_module_call" in kinds
    assert "terraform_module_definition" in kinds
    assert any(m.get("kind") == "helm_chart" for m in mods["kubernetes"])
    assert any(m.get("kind") == "kustomize" for m in mods["kubernetes"])


def test_change_planner_terraform_paths(tmp_path: Path) -> None:
    paths = ["terraform/main.tf", "README.md"]
    summary = {"top_level_folders": ["terraform"], "file_types": {"terraform": 1}}
    mods = {"terraform": [], "kubernetes": []}
    plan = build_codegen_plan(
        "terraform_eks_cluster",
        {"cluster_name": {"value": "c1"}},
        paths,
        summary,
        mods,
    )
    assert plan.get("family") == "terraform"
    assert plan.get("update_paths") or plan.get("create_paths")


def test_rank_paths_prefers_plan_targets() -> None:
    plan = {"create_paths": ["k8s/new.yaml"], "update_paths": []}
    paths = ["other.txt", "k8s/new.yaml", "k8s/old.yaml"]
    ranked = rank_paths("k8s_deployment", {"app": {"value": "x"}}, paths, plan, top_k=10)
    assert ranked[0]["path"] == "k8s/new.yaml"
    assert ranked[0]["score"] >= ranked[-1]["score"]


def test_context_budget_truncates_large_section() -> None:
    huge = "word " * 50000
    text, meta = build_bounded_repo_rag_text(
        [("small", "hi"), ("big", huge)],
        max_tokens=500,
    )
    assert approx_tokens(text) <= 550
    assert meta.get("truncated") or meta.get("dropped_sections")
