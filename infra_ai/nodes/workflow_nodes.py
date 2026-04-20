from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from infra_ai.config import get_settings
from infra_ai.llm.factory import get_chat_model
from infra_ai.milvus_store import query_skill_chunks
from infra_ai.nodes.llm_utils import extract_json_object, invoke_structured, mock_llm_enabled
from infra_ai.schemas.config_plan import ConfigPlan, ConfigPlanItem
from infra_ai.schemas.requirements import RequirementAnalysis
from infra_ai.services.git_service import GitService
from infra_ai.skills.loader import load_skill_markdown
from infra_ai.state import InfraGraphState
from infra_ai.validation.deterministic import validate_config_fields
from infra_ai.validation.plugins import run_plugins

logger = logging.getLogger(__name__)

DEFAULT_ENVS = ["dev", "test", "prod"]


def _heuristic_requirement(text: str, configs: dict[str, Any]) -> RequirementAnalysis:
    t = text.lower()
    tech: list[str] = []
    if "react" in t:
        tech.append("react")
    app_type = "frontend" if "frontend" in t or "react" in t else None
    merged = RequirementAnalysis(
        application_type=app_type,
        application_tech=tech,
        extra_configs={k: v for k, v in configs.items()},
    )
    return merged


def requirement_analysis_node(state: InfraGraphState) -> dict[str, Any]:
    text = state.get("raw_user_text") or ""
    configs = state.get("raw_user_configs") or {}
    if mock_llm_enabled():
        analysis = _heuristic_requirement(text, configs)
    else:
        try:
            llm = get_chat_model("requirement")
            prompt = (
                "Extract infrastructure requirements as JSON matching the schema. "
                "Put unknown keys under extra_configs.\n\n"
                f"User text:\n{text}\n\nPartial configs JSON:\n{json.dumps(configs, indent=2)}"
            )
            analysis = invoke_structured(llm, prompt, RequirementAnalysis)
        except Exception:  # noqa: BLE001
            logger.exception("Requirement LLM failed; using heuristic fallback")
            analysis = _heuristic_requirement(text, configs)
    return {
        "requirement_analysis": analysis.to_state_dict(),
        "events": [{"node": "requirement_analysis", "ok": True}],
    }


def _expand_plan_items(items: list[ConfigPlanItem], req: dict[str, Any]) -> list[ConfigPlanItem]:
    envs = req.get("environments") or []
    if not envs:
        envs = DEFAULT_ENVS
    out: list[ConfigPlanItem] = []
    for item in items:
        if item.environment:
            out.append(item)
            continue
        for e in envs:
            clone = item.model_copy(update={"environment": e, "id": f"{item.id}-{e}"})
            out.append(clone)
    return out


def _heuristic_plan(req: dict[str, Any]) -> ConfigPlan:
    app = req.get("application_type") or "app"
    items = [
        ConfigPlanItem(
            id=f"tf-eks-{app}",
            description=f"Terraform EKS cluster for {app}",
            type="terraform_eks_cluster",
        ),
        ConfigPlanItem(
            id=f"k8s-deploy-{app}",
            description=f"Kubernetes Deployment for {app}",
            type="k8s_deployment",
        ),
    ]
    return ConfigPlan(items=_expand_plan_items(items, req))


def config_analysis_node(state: InfraGraphState) -> dict[str, Any]:
    req = state.get("requirement_analysis") or {}
    if mock_llm_enabled():
        plan = _heuristic_plan(req)
    else:
        try:
            llm = get_chat_model("config_plan")
            prompt = (
                "From the requirement JSON, list concrete infra config artifacts. "
                "Use types: terraform_eks_cluster, k8s_deployment, terraform_storage. "
                "If environment is missing for an item, leave environment null for expansion.\n\n"
                f"Requirements:\n{json.dumps(req, indent=2)}"
            )
            plan = invoke_structured(llm, prompt, ConfigPlan)
        except Exception:  # noqa: BLE001
            logger.exception("Config plan LLM failed; using heuristic fallback")
            plan = _heuristic_plan(req)
    items = _expand_plan_items(plan.items, req)
    s = get_settings()
    return {
        "config_plan": [i.model_dump() for i in items],
        "current_config_index": 0,
        "repo_url": s.git_repo_url,
        "target_branch": s.git_default_branch,
        "git_remote_name": s.git_remote_name,
        "events": [{"node": "config_analysis", "count": len(items)}],
    }


def loop_entry_node(state: InfraGraphState) -> dict[str, Any]:
    plan = state.get("config_plan") or []
    idx = int(state.get("current_config_index") or 0)
    if idx >= len(plan):
        return {
            "workflow_status": "completed",
            "events": [{"node": "loop_entry", "status": "no_more_items"}],
        }
    item = plan[idx]
    return {
        "current_config_item": item,
        "workflow_status": "running",
        "events": [{"node": "loop_entry", "index": idx, "id": item.get("id")}],
    }


def route_after_loop(state: InfraGraphState) -> Literal["infra", "finalize"]:
    plan = state.get("config_plan") or []
    idx = int(state.get("current_config_index") or 0)
    if state.get("workflow_status") == "completed" or idx >= len(plan):
        return "finalize"
    return "infra"


def _envelope(value: Any, agent_generated: bool, confidence: float) -> dict[str, Any]:
    return {
        "value": value,
        "agent_generated": agent_generated,
        "confidence_score": min(9.9, max(0.0, confidence)),
    }


def _mock_fields(artifact_type: str, env: str, region: str, app: str) -> dict[str, Any]:
    if artifact_type == "terraform_eks_cluster":
        return {
            "environment": _envelope(env, False, 9.9),
            "region": _envelope(region, False, 9.9),
            "cluster_name": _envelope(f"{app}-{env}-eks", True, 7.5),
            "kubernetes_version": _envelope("1.29", True, 6.0),
            "node_desired_size": _envelope(2, True, 6.0),
            "node_min_size": _envelope(1, True, 6.0),
            "node_max_size": _envelope(4, True, 6.0),
            "node_instance_types": _envelope(["m5.large"], True, 6.0),
        }
    if artifact_type == "terraform_storage":
        return {
            "environment": _envelope(env, False, 9.9),
            "region": _envelope(region, False, 9.9),
            "bucket_name": _envelope(f"{app}-{env}-data", True, 7.0),
            "enable_versioning": _envelope(True, True, 6.0),
            "enable_kms": _envelope(True, True, 6.0),
        }
    image = "nginx:1.25.3"
    return {
        "environment": _envelope(env, False, 9.9),
        "namespace": _envelope(f"{app}-{env}", True, 7.0),
        "app_name": _envelope(app, False, 9.9),
        "image": _envelope(image, True, 7.0),
        "replicas": _envelope(2, True, 7.0),
        "container_port": _envelope(80, True, 7.0),
        "cpu_request": _envelope("250m", True, 6.0),
        "cpu_limit": _envelope("500m", True, 6.0),
        "memory_request": _envelope("256Mi", True, 6.0),
        "memory_limit": _envelope("512Mi", True, 6.0),
    }


def infra_builder_node(state: InfraGraphState) -> dict[str, Any]:
    req = state.get("requirement_analysis") or {}
    item = state.get("current_config_item") or {}
    artifact_type = item.get("type") or "k8s_deployment"
    skill = load_skill_markdown(artifact_type)
    extra_chunks = query_skill_chunks(json.dumps(req), artifact_type)
    if extra_chunks:
        skill = skill + "\n\n## Retrieved chunks\n" + "\n".join(extra_chunks)

    env = item.get("environment") or "dev"
    region = req.get("region") or "us-east-1"
    app = req.get("application_type") or "app"

    if mock_llm_enabled():
        fields = _mock_fields(artifact_type, env, region, app)
    else:
        try:
            llm = get_chat_model("builder")
            prompt = (
                "You populate infrastructure config fields as JSON. "
                "Each key maps to an object with keys: value, agent_generated (bool), "
                "confidence_score (0-9.9; use 9.9 when taken from user input).\n\n"
                f"Requirement JSON:\n{json.dumps(req, indent=2)}\n\n"
                f"Current artifact plan item:\n{json.dumps(item, indent=2)}\n\n"
                f"Skill:\n{skill}\n\n"
                "Return ONLY a JSON object of fields."
            )
            msg = llm.invoke([HumanMessage(content=prompt)])
            content = msg.content if hasattr(msg, "content") else str(msg)
            parsed = extract_json_object(str(content))
            if not parsed:
                raise ValueError("No JSON in builder response")
            fields = parsed
        except Exception:  # noqa: BLE001
            logger.exception("Builder LLM failed; using mock fields")
            fields = _mock_fields(artifact_type, env, region, app)
    return {
        "config_fields_output": fields,
        "events": [{"node": "infra_builder", "artifact": artifact_type}],
    }


def infra_validator_node(state: InfraGraphState) -> dict[str, Any]:
    fields = state.get("config_fields_output") or {}
    item = state.get("current_config_item") or {}
    artifact_type = item.get("type") or "k8s_deployment"
    ok, errs = validate_config_fields(fields, artifact_type)
    ok2, errs2 = run_plugins(fields, artifact_type)
    ok = ok and ok2
    errs = errs + errs2
    events = [{"node": "infra_validator", "ok": ok, "errors": errs}]
    if not ok:
        logger.warning("Validation issues: %s", errs)
    return {"events": events}


def human_review_node(state: InfraGraphState) -> dict[str, Any]:
    payload = {
        "kind": "review_fields",
        "config_fields_output": state.get("config_fields_output"),
        "current_config_item": state.get("current_config_item"),
    }
    edited = interrupt(payload)
    if isinstance(edited, dict) and "config_fields" in edited:
        fields = edited["config_fields"]
    elif isinstance(edited, dict):
        fields = edited
    else:
        fields = state.get("config_fields_output") or {}
    return {
        "config_fields_output": fields,
        "human_review_status": "reviewed",
        "last_interrupt_kind": "review_fields",
        "events": [{"node": "human_review", "received": True}],
    }


def human_repo_node(state: InfraGraphState) -> dict[str, Any]:
    s = get_settings()
    repo = state.get("repo_url") or s.git_repo_url
    payload = {
        "kind": "confirm_repo",
        "repo_url": repo,
        "target_branch": state.get("target_branch") or s.git_default_branch,
        "message": (
            "Confirm code generation. Override repo_url: use an https/git URL to clone and push a new branch, "
            "or a filesystem path (e.g. ./my-infra-out or C:/infra-out) to write files locally only (no GitHub push)."
        ),
    }
    conf = interrupt(payload)
    if not isinstance(conf, dict):
        conf = {}
    if conf.get("repo_url"):
        repo = conf["repo_url"]
    branch = conf.get("target_branch") or state.get("target_branch") or s.git_default_branch
    confirm = conf.get("confirm", True)
    return {
        "repo_url": repo,
        "target_branch": branch,
        "last_interrupt_kind": "confirm_repo",
        "events": [{"node": "human_repo", "confirm": confirm}],
    }


CODE_BLOCK = re.compile(r"```(?:\w+)?\s*\n([\s\S]*?)```", re.MULTILINE)


def _parse_generated_files(text: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    parts = re.split(r"^###\s+", text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        path_line = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]) if len(lines) > 1 else part
        m = CODE_BLOCK.search(body)
        content = m.group(1).strip() if m else body.strip()
        if path_line and ("/" in path_line or path_line.endswith((".tf", ".yaml", ".yml"))):
            files.append((path_line.split()[0], content))
    if not files:
        m = CODE_BLOCK.search(text)
        if m:
            files.append(("generated/main.tf", m.group(1).strip()))
    return files


def _codegen_system_messages(artifact: str) -> list[Any]:
    if artifact.startswith("terraform_"):
        return [
            SystemMessage(
                content=(
                    "You are a Terraform expert.\n\n"
                    "Use terraform-aws-modules/eks/aws best practices.\n"
                    "Follow this pattern:\n"
                    "- VPC module\n"
                    "- EKS module (no custom IAM)\n"
                    "- Outputs only from module\n\n"
                    "Follow these strict rules:\n"
                    "1. If using terraform-aws-modules/eks/aws:\n"
                    "   - DO NOT create IAM roles manually unless explicitly required\n"
                    "   - DO NOT create security groups unless needed\n"
                    "2. Separate infrastructure and Kubernetes resources:\n"
                    "   - No kubernetes_* resources in same apply as EKS creation\n"
                    "3. Ensure dependency correctness:\n"
                    "   - Providers must not depend on resources created in same plan\n"
                    "4. Prefer module defaults unless customization is required\n"
                    "5. Output only valid, deployable Terraform\n\n"
                    "After generating:\n"
                    "- Simulate terraform plan mentally\n"
                    "- Identify dependency or lifecycle issues\n"
                    "- Fix them before output\n"
                )
            )
        ]
    return [
        SystemMessage(
            content=(
                "You are a Kubernetes manifest expert.\n\n"
                "Generate valid Kubernetes YAML only.\n"
                "Do not include Terraform or EKS cluster creation resources.\n"
                "Use apiVersion/apps/v1 for Deployment and apiVersion/v1 for Service.\n"
                "Do not use :latest image tags.\n"
                "Keep resources focused on the application workload and common defaults.\n"
            )
        )
    ]


def _terraform_fmt(paths: list[Path]) -> None:
    tf_files = [p for p in paths if p.suffix == ".tf"]
    if not tf_files:
        return
    try:
        subprocess.run(
            ["terraform", "fmt", "-no-color", *[str(p) for p in tf_files]],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        logger.info("terraform binary not found; skipping fmt")


def codegen_node(state: InfraGraphState) -> dict[str, Any]:
    item = state.get("current_config_item") or {}
    fields = state.get("config_fields_output") or {}
    artifact = item.get("type") or "k8s_deployment"
    def _mock_codegen_text() -> str:
        if artifact == "k8s_deployment":
            return (
                "### k8s/deployment.yaml\n```yaml\n"
                "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
                "  name: app\nspec:\n  replicas: 1\n  selector: {}\n  template: {}\n```\n"
            )
        return (
            "### terraform/main.tf\n```hcl\n"
            'terraform {\n  required_version = ">= 1.5.0"\n}\n```\n'
        )

    if mock_llm_enabled():
        text = _mock_codegen_text()
    else:
        try:
            llm = get_chat_model("codegen")
            prompt = (
                "Generate infrastructure files. Use markdown sections starting with "
                "'### <relative/path>' followed by a fenced code block.\n\n"
                f"Artifact type: {artifact}\n"
                f"Fields JSON:\n{json.dumps(fields, indent=2)}\n"
            )
            messages = _codegen_system_messages(artifact) + [HumanMessage(content=prompt)]
            msg = llm.invoke(messages)
            text = str(msg.content)
        except Exception:  # noqa: BLE001
            logger.exception("Codegen LLM failed; using stub output")
            text = _mock_codegen_text()

    files = _parse_generated_files(text)
    tmp = Path(tempfile.mkdtemp(prefix="infra-ai-codegen-"))
    written_fmt: list[tuple[str, str]] = []
    try:
        for rel, content in files:
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        tf_paths = [p for p in tmp.rglob("*.tf") if p.is_file()]
        _terraform_fmt(tf_paths)
        for rel, _ in files:
            p = tmp / rel
            if p.is_file():
                written_fmt.append((rel, p.read_text(encoding="utf-8")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "generated_files": [{"path": r, "content": c} for r, c in written_fmt],
        "events": [{"node": "codegen", "files": len(written_fmt)}],
    }


def git_push_node(state: InfraGraphState) -> dict[str, Any]:
    item = state.get("current_config_item") or {}
    gfs = state.get("generated_files") or []
    files = [(g["path"], g["content"]) for g in gfs]
    prefix = str(item.get("id") or "config").replace("/", "-")[:40]
    svc = GitService(
        repo_url=state.get("repo_url"),
        default_branch=state.get("target_branch"),
        remote_name=state.get("git_remote_name"),
    )
    branch, messages = svc.push_files(files, branch_prefix=prefix)
    return {
        "last_git_branch": branch,
        "events": [{"node": "git_push", "messages": messages, "branch": branch}],
    }


def human_continue_node(state: InfraGraphState) -> dict[str, Any]:
    item = state.get("current_config_item") or {}
    desc = item.get("description") or item.get("id")
    payload = {
        "kind": "continue_next",
        "message": f'Code deployed for "{desc}". Continue with next config file?',
        "current_config_item": item,
    }
    ans = interrupt(payload)
    cont = True
    if isinstance(ans, dict):
        cont = bool(ans.get("continue_next", True))
    elif isinstance(ans, bool):
        cont = ans
    updates: dict[str, Any] = {
        "last_interrupt_kind": "continue_next",
        "events": [{"node": "human_continue", "continue_next": cont}],
    }
    if cont:
        idx = int(state.get("current_config_index") or 0)
        updates["current_config_index"] = idx + 1
    else:
        updates["workflow_status"] = "completed"
    return updates


def route_after_continue(state: InfraGraphState) -> Literal["loop", "finalize"]:
    if state.get("workflow_status") == "completed":
        return "finalize"
    return "loop"


def finalize_node(state: InfraGraphState) -> dict[str, Any]:
    return {"events": [{"node": "finalize", "status": state.get("workflow_status", "completed")}]}
