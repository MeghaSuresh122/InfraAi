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
    logger.info("=== REQUIREMENT ANALYSIS STAGE ===")
    logger.info("User text length: %d chars", len(text))
    logger.info("Config keys: %s", list(configs.keys()))
    
    if mock_llm_enabled():
        logger.debug("Using mock LLM for requirement analysis")
        analysis = _heuristic_requirement(text, configs)
    else:
        try:
            logger.debug("Invoking LLM for requirement analysis")
            llm = get_chat_model("requirement")
            prompt = (
                "Extract infrastructure requirements as JSON matching the schema. "
                "Put unknown keys under extra_configs.\n\n"
                f"User text:\n{text}\n\nPartial configs JSON:\n{json.dumps(configs, indent=2)}"
            )
            analysis = invoke_structured(llm, prompt, RequirementAnalysis)
            logger.info("Requirement analysis completed. App type: %s", analysis.application_type)
        except Exception as e:  # noqa: BLE001
            logger.exception("Requirement LLM failed")
            error_payload = {
                "kind": "requirement_error",
                "message": f"Requirement analysis failed: {e}",
                "error": str(e),
                "text": text,
                "configs": configs
            }
            edited = interrupt(error_payload)
            if edited and edited.get("retry"):
                logger.info("Retrying requirement analysis after user review")
                llm = get_chat_model("requirement")
                prompt = (
                    "Extract infrastructure requirements as JSON matching the schema. "
                    "Put unknown keys under extra_configs.\n\n"
                    f"User text:\n{text}\n\nPartial configs JSON:\n{json.dumps(configs, indent=2)}"
                )
                analysis = invoke_structured(llm, prompt, RequirementAnalysis)
                logger.info("Requirement analysis retry completed")
            else:
                raise RuntimeError(f"Requirement analysis aborted: {e}")
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
    logger.info("=== CONFIG ANALYSIS STAGE ===")
    req = state.get("requirement_analysis") or {}
    if mock_llm_enabled():
        logger.debug("Using mock LLM for config analysis")
        plan = _heuristic_plan(req)
    else:
        try:
            logger.debug("Invoking LLM for config plan generation")
            llm = get_chat_model("config_plan")
            prompt = (
                "From the requirement JSON, list concrete infra config artifacts. "
                "Use types: terraform_eks_cluster, k8s_deployment, terraform_storage. "
                "If environment is missing for an item, leave environment null for expansion.\n\n"
                f"Requirements:\n{json.dumps(req, indent=2)}"
            )
            plan = invoke_structured(llm, prompt, ConfigPlan)
            logger.info("Config plan generated with %d items", len(plan.items))
        except Exception as e:  # noqa: BLE001
            logger.exception("Config plan LLM failed")
            error_payload = {
                "kind": "config_plan_error",
                "message": f"Config plan analysis failed: {e}",
                "error": str(e),
                "requirements": req
            }
            edited = interrupt(error_payload)
            if edited and edited.get("retry"):
                logger.info("Retrying config plan after user review")
                llm = get_chat_model("config_plan")
                prompt = (
                    "From the requirement JSON, list concrete infra config artifacts. "
                    "Use types: terraform_eks_cluster, k8s_deployment, terraform_storage. "
                    "If environment is missing for an item, leave environment null for expansion.\n\n"
                    f"Requirements:\n{json.dumps(req, indent=2)}"
                )
                plan = invoke_structured(llm, prompt, ConfigPlan)
                logger.info("Config plan retry completed")
            else:
                raise RuntimeError(f"Config plan analysis aborted: {e}")
    items = _expand_plan_items(plan.items, req)
    s = get_settings()
    logger.info("Final config plan has %d items after expansion", len(items))
    logger.info("Target branch: %s, Repo: %s", s.git_default_branch, s.git_repo_url)
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
    logger.info("=== LOOP ENTRY ===")
    logger.info("Processing item %d of %d", idx + 1, len(plan))
    
    if idx >= len(plan):
        logger.info("No more items to process. Finalizing workflow.")
        return {
            "workflow_status": "completed",
            "events": [{"node": "loop_entry", "status": "no_more_items"}],
        }
    item = plan[idx]
    item_id = item.get("id")
    item_desc = item.get("description", item_id)
    logger.info("Processing config item: %s - %s", item_id, item_desc)
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
    logger.info("=== INFRA BUILDER STAGE ===")
    req = state.get("requirement_analysis") or {}
    item = state.get("current_config_item") or {}
    artifact_type = item.get("type") or "k8s_deployment"
    logger.info("Building artifact: %s", artifact_type)
    
    skill = load_skill_markdown(artifact_type)
    extra_chunks = query_skill_chunks(json.dumps(req), artifact_type)
    if extra_chunks:
        logger.info("Retrieved %d skill chunks for artifact", len(extra_chunks))
        skill = skill + "\n\n## Retrieved chunks\n" + "\n".join(extra_chunks)

    env = item.get("environment") or "dev"
    region = req.get("region") or "us-east-1"
    app = req.get("application_type") or "app"
    logger.debug("Artifact config - Env: %s, Region: %s, App: %s", env, region, app)

    if mock_llm_enabled():
        logger.debug("Using mock fields for builder")
        fields = _mock_fields(artifact_type, env, region, app)
    else:
        try:
            logger.debug("Invoking LLM for infra builder")
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
            logger.info("Builder LLM returned %d fields", len(fields))
        except Exception as e:  # noqa: BLE001
            logger.exception("Builder LLM failed for artifact: %s", artifact_type)
            error_payload = {
                "kind": "builder_error",
                "message": f"Infra builder failed for {artifact_type}: {e}",
                "error": str(e),
                "artifact_type": artifact_type,
                "req": req,
                "item": item
            }
            edited = interrupt(error_payload)
            if edited and edited.get("retry"):
                logger.info("Retrying builder after user review")
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
                logger.info("Builder retry completed")
            else:
                raise RuntimeError(f"Infra builder aborted: {e}")
    return {
        "config_fields_output": fields,
        "events": [{"node": "infra_builder", "artifact": artifact_type}],
    }


def infra_validator_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== INFRA VALIDATOR STAGE ===")
    fields = state.get("config_fields_output") or {}
    item = state.get("current_config_item") or {}
    artifact_type = item.get("type") or "k8s_deployment"
    logger.info("Validating artifact: %s", artifact_type)
    
    ok, errs = validate_config_fields(fields, artifact_type)
    ok2, errs2 = run_plugins(fields, artifact_type)
    ok = ok and ok2
    errs = errs + errs2
    
    if ok:
        logger.info("Validation passed for artifact: %s", artifact_type)
    else:
        logger.warning("Validation failed for artifact: %s. Errors: %s", artifact_type, errs)
    
    events = [{"node": "infra_validator", "ok": ok, "errors": errs}]
    return {"events": events}


def human_review_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== HUMAN REVIEW STAGE ===")
    logger.info("Pausing for human review of configuration fields")
    payload = {
        "kind": "review_fields",
        "config_fields_output": state.get("config_fields_output"),
        "current_config_item": state.get("current_config_item"),
    }
    edited = interrupt(payload)
    if isinstance(edited, dict) and "config_fields" in edited:
        fields = edited["config_fields"]
        logger.info("User provided updated fields")
    elif isinstance(edited, dict):
        fields = edited
        logger.info("User confirmed fields without changes")
    else:
        fields = state.get("config_fields_output") or {}
        logger.info("Human review completed")
    return {
        "config_fields_output": fields,
        "human_review_status": "reviewed",
        "last_interrupt_kind": "review_fields",
        "events": [{"node": "human_review", "received": True}],
    }


def human_repo_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== REPO CONFIRMATION STAGE ===")
    s = get_settings()
    repo = state.get("repo_url") or s.git_repo_url
    target_branch = state.get("target_branch") or s.git_default_branch
    logger.info("Current repo: %s, Target branch: %s", repo, target_branch)
    logger.info("Pausing for repo confirmation")
    
    payload = {
        "kind": "confirm_repo",
        "repo_url": repo,
        "target_branch": target_branch,
        "message": (
            "Confirm code generation. Override repo_url: use an https/git URL to clone and push a new branch, "
            "or a filesystem path (e.g. ./my-infra-out or C:/infra-out) to write files locally only (no GitHub push)."
        ),
    }
    conf = interrupt(payload)
    if not isinstance(conf, dict):
        conf = {}
    if "repo_url" in conf:
        repo = conf["repo_url"]
        logger.info("User provided override repo_url: %s", repo)
    branch = conf.get("target_branch") or state.get("target_branch") or s.git_default_branch
    confirm = conf.get("confirm", True)
    logger.info("Repo confirmation received. Confirmed: %s", confirm)
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
                    "6. Do not create duplicate terraform blocks across files\n"
                    "Before generating Terraform:\n"
                    "- Ensure Kubernetes version is currently supported by AWS EKS\n"
                    "- Use latest stable versions of Terraform modules unless specified\n"
                    "- Avoid deprecated provider/module versions\n"
                    # "After generating:\n"
                    # "- Simulate terraform plan mentally\n"
                    # "- Identify dependency or lifecycle issues\n"
                    # "- Fix them before output\n"
                    "After generating Terraform:\n"
                    "1. Validate structure (files, blocks)\n"
                    "2. Validate AWS compatibility (versions, services)\n"
                    "3. Identify:\n"
                    "   - deprecated versions\n"
                    "   - duplicate blocks\n"
                    "   - missing best practices\n"
                    "4. Fix issues automatically\n"
                    "5. Output final corrected version\n"
                    "Before final output, verify:\n"
                    "- No hardcoded availability zones\n"
                    "- Cluster endpoint access is explicitly defined\n"
                    "- Cost-impacting resources (NAT Gateway) are intentional\n"
                    "- Tags are consistently applied\n"
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
    logger.info("=== CODEGEN STAGE ===")
    item = state.get("current_config_item") or {}
    fields = state.get("config_fields_output") or {}
    artifact = item.get("type") or "k8s_deployment"
    logger.info("Generating code for artifact: %s", artifact)
    
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
        logger.debug("Using mock codegen")
        text = _mock_codegen_text()
    else:
        try:
            logger.debug("Invoking LLM for code generation")
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
            logger.info("Code generation completed")
        except Exception as e:  # noqa: BLE001
            logger.exception("Codegen LLM failed for artifact: %s", artifact)
            error_payload = {
                "kind": "codegen_error",
                "message": f"Codegen failed for artifact '{artifact}': {e}",
                "error": str(e),
                "artifact": artifact,
                "fields": fields
            }
            edited = interrupt(error_payload)
            if edited and edited.get("retry"):
                logger.info("Retrying code generation after user review")
                # Retry the LLM call
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
                logger.info("Code generation retry completed")
            else:
                raise RuntimeError(f"Codegen aborted: {e}")

    files = _parse_generated_files(text)
    logger.info("Parsed %d files from codegen output", len(files))
    
    tmp = Path(tempfile.mkdtemp(prefix="infra-ai-codegen-"))
    written_fmt: list[tuple[str, str]] = []
    try:
        for rel, content in files:
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            logger.debug("Generated file: %s", rel)
        
        tf_paths = [p for p in tmp.rglob("*.tf") if p.is_file()]
        if tf_paths:
            logger.info("Running terraform fmt on %d files", len(tf_paths))
            _terraform_fmt(tf_paths)
        
        for rel, _ in files:
            p = tmp / rel
            if p.is_file():
                written_fmt.append((rel, p.read_text(encoding="utf-8")))
        logger.info("Generated files ready for push: %d files", len(written_fmt))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "generated_files": [{"path": r, "content": c} for r, c in written_fmt],
        "events": [{"node": "codegen", "files": len(written_fmt)}],
    }


def git_push_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== GIT PUSH STAGE ===")
    item = state.get("current_config_item") or {}
    gfs = state.get("generated_files") or []
    files = [(g["path"], g["content"]) for g in gfs]
    prefix = str(item.get("id") or "config").replace("/", "-")[:40]
    logger.info("Pushing %d files to git. Prefix: %s", len(files), prefix)
    
    svc = GitService(
        repo_url=state.get("repo_url"),
        default_branch=state.get("target_branch"),
        remote_name=state.get("git_remote_name"),
    )
    branch, messages = svc.push_files(files, branch_prefix=prefix)
    logger.info("Git push result. Branch: %s. Messages: %s", branch, messages)
    
    # Check if GitHub API push failed
    push_failed = any("GitHub API push failed" in msg for msg in messages)
    if push_failed:
        error_msg = next(msg for msg in messages if "GitHub API push failed" in msg)
        logger.error("GitHub API push failed: %s", error_msg)
        
        # Extract the actual error from the message
        error_payload = {
            "kind": "git_push_error",
            "message": f"GitHub API push failed. Local copy saved. You can retry after fixing the issue.",
            "error": error_msg,
            "repo_url": state.get("repo_url"),
            "target_branch": state.get("target_branch"),
            "branch": branch,
            "messages": messages,
        }
        edited = interrupt(error_payload)
        if edited and edited.get("retry"):
            logger.info("User requested retry for git push")
            # Retry push with same branch and files
            svc = GitService(
                repo_url=state.get("repo_url"),
                default_branch=state.get("target_branch"),
                remote_name=state.get("git_remote_name"),
            )
            branch, messages = svc.push_files(files, branch_prefix=prefix)
            logger.info("Git push retry completed. Branch: %s", branch)
            # Check again if retry succeeded
            if any("GitHub API push failed" in msg for msg in messages):
                logger.error("Git push retry still failed")
                raise RuntimeError("GitHub API push failed even after retry. Check credentials and network connectivity.")
        else:
            logger.error("User did not retry git push")
            raise RuntimeError(f"GitHub API push failed: {error_msg}")
    
    logger.info("Git push completed successfully")
    
    # Create PR if it's a remote GitHub repo
    pr_url = svc.create_pull_request(
        head_branch=branch,
        title=f"InfraAi: {prefix} configs",
        body=f"Generated infrastructure configs for {item.get('description', item.get('id', 'config'))}\n\nBranch: {branch}"
    )
    if pr_url:
        messages.append(f"PR created: {pr_url}")
        logger.info("PR created: %s", pr_url)
    
    return {
        "last_git_branch": branch,
        "last_pr_url": pr_url,
        "events": [{"node": "git_push", "messages": messages, "branch": branch, "pr_url": pr_url}],
    }


def human_continue_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== CONTINUE PROMPT STAGE ===")
    item = state.get("current_config_item") or {}
    desc = item.get("description") or item.get("id")
    logger.info("Asking user to continue after processing: %s", desc)
    
    pr_url = state.get("last_pr_url")
    message = f'Code generated for "{desc}".'
    if pr_url:
        message += f" PR: {pr_url}."
    message += " Continue with next config file?"
    
    payload = {
        "kind": "continue_next",
        "message": message,
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
        logger.info("User chose to continue. Next index: %d", idx + 1)
    else:
        updates["workflow_status"] = "completed"
        logger.info("User chose to stop. Workflow will be finalized.")
    return updates


def route_after_continue(state: InfraGraphState) -> Literal["loop", "finalize"]:
    if state.get("workflow_status") == "completed":
        return "finalize"
    return "loop"


def finalize_node(state: InfraGraphState) -> dict[str, Any]:
    status = state.get("workflow_status", "completed")
    logger.info("=== FINALIZE STAGE ===")
    logger.info("Workflow status: %s", status)
    logger.info("Workflow execution completed successfully")
    return {"events": [{"node": "finalize", "status": status}]}
