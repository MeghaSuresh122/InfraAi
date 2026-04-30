from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.types import interrupt

from infra_ai.config import get_settings
from infra_ai.llm.factory import get_chat_model
from infra_ai.nodes.llm_utils import invoke_structured, mock_llm_enabled
from infra_ai.schemas.config_plan import ConfigPlan, ConfigPlanItem
from infra_ai.schemas.requirements import RequirementAnalysis
from infra_ai.state import InfraGraphState

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
            if analysis.application_type is None:
                raise Exception("Application type could not be identified by LLM: Retry or try changing the model")
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
                f"Requirements:\n{json.dumps(req, indent=2)}\n\n"
                "Only return a JSON object matching the schema."
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
                    f"Requirements:\n{json.dumps(req, indent=2)}\n\n"
                    "Only return a JSON object matching the schema."
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
