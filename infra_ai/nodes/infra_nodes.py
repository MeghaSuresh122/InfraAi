import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from infra_ai.config import get_settings
from infra_ai.llm.factory import get_chat_model
from infra_ai.milvus_store import query_skill_chunks
from infra_ai.nodes.llm_utils import extract_json_object, mock_llm_enabled
from infra_ai.schemas.fields import ConfigFieldsEnvelope
from infra_ai.skills.loader import load_skill_markdown
from infra_ai.state import InfraGraphState
from infra_ai.validation.deterministic import validate_config_fields
from infra_ai.validation.plugins import run_plugins

logger = logging.getLogger(__name__)

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
                "Return ONLY a JSON object of fields. Return object must be a valid json object. Do not include any explanations, only return the JSON object."
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
    
    # Validate fields using Pydantic before returning
    try:
        validated_fields = ConfigFieldsEnvelope.model_validate({"fields": fields})
        logger.debug("Fields validated successfully via ConfigFieldsEnvelope")
        return {
            "config_fields_output": fields,
            "events": [{"node": "infra_builder", "artifact": artifact_type}],
        }
    except Exception as ve:
        logger.error("Pydantic validation failed for fields: %s.", ve)
        if fields.get("type", "") == "text" and "text" in fields:
            parsed_fields = extract_json_object(str(fields["text"]))
            if not parsed_fields:
                raise ValueError("No JSON in builder response")
            fields = parsed_fields
            try:
                validated_fields = ConfigFieldsEnvelope.model_validate({"fields": fields})
                logger.debug("Fields validated successfully via ConfigFieldsEnvelope in second attempt after extracting from text")
                return {
                    "config_fields_output": fields,
                    "events": [{"node": "infra_builder", "artifact": artifact_type}],
                }
            except Exception as ve2:
                logger.error("Pydantic validation failed for fields after second attempt: %s.", ve2)
        
        error_payload = {
            "kind": "field_validation_error",
            "message": f"Field validation failed: {ve}",
            "error": str(ve),
            "fields": fields,
            "artifact_type": artifact_type,
        }
        edited = interrupt(error_payload)
        if edited and edited.get("retry"):
            logger.info("Retrying infra builder after field validation error")
            # Retry the entire builder LLM call
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
            # Validate again after retry
            validated_fields = ConfigFieldsEnvelope.model_validate({"fields": fields})
        else:
            raise ValueError(f"Infra builder returned invalid fields that failed validation: {ve}")
    


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
