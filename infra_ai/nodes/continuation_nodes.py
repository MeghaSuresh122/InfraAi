import logging
from typing import Any, Literal

from langgraph.types import interrupt

from infra_ai.state import InfraGraphState

logger = logging.getLogger(__name__)

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
