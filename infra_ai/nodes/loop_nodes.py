import logging
from typing import Any, Literal

from langchain_core.messages import RemoveMessage
from langgraph.types import Command

from infra_ai.state import InfraGraphState

logger = logging.getLogger(__name__)


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

def clear_messages_node(state: InfraGraphState):
    logger.info("Clearing messages before next code generation")
    return Command(
        update={
            "messages": [
                RemoveMessage(id=m.id) for m in state.get("messages", [])
            ],  # fully replaces message list,
            "tool_calls": [],
            "tool_call_count": 0
        }
    )
