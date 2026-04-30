"""Helpers to invoke the compiled graph and normalize interrupts (v1 API)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import Command
from langchain_core.messages import RemoveMessage

from infra_ai.graphs.main import build_app_graph
from infra_ai.logging_config import get_logger

logger = get_logger(__name__)
INTERRUPT_KEY = "__interrupt__"


def _serialize_interrupts(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if raw is None:
        return out
    for item in raw:
        if hasattr(item, "value") and hasattr(item, "id"):
            out.append({"id": item.id, "value": item.value})
        elif isinstance(item, dict):
            out.append(item)
    return out


def invoke_until_interrupt(
    payload: dict[str, Any],
    *,
    thread_id: str | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """
    Start or continue a run. Returns (thread_id, state_dict, interrupts_json).
    If interrupts list is non-empty, call `resume_run` with the same thread_id.
    """
    tid = thread_id or str(uuid4())
    logger.info("Starting workflow execution with thread_id: %s", tid)
    config: dict[str, Any] = {"configurable": {"thread_id": tid}}
    graph = build_app_graph()
    try:
        result = graph.invoke(payload, config)
        if isinstance(result, dict) and INTERRUPT_KEY in result:
            ints = _serialize_interrupts(result.pop(INTERRUPT_KEY))
            logger.info("Workflow paused at interrupt. Count: %d", len(ints))
            return tid, result, ints
        logger.info("Workflow completed successfully")
        return tid, result if isinstance(result, dict) else {}, []
    except Exception as e:
        logger.exception("Workflow execution failed with thread_id: %s", tid)
        raise


def resume_run(
    thread_id: str,
    resume: Any,
    *,
    update: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info("Resuming workflow for thread_id: %s", thread_id)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    graph = build_app_graph()
    cmd = Command(resume=resume, update=update or None)
    try:
        result = graph.invoke(cmd, config)
        if isinstance(result, dict) and INTERRUPT_KEY in result:
            ints = _serialize_interrupts(result.pop(INTERRUPT_KEY))
            logger.info("Workflow paused at another interrupt. Count: %d", len(ints))
            return result, ints
        logger.info("Workflow completed after resume")
        return result if isinstance(result, dict) else {}, []
    except Exception as e:
        logger.exception("Workflow resume failed for thread_id: %s", thread_id)
        raise


def retry_run(
    thread_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Retry a run from the last successful checkpoint (e.g. after a node failure).
    Returns (state_dict, interrupts_json).
    """
    logger.info("Retrying workflow for thread_id: %s", thread_id)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    graph = build_app_graph()
    cmd = Command(update={
        "messages": [
            RemoveMessage(id=m.id) for m in graph.get_state(config).values.get("messages", [])
        ],
        "tool_calls": [],
        "tool_call_count": 0
    })
    try:
        # Pass None as input to resume from the last successful checkpoint
        result = graph.invoke(cmd, config)
        if isinstance(result, dict) and INTERRUPT_KEY in result:
            ints = _serialize_interrupts(result.pop(INTERRUPT_KEY))
            logger.info("Workflow paused at interrupt after retry. Count: %d", len(ints))
            return result, ints
        logger.info("Workflow completed after retry")
        return result if isinstance(result, dict) else {}, []
    except Exception as e:
        logger.exception("Workflow retry failed for thread_id: %s", thread_id)
        raise
