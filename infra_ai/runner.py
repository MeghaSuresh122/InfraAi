"""Helpers to invoke the compiled graph and normalize interrupts (v1 API)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import Command

from infra_ai.graphs.main import build_app_graph

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
    config: dict[str, Any] = {"configurable": {"thread_id": tid}}
    graph = build_app_graph()
    result = graph.invoke(payload, config)
    if isinstance(result, dict) and INTERRUPT_KEY in result:
        ints = _serialize_interrupts(result.pop(INTERRUPT_KEY))
        return tid, result, ints
    return tid, result if isinstance(result, dict) else {}, []


def resume_run(
    thread_id: str,
    resume: Any,
    *,
    update: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    graph = build_app_graph()
    cmd = Command(resume=resume, update=update or None)
    result = graph.invoke(cmd, config)
    if isinstance(result, dict) and INTERRUPT_KEY in result:
        ints = _serialize_interrupts(result.pop(INTERRUPT_KEY))
        return result, ints
    return result if isinstance(result, dict) else {}, []
