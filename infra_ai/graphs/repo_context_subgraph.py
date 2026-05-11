"""Subgraph: resolve repo HEAD, cache/analyze tree+modules, persist, plan + bounded RAG text."""

from langgraph.graph import END, START, StateGraph

from infra_ai.nodes.repo_context_nodes import (
    repo_context_analyze_node,
    repo_context_head_node,
    repo_context_persist_node,
    repo_context_plan_rag_node,
    repo_context_resolve_node,
    route_after_repo_context_head,
    route_after_repo_context_resolve,
)
from infra_ai.state import InfraGraphState


def build_repo_context_subgraph():
    g = StateGraph(InfraGraphState)
    g.add_node("rc_head", repo_context_head_node)
    g.add_node("rc_resolve", repo_context_resolve_node)
    g.add_node("rc_analyze", repo_context_analyze_node)
    g.add_node("rc_persist", repo_context_persist_node)
    g.add_node("rc_plan_rag", repo_context_plan_rag_node)

    g.add_edge(START, "rc_head")
    g.add_conditional_edges(
        "rc_head",
        route_after_repo_context_head,
        {"rc_plan_rag": "rc_plan_rag", "rc_resolve": "rc_resolve"},
    )
    g.add_conditional_edges(
        "rc_resolve",
        route_after_repo_context_resolve,
        {"rc_plan_rag": "rc_plan_rag", "rc_analyze": "rc_analyze"},
    )
    g.add_edge("rc_analyze", "rc_persist")
    g.add_edge("rc_persist", "rc_plan_rag")
    g.add_edge("rc_plan_rag", END)
    return g.compile()
