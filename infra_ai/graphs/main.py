from functools import lru_cache

import logging

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize sqlite connection for checkpoints
_sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)

from infra_ai.graphs.infra_subgraph import build_infra_subgraph
from infra_ai.logging_config import get_logger
from infra_ai.nodes.workflow_nodes import (
    codegen_node,
    config_analysis_node,
    finalize_node,
    git_push_node,
    human_continue_node,
    human_repo_node,
    human_review_node,
    loop_entry_node,
    requirement_analysis_node,
    route_after_continue,
    route_after_loop,
)
from infra_ai.state import InfraGraphState
from infra_ai.nodes.tools import ToolsLoader

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def build_app_graph():
    logger.debug("Building application graph")
    workflow = StateGraph(InfraGraphState)
    
    # Add nodes
    workflow.add_node("requirement_analysis", requirement_analysis_node)
    workflow.add_node("config_analysis", config_analysis_node)
    workflow.add_node("loop_entry", loop_entry_node)
    workflow.add_node("infra", build_infra_subgraph())
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("human_repo", human_repo_node)
    workflow.add_node("codegen", codegen_node)

    tools = ToolsLoader()._load_all_tools()
    workflow.add_node("codegen_tools", ToolNode(tools))
    
    workflow.add_node("git_push", git_push_node)
    workflow.add_node("human_continue", human_continue_node)
    workflow.add_node("finalize", finalize_node)
    logger.debug("Added %d nodes to graph", 10)

    # Add edges
    workflow.add_edge(START, "requirement_analysis")
    workflow.add_edge("requirement_analysis", "config_analysis")
    workflow.add_edge("config_analysis", "loop_entry")
    workflow.add_conditional_edges(
        "loop_entry",
        route_after_loop,
        {"infra": "infra", "finalize": "finalize"},
    )
    workflow.add_edge("infra", "human_review")
    workflow.add_edge("human_review", "human_repo")
    workflow.add_edge("human_repo", "codegen")
    workflow.add_conditional_edges(
        "codegen",
        tools_condition,
        {
            "tools": "codegen_tools",
            "__end__": "git_push"
        }
    )
    workflow.add_edge("codegen_tools", "codegen")
    # workflow.add_edge("codegen", "git_push")
    workflow.add_edge("git_push", "human_continue")
    workflow.add_conditional_edges(
        "human_continue",
        route_after_continue,
        {"loop": "loop_entry", "finalize": "finalize"},
    )
    workflow.add_edge("human_continue", END)
    logger.debug("Added edges to graph")

    checkpointer = SqliteSaver(_sqlite_conn)
    checkpointer.setup()
    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Application graph built and compiled successfully")
    return graph


def reset_app_graph_cache() -> None:
    """Clear compiled graph (e.g. in tests)."""
    build_app_graph.cache_clear()
