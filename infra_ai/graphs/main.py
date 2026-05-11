import sqlite3

from functools import lru_cache
from typing import Any
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode #, tools_condition
from langchain_core.messages import ToolMessage

# Initialize sqlite connection for checkpoints
_sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)

from infra_ai.graphs.infra_subgraph import build_infra_subgraph
from infra_ai.graphs.repo_context_subgraph import build_repo_context_subgraph
from infra_ai.logging_config import get_logger
from infra_ai.nodes.analysis_nodes import (
    config_analysis_node,
    requirement_analysis_node,
)
from infra_ai.nodes.loop_nodes import (
    clear_messages_node,
    loop_entry_node,
    route_after_loop,
)
from infra_ai.nodes.infra_nodes import (
    human_repo_node,
    human_review_node,
)
from infra_ai.nodes.codegen_nodes import (
    codegen_node,
    git_push_node,
    route_after_git_push
)
from infra_ai.nodes.continuation_nodes import (
    human_continue_node,
    route_after_continue,
    finalize_node,
)
from infra_ai.state import InfraGraphState
from infra_ai.nodes.tools import global_tools_loader

logger = get_logger(__name__)


def _codegen_tools_with_error_handling(tools_node: ToolNode) -> Any:
    """Wrapper around ToolNode that catches errors and returns them in state.
    
    If a tool invocation fails (e.g., MCP server error), the error is captured
    and returned as a ToolMessage so the workflow can handle it gracefully.
    """
    def wrapped_node(state: InfraGraphState) -> dict[str, Any]:
        try:
            result = tools_node.invoke(state)
            return result
        except Exception as e:
            logger.exception("Tool execution failed in codegen_tools")
            messages = state.get("messages", [])
            error_msg = ToolMessage(
                content=f"Tool invocation error: {str(e)}",
                tool_call_id="error",
                name="error",
            )
            return {
                "messages": messages + [error_msg],
                "workflow_status": "tool_error",
            }
    return wrapped_node



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
    workflow.add_node("clear_messages", clear_messages_node)
    workflow.add_node("repo_context", build_repo_context_subgraph())
    workflow.add_node("codegen", codegen_node)

    # tools = ToolsLoader()._load_all_tools()
    tools = global_tools_loader._load_all_tools()
    tools_node = ToolNode(tools)
    wrapped_tools_node = _codegen_tools_with_error_handling(tools_node)
    workflow.add_node("codegen_tools", wrapped_tools_node)
    
    workflow.add_node("git_push", git_push_node)
    workflow.add_node("human_continue", human_continue_node)
    workflow.add_node("finalize", finalize_node)
    logger.debug("Added %d nodes to graph", 11)

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
    workflow.add_edge("human_repo", "clear_messages")
    workflow.add_edge("clear_messages", "repo_context")
    workflow.add_edge("repo_context", "codegen")
    from infra_ai.nodes.tools import tools_condition
    
    def route_after_codegen(state: InfraGraphState) -> str:
        """Route after codegen: check for tools or end."""
        try:
            result = tools_condition(state)
            return result
        except ValueError:
            # No messages in state; end the workflow
            return "__end__"
    
    workflow.add_conditional_edges(
        "codegen",
        route_after_codegen,
        {
            "tools": "codegen_tools",
            "__end__": "git_push"
        }
    )
    # After tools execution (success or error), loop back to codegen
    # This allows codegen to handle the tool result or retry if needed
    workflow.add_edge("codegen_tools", "codegen")
    # workflow.add_edge("codegen", "git_push")
    workflow.add_conditional_edges(
        "git_push",
        route_after_git_push,
        {"codegen": "codegen", "human_continue": "human_continue"},
    )
    # workflow.add_edge("git_push", "human_continue")
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
