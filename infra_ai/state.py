import operator
from typing import Annotated, Any, Literal, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InfraGraphState(TypedDict, total=False):
    """Main LangGraph state."""

    messages: Annotated[list[BaseMessage], add_messages]

    raw_user_text: str
    raw_user_configs: dict[str, Any]
    requirement_analysis: dict[str, Any]
    config_plan: list[dict[str, Any]]
    current_config_index: int
    current_config_item: dict[str, Any]
    config_fields_output: dict[str, Any]
    repo_url: str
    target_branch: str
    git_remote_name: str
    human_review_status: str
    generated_files: list[dict[str, Any]]
    last_git_branch: str
    last_pr_url: str
    last_interrupt_kind: Literal["review_fields", "confirm_repo", "continue_next", ""] | str
    workflow_status: Literal["running", "completed", "aborted"] | str
    events: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: List[Dict[str, Any]]
    tool_call_count: int

    # Repo-aware codegen (cached across loop iterations; not cleared by clear_messages)
    repo_head_commit: str
    repo_context: dict[str, Any]
    """Snapshot: repo_head_commit, repo_tree_paths, repo_summary, reusable_modules."""
    relevant_repo_files: list[dict[str, Any]]
    """Ranked paths: path, score, reason, excerpt optional."""
    codegen_plan: dict[str, Any]
    """create_paths, update_paths, reuse_modules, notes."""
    repo_context_status: str
    """ok, skipped_no_repo, error."""
    repo_context_source: str
    """cache_sqlite, fresh_clone, fresh_local, state_reuse."""
    repo_rag_context_text: str
    """Token-bounded text injected into codegen (may be truncated)."""
    repo_context_events: Annotated[list[dict[str, Any]], operator.add]
