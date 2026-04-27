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
    tool_call_logs: List[Dict[str, Any]]
