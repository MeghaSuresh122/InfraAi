import json
import asyncio

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, StreamableHttpConnection
# from infra_ai.nodes.tools_logger import tool_logger

# Override langgraph.prebuilt.tools_condition
from langchain_core.messages import (
    AnyMessage
)
from typing import (
    Any,
    Literal
)
from pydantic import BaseModel
def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Conditional routing function for tool-calling workflows.

    This utility function implements the standard conditional logic for ReAct-style
    agents: if the last `AIMessage` contains tool calls, route to the tool execution
    node; otherwise, end the workflow. This pattern is fundamental to most tool-calling
    agent architectures.

    The function handles multiple state formats commonly used in LangGraph applications,
    making it flexible for different graph designs while maintaining consistent behavior.

    Args:
        state: The current graph state to examine for tool calls. Supported formats:
            - Dictionary containing a messages key (for `StateGraph`)
            - `BaseModel` instance with a messages attribute
        messages_key: The key or attribute name containing the message list in the state.
            This allows customization for graphs using different state schemas.

    Returns:
        Either `'tools'` if tool calls are present in the last `AIMessage`, or `'__end__'`
            to terminate the workflow. These are the standard routing destinations for
            tool-calling conditional edges.

    Raises:
        ValueError: If no messages can be found in the provided state format.

    Example:
        Basic usage in a ReAct agent:

        ```python
        from langgraph.graph import StateGraph
        from langchain.tools import ToolNode
        from langchain.tools.tool_node import tools_condition
        from typing_extensions import TypedDict


        class State(TypedDict):
            messages: list


        graph = StateGraph(State)
        graph.add_node("llm", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_conditional_edges(
            "llm",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "tools", "__end__": "__end__"},
        )
        ```

        Custom messages key:

        ```python
        def custom_condition(state):
            return tools_condition(state, messages_key="chat_history")
        ```

    !!! note
        This function is designed to work seamlessly with `ToolNode` and standard
        LangGraph patterns. It expects the last message to be an `AIMessage` when
        tool calls are present, which is the standard output format for tool-calling
        language models.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif (isinstance(state, dict) and (messages := state.get(messages_key, []))) or (
        messages := getattr(state, messages_key, [])
    ):
        ai_message = messages[-1]
    else:
        msg = f"No messages found in input state to tool_edge: {state}"
        raise ValueError(msg)
    if ( hasattr(ai_message, "tool_calls")
            and
        len(ai_message.tool_calls) > 0
        #     and
        # state.get("tool_call_count", 0) <= 3 # Custom: max 3 tool calls
    ):
        return "tools"
    return "__end__"

class ToolsLoader:
    def __init__(self, tools_config_path: str = "infra_ai/nodes/tools_config.json"):
        self.config_path = tools_config_path
        self.tools = None

    async def load_mcp_servers_from_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        all_tools = []
        mcp_servers = config.get("mcp_servers", {})
        for name, server_config in mcp_servers.items():
            server_type = server_config.get("type", "stdio")
            connection = None
            if server_type == "stdio":
                connection = StdioConnection(
                command=server_config.get("command"),
                args=server_config.get("args", []),
                )
            elif server_type == "sse":
                connection = SSEConnection(transport="sse", url=server_config.get("url"))
            elif server_type.lower() in ("streamablehttp", "streamable_http", "http"):
                connection = StreamableHttpConnection(transport="streamable_http", url=server_config.get("url"))
            if connection:
                tools = await load_mcp_tools(session=None, connection=connection)
                all_tools.extend(tools)
        return all_tools

    def load_mcp_servers_sync(self):
        """Synchronous wrapper for loading MCP servers."""
        return asyncio.run(self.load_mcp_servers_from_config())
    
    def _load_all_tools(self):
        # load custom tools and mcp tools
        self.tools = self._load_mcp_tools()
        return self.tools

    def _load_mcp_tools(self):
       mcp_tools = self.load_mcp_servers_sync()
       return [self._create_wrapped_tool(tool) for tool in mcp_tools]

    def _create_wrapped_tool(self, tool):
       """Create a tool that works in both sync and async contexts."""
       return StructuredTool(
           name=tool.name,
           description=tool.description,
           args_schema=tool.args_schema,
           func=self._make_sync_wrapper(tool), # For sync calls
           coroutine=self._make_async_wrapper(tool) # For async calls
       )
    
    def _make_sync_wrapper(self, tool):
        def sync_wrapper(**kwargs):
            # tool_logger.log(tool.name, kwargs)
            # tool_calls = kwargs.pop("_tool_calls", None)
            # if tool_calls is not None:
            #     tool_calls.append({
            #         "tool": tool.name,
            #         "args": kwargs
            #     })
            
            # If the tool is async-only, run its async method in a new event loop
            if not hasattr(tool, "invoke"):
                return asyncio.run(tool.ainvoke(kwargs))
            try:
                return tool.invoke(kwargs)
            except NotImplementedError:
                return asyncio.run(tool.ainvoke(kwargs))
        return sync_wrapper

    def _make_async_wrapper(self, tool):
        async def async_wrapper(**kwargs):
            # tool_logger.log(tool.name, kwargs)
            # tool_calls = kwargs.pop("_tool_calls", None)
            # if tool_calls is not None:
            #     tool_calls.append({
            #         "tool": tool.name,
            #         "args": kwargs
            #     })

            # Directly call the async method
            return await tool.ainvoke(kwargs)
        return async_wrapper

# Initialize global tools loader and load tools
global_tools_loader = ToolsLoader()