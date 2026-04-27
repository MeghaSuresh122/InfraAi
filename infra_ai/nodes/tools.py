import json
import asyncio

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, StreamableHttpConnection
from infra_ai.nodes.tools_logger import tool_logger

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
            tool_logger.log(tool.name, kwargs)
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
            tool_logger.log(tool.name, kwargs)
            # Directly call the async method
            return await tool.ainvoke(kwargs)
        return async_wrapper

# Initialize global tools loader and load tools
global_tools_loader = ToolsLoader()