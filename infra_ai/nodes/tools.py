import json
import asyncio

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection

class ToolsLoader:
    def __init__(self, tools_config_path: str = "infra_ai/nodes/tools_config.json"):
        self.config_path = tools_config_path

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
            if connection:
                tools = await load_mcp_tools(session=None, connection=connection)
                all_tools.extend(tools)
        return all_tools

    def load_mcp_servers_sync(self):
        """Synchronous wrapper for loading MCP servers."""
        return asyncio.run(self.load_mcp_servers_from_config())
    
    def _load_all_tools(self):
        # load custom tools and mcp tools
        return self._load_mcp_tools()

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
           # If the tool is async-only, run its async method in a new event loop
           if not hasattr(tool, "invoke"):
               return asyncio.run(tool.ainvoke(kwargs))
           return tool.invoke(kwargs)
       return sync_wrapper

    def _make_async_wrapper(self, tool):
       async def async_wrapper(**kwargs):
           # Directly call the async method
           return await tool.ainvoke(kwargs)
       return async_wrapper