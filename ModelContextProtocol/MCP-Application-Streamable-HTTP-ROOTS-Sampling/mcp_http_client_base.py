import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import logging

logging.getLogger("mcp").setLevel(logging.WARNING)

class MCPHTTPClient:
    """Base MCP HTTP Client with pure protocol logic"""
    def __init__(self, server_url: str, roots_dir: str) -> None:
        self.server_url = server_url
        self.roots_dir = roots_dir
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._connected = False


    async def connet(self):
        """Connect to HTTP MCP server via Streamablle HTTP. Safe to call multiple times"""
        if self._connected:
            return

        #FastMCP uses /mcpendpoint for streamable HTTP
        mcp_url = f"{self.server_url}/mcp"
        read, write, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(mcp_url)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()
        self._connected = True

    async def list_tools(self):
        """List all available tools from the HTTP server."""
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """Execute a tool on the HTTP server."""
        result = await self.session.call_tool(tool_name, arguments)
        return result

    async def list_resources(self):
        """List all available resource templates from the HTTP server."""
        result = await self.session.list_resource_templates()
        return result.resourceTemplates

    async def read_resource(self, uri: str):
        """Read a resource by URI from the HTTP server."""
        result = await self.session.read_resource(uri)
        return result

    async def list_prompts(self):
        """List all available prompts from the HTTP server."""
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, arguments: dict):
        """Get a rendered prompt template from the HTTP server."""
        result = await self.session.get_prompt(prompt_name, arguments)
        return result

    async def cleanup(self):
        """Clean up resources and close HTTP connection."""
        await self.exit_stack.aclose()
