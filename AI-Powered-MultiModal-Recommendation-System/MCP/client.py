# Libraries for MCP client, LLM handling, and async operations
from ast import arguments
import asyncio
import json
from pathlib import Path
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    Root,
    TextContent,
    CreateMessageResult,
    CreateMessageRequestParams,
)

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
PROJECT_DIR = Path(__file__).parent.resolve()

# StdioServerParameters launches "python server.py" via stdin/stdout
server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)


# Roots And Sampling
def list_roots() -> list[Root]:
    """Limit the server's file access to this project directory."""
    return [Root(uri=f"file://{PROJECT_DIR}", name=PROJECT_DIR.name)]

anthropic_client = Anthropic()

async def handle_sampling(params: CreateMessageRequestParams)  -> CreateMessageResult:
    """Run a Claude.LLm call on behalf of the server and return the result."""
    prompt = params.messages[0].content.text

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=params.maxTokens or 200,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text
    print(f"  LLM Response: {response_text[:100]}...")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model="claude-sonnet-4-20250514",
    )

async def call_tool(tool_name: str, arguments: str) -> str:
        """Call a tool on the server and return its output."""
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write, sampling_callback=handle_sampling, list_roots_callback=list_roots) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments = arguments)

                return json.loads(result.content[0].text)


# CONNECTION & DISCOVERY
async def verify_connection():
    """Connect to the server and verify all expected tools and resources exist."""
    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()

            # list_tools() sends a "tools/list" JSON-RPC request to the server
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            print("--- START SCREENSHOT ---")
            print(f"\nDiscovered {len(tool_names)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")

            assert "get_restaurant_info" in tool_names, "FAIL: get_restaurant_info not found!"
            assert "recommend_by_vibe" in tool_names, "FAIL: recommend_by_vibe not found!"
            assert "get_review" in tool_names, "FAIL: get_review not found!"
            print("\nAll required tools verified!")

            # list_resources() discovers data endpoints the server exposes
            resources_result = await session.list_resources()
            print(f"\nDiscovered {len(resources_result.resources)} resources:")
            for resource in resources_result.resources:
                print(f"  - {resource.uri}: {resource.name}")

            roots = list_roots()
            print(f"\nConfigured {len(roots)} roots:")
            for root in roots:
                print(f"  - {root.name}: {root.uri}")

            print("--- END SCREENSHOT ---")
