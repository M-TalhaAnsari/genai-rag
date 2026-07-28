"""
mcp_service/mcp_client.py
---------------------------
MCP client that connects to mcp_server.py via stdio transport.

Responsibilities:
  - Exposes list_roots() to limit server file access to this project.
  - Handles sampling callbacks using Groq (primary) / Gemini (fallback).
  - Provides call_tool() and verify_connection() utilities.

Usage:
    python -m mcp_service.mcp_client
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Root, TextContent, CreateMessageResult, CreateMessageRequestParams

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")
PROJECT_DIR = Path(__file__).parent.parent.resolve()

server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)

# ---------------------------------------------------------------------------
# LLM setup – Groq primary, Gemini fallback
# ---------------------------------------------------------------------------

groq_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

gemini_llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0.7,
)


def _invoke_llm(prompt: str, max_tokens: int) -> str:
    """Call Groq; fall back to Gemini on any error."""
    try:
        response = groq_llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Groq failed ({e}), switching to Gemini…")
        response = gemini_llm.invoke([HumanMessage(content=prompt)])
        return response.content


# ---------------------------------------------------------------------------
# MCP callbacks
# ---------------------------------------------------------------------------

def list_roots() -> list[Root]:
    """Restrict the server's file access to this project directory."""
    return [Root(uri=f"file://{PROJECT_DIR}", name=PROJECT_DIR.name)]


async def handle_sampling(params: CreateMessageRequestParams) -> CreateMessageResult:
    """Run an LLM call on behalf of the server and return the result."""
    prompt = params.messages[0].content.text
    response_text = _invoke_llm(prompt, params.maxTokens or 200)
    print(f"  Sampling response: {response_text[:100]}…")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model="groq/llama-3.3-70b-versatile",
    )


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------

async def call_tool(tool_name: str, tool_arguments: dict) -> dict:
    """Call a tool on the MCP server and return its parsed JSON output."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=tool_arguments)
            return json.loads(result.content[0].text)


async def verify_connection():
    """Connect to the server and verify all expected tools and resources exist."""
    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"\nDiscovered {len(tool_names)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:80]}…")

            required = {"get_restaurant_info", "recommend_by_vibe", "get_review"}
            missing = required - set(tool_names)
            assert not missing, f"Missing tools: {missing}"
            print("\nAll required tools verified.")

            resources_result = await session.list_resources()
            print(f"\nDiscovered {len(resources_result.resources)} resources:")
            for r in resources_result.resources:
                print(f"  - {r.uri}: {r.name}")

            roots = list_roots()
            print(f"\nConfigured {len(roots)} roots:")
            for root in roots:
                print(f"  - {root.name}: {root.uri}")


if __name__ == "__main__":
    asyncio.run(verify_connection())
