"""
mcp_service/mcp_client.py
--------------------------
MCP client that connects to mcp_server.py via stdio transport.

Responsibilities:
  - Discovers all tools from the server at startup
  - Handles LLM sampling callbacks using Groq (primary) / Gemini (fallback)
  - Provides call_tool() for programmatic tool calls
  - Provides verify_connection() to confirm server health
  - Provides interactive_chat() — a ReAct loop where the LLM
    decides which tools to call based on user input

Usage:

  Verify server is working:
    python mcp_service/mcp_client.py --verify

  Interactive chat:
    python mcp_service/mcp_client.py --chat

  Programmatic:
    from mcp_service.mcp_client import call_tool
    result = asyncio.run(call_tool("search_restaurants", {"query": "biryani Lahore"}))
"""

import asyncio
import json
import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Root, TextContent, CreateMessageResult, CreateMessageRequestParams

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────

SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")
PROJECT_DIR   = Path(__file__).parent.parent.resolve()

server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)

# ── LLM setup — Groq primary, Gemini fallback ──────────────────────────────

_groq = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

_gemini = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    model="gemini-1.5-flash",
    temperature=0.7,
)


def _invoke_llm_sync(prompt: str, max_tokens: int = 512) -> str:
    """Call Groq synchronously. Fall back to Gemini on any error."""
    try:
        return _groq.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        print(f"[client] Groq failed ({e}), switching to Gemini...")
        return _gemini.invoke([HumanMessage(content=prompt)]).content


# ── MCP callbacks ──────────────────────────────────────────────────────────

def list_roots() -> list[Root]:
    """Restrict server file access to project directory."""
    return [Root(uri=f"file://{PROJECT_DIR}", name=PROJECT_DIR.name)]


async def handle_sampling(params: CreateMessageRequestParams) -> CreateMessageResult:
    """
    Handle sampling requests from the MCP server.
    The server asks the client to run an LLM call on its behalf.
    We use Groq with Gemini fallback.
    """
    prompt = params.messages[0].content.text
    response_text = _invoke_llm_sync(prompt, params.maxTokens or 512)
    print(f"[sampling] {response_text[:80]}...")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model="groq/llama-3.3-70b-versatile",
    )


# ── Core utilities ─────────────────────────────────────────────────────────

async def call_tool(tool_name: str, tool_arguments: dict) -> dict:
    """
    Call a single MCP tool and return its parsed JSON result.

    Args:
        tool_name:       One of: search_restaurants, get_recommendations,
                         submit_feedback, get_user_profile, get_analytics
        tool_arguments:  Dict of arguments matching the tool's schema.

    Returns:
        Parsed dict from the tool's JSON response.

    Example:
        result = await call_tool(
            "search_restaurants",
            {"query": "biryani Lahore", "top_k": 5}
        )
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=tool_arguments)

            raw = result.content[0].text if result.content else "{}"
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw_response": raw}


async def verify_connection() -> bool:
    """
    Connect to the MCP server, discover all tools and resources,
    and confirm everything expected is present.

    Returns True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("Connoisseur MCP — Connection Verification")
    print("=" * 60)

    required_tools = {
        "search_restaurants",
        "get_recommendations",
        "submit_feedback",
        "get_user_profile",
        "get_analytics",
    }

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(
                read, write,
                sampling_callback=handle_sampling,
                list_roots_callback=list_roots,
            ) as session:
                await session.initialize()

                # Tools
                tools_result = await session.list_tools()
                found_tools = {t.name for t in tools_result.tools}

                print(f"\nDiscovered {len(found_tools)} tools:")
                for tool in tools_result.tools:
                    status = "✅" if tool.name in required_tools else "➕"
                    print(f"  {status} {tool.name}: {(tool.description or '')[:70]}...")

                missing = required_tools - found_tools
                if missing:
                    print(f"\n❌ Missing tools: {missing}")
                    return False
                print("\n✅ All required tools present.")

                # Resources
                resources_result = await session.list_resources()
                print(f"\nDiscovered {len(resources_result.resources)} resources:")
                for r in resources_result.resources:
                    print(f"  📦 {r.uri}: {r.name}")

                # Roots
                roots = list_roots()
                print(f"\nConfigured {len(roots)} roots:")
                for root in roots:
                    print(f"  📁 {root.name}: {root.uri}")

                # Quick smoke test — call get_analytics
                print("\nSmoke test: calling get_analytics...")
                result = await session.call_tool("get_analytics", arguments={})
                raw = result.content[0].text if result.content else "{}"
                data = json.loads(raw)
                print(f"  Search volume total: {data.get('search_volume', {}).get('total', 'N/A')}")
                print("✅ Smoke test passed.")

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

    print("\n✅ All checks passed. MCP server is healthy.")
    return True


# ── ReAct chat loop ────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are a helpful restaurant discovery assistant for Pakistan.

You have access to tools that search for restaurants in Lahore, Islamabad,
Karachi, and Rawalpindi. Use them to answer user questions accurately.

Guidelines:
- Use search_restaurants for quick searches.
- Use get_recommendations when the user wants personalised suggestions.
- Use submit_feedback when the user says they liked or disliked a result.
- Always mention restaurant names, cuisine types, and cities in your response.
- If no results are found, suggest rephrasing the query.
- Be concise and friendly.

The user's user_id for this session is: {user_id}
"""


async def interactive_chat(user_id: str = "guest"):
    """
    Run an interactive ReAct chat loop.

    The LLM decides which MCP tools to call based on the user's message,
    calls them via the MCP server, and uses the results to form a response.

    Args:
        user_id: Used for personalisation via feedback history.
    """
    print("\n🍽️  Connoisseur — Restaurant Discovery for Pakistan")
    print("   Type your query, or 'quit' to exit.\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()

            # Discover tools and convert to OpenAI-style schema for LLM
            mcp_tools = await session.list_tools()
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in mcp_tools
            ]

            # Bind tools to LLM
            try:
                model = _groq.bind_tools(openai_tools)
            except Exception:
                model = _gemini.bind_tools(openai_tools)

            messages = [
                SystemMessage(content=CHAT_SYSTEM_PROMPT.format(user_id=user_id))
            ]

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input or user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                messages.append(HumanMessage(content=user_input))

                # ReAct loop — up to 5 tool call rounds
                for _ in range(5):
                    response = await model.ainvoke(messages)
                    messages.append(response)

                    # No tool calls — LLM has a final answer
                    if not response.tool_calls:
                        raw = response.content
                        if isinstance(raw, list):
                            text = " ".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in raw
                            )
                        else:
                            text = str(raw)
                        print(f"\nAssistant: {text}\n")
                        break

                    # Execute each tool call via MCP server
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        # Inject user_id into tools that support it
                        if "user_id" in tool_args or tool_name in (
                            "search_restaurants", "get_recommendations", "submit_feedback"
                        ):
                            tool_args.setdefault("user_id", user_id)

                        print(f"  [calling {tool_name}...]")
                        result = await session.call_tool(tool_name, tool_args)

                        tool_output = (
                            " ".join(
                                item.text if hasattr(item, "text") else str(item)
                                for item in result.content
                            )
                            if result.content else "(no result)"
                        )

                        messages.append(
                            ToolMessage(
                                content=tool_output,
                                tool_call_id=tool_call["id"]
                            )
                        )


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connoisseur MCP Client")
    parser.add_argument("--verify", action="store_true",
                        help="Verify MCP server connection and exit")
    parser.add_argument("--chat", action="store_true",
                        help="Start interactive chat session")
    parser.add_argument("--user-id", default="guest",
                        help="User ID for personalisation (default: guest)")
    args = parser.parse_args()

    if args.verify:
        ok = asyncio.run(verify_connection())
        sys.exit(0 if ok else 1)

    elif args.chat:
        asyncio.run(interactive_chat(user_id=args.user_id))

    else:
        # Default: verify then start chat
        asyncio.run(verify_connection())
        print()
        asyncio.run(interactive_chat(user_id=args.user_id))
