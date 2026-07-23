import asyncio
import sys
import json
from urllib.parse import quote
from typing import Optional, Dict, Any, List, Union
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult

from anthropic import Anthropic
import anthropic as anthropic_module
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Claude model identifier for API calls
MODEL_ID = "claude-sonnet-4-5-20250929"

# Groq fallback model — strong open-source model with tool-use support
GROQ_MODEL_ID = "llama-3.3-70b-versatile"


class MCPClient:
    """MCP (Model Context Protocol) client for interacting with MCP servers and Claude.

    This client manages connections to MCP servers, handles tool execution,
    and provides an interactive interface for querying Claude with MCP tools.
    Falls back to Groq (OpenAI-compatible) if the Anthropic API is unavailable.
    """

    def __init__(self):
        """Initialize the MCP client with session management and API clients.

        Sets up:
        - AsyncExitStack for managing async context managers
        - Anthropic client for Claude API interactions (primary)
        - Groq client for fallback LLM interactions
        """
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.groq = Groq()  # Reads GROQ_API_KEY from environment automatically

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server via stdio transport.

        Establishes a connection to an MCP server by launching the server script
        as a subprocess and communicating via stdin/stdout.

        Args:
            server_script_path: Path to the server script (.py, .js, or .ts file)

        Raises:
            ValueError: If server_script_path is not a .py, .js, or .ts file
        """
        is_python = server_script_path.endswith('.py')
        is_ts = server_script_path.endswith('.ts')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_ts or is_js):
            raise ValueError("Server script must be a .py, .js, or .ts file")

        self.client = Client(
            server_script_path,
            elicitation_handler=self.handle_elicitation,
            progress_handler=self.handle_progress,
            message_handler=self.handle_message
        )

        await self.exit_stack.enter_async_context(self.client)

    async def handle_elicitation(self, message: str, response_type: type, params, context):
        """Handle elicitation requests from the MCP server."""
        print(f"Server asks: {message}")

        user_data = {}
        for field_name, field_type in response_type.__annotations__.items():
            user_input = input(f"Enter value for '{field_name}' ({field_type.__name__}): ").strip()
            if not user_input:
                return ElicitResult(action="decline")
            user_data[field_name] = user_input

        return response_type(**user_data)

    async def handle_progress(self, progress: float, total: float | None, message: str | None) -> None:
        """Handle progress notifications from the MCP server."""
        if total is not None:
            percentage = (progress / total) * 100
            print(f"Progress: {percentage:.1f}% - {message or ''}")
        else:
            print(f"Progress: {progress} - {message or ''}")

    async def handle_message(self, message):
        """Handle notification messages from the MCP server."""
        if hasattr(message, 'root'):
            method = message.root.method
            print(f"Received: {method}")
            if method == "notifications/tools/list_changed":
                print("Tools have changed - might want to refresh tool cache")
            elif method == "notifications/resources/list_changed":
                print("Resources have changed")

    async def _get_tools(self) -> List[Dict[str, Any]]:
        """Retrieve available tools from the MCP server in Anthropic format."""
        tools_response = await self.client.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "MCP Tool",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response
        ]
        return tools

    def _convert_tools_for_groq(self, anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-format tools to OpenAI/Groq-compatible format.

        Anthropic uses `input_schema`; OpenAI/Groq uses `parameters` nested under `function`.

        Args:
            anthropic_tools: Tool definitions in Anthropic format

        Returns:
            Tool definitions in OpenAI/Groq format
        """
        groq_tools = []
        for tool in anthropic_tools:
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", "MCP Tool"),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
            })
        return groq_tools

    async def _execute_tool_calls(self, tool_calls_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a list of tool calls via MCP and return results.

        Args:
            tool_calls_info: List of dicts with keys: id, name, args

        Returns:
            List of tool result dicts ready to append to the message history
        """
        tool_results = []
        for tc in tool_calls_info:
            try:
                result = await self.client.call_tool(tc["name"], tc["args"])

                if isinstance(result.content, list):
                    result_text = "\n".join([
                        c.text if hasattr(c, 'text') else str(c)
                        for c in result.content
                    ])
                else:
                    result_text = result.content

                tool_results.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "result": result_text,
                    "is_error": False,
                })
            except Exception as e:
                print(f"Error calling tool {tc['name']}: {e}")
                tool_results.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "result": f"Error: {str(e)}",
                    "is_error": True,
                })
        return tool_results

    async def _process_with_anthropic(self, query: str, available_tools: List[Dict[str, Any]]) -> str:
        """Run the agentic loop using the Anthropic (Claude) API.

        Args:
            query: User query string
            available_tools: Tool list in Anthropic format

        Returns:
            Final text response from Claude
        """
        messages = [{"role": "user", "content": query}]

        response = self.anthropic.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
            messages=messages,
            tools=available_tools
        )

        while response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            # Collect tool calls from this response
            tool_calls_info = [
                {"id": block.id, "name": block.name, "args": block.input}
                for block in response.content
                if block.type == "tool_use"
            ]

            tool_results = await self._execute_tool_calls(tool_calls_info)

            # Format results back into Anthropic's expected message structure
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr["id"],
                        "content": tr["result"],
                        **({"is_error": True} if tr["is_error"] else {}),
                    }
                    for tr in tool_results
                ]
            })

            response = self.anthropic.messages.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=messages,
                tools=available_tools
            )

        return "\n".join(
            block.text for block in response.content if hasattr(block, 'text')
        )

    async def _process_with_groq(self, query: str, available_tools: List[Dict[str, Any]]) -> str:
        """Run the agentic loop using the Groq API (OpenAI-compatible).

        Args:
            query: User query string
            available_tools: Tool list in Anthropic format (converted internally)

        Returns:
            Final text response from the Groq model
        """
        print("[Fallback] Anthropic API unavailable — using Groq as fallback.")

        groq_tools = self._convert_tools_for_groq(available_tools)
        messages = [{"role": "user", "content": query}]

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL_ID,
            max_tokens=4096,
            messages=messages,
            tools=groq_tools if groq_tools else None,
            tool_choice="auto" if groq_tools else None,
        )

        # Agentic loop for Groq (OpenAI-style)
        while response.choices[0].finish_reason == "tool_calls":
            assistant_message = response.choices[0].message

            # Append assistant turn (with tool_calls) to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (assistant_message.tool_calls or [])
                ]
            })

            # Parse and execute all tool calls
            tool_calls_info = []
            for tc in (assistant_message.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls_info.append({"id": tc.id, "name": tc.function.name, "args": args})

            tool_results = await self._execute_tool_calls(tool_calls_info)

            # Append each tool result as a separate "tool" role message (OpenAI format)
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["id"],
                    "name": tr["name"],
                    "content": tr["result"],
                })

            response = self.groq.chat.completions.create(
                model=GROQ_MODEL_ID,
                max_tokens=4096,
                messages=messages,
                tools=groq_tools if groq_tools else None,
                tool_choice="auto" if groq_tools else None,
            )

        return response.choices[0].message.content or ""

    async def _get_prompts(self):
        """Retrieve available prompts from the MCP server."""
        return await self.client.list_prompts()

    async def _get_resources(self):
        """Retrieve available resources from the MCP server."""
        return await self.client.list_resources()

    async def _get_resource_templates(self):
        """Retrieve available resource templates from the MCP server."""
        return await self.client.list_resource_templates()

    async def process_query(self, query: str) -> str:
        """Process a query using Claude with MCP tools, falling back to Groq on failure.

        Tries the Anthropic API first. If it raises an authentication error,
        permission error, or any API-level error, automatically retries with Groq.

        Args:
            query: The user's query to process

        Returns:
            The final text response from whichever provider succeeded
        """
        available_tools = await self._get_tools()

        try:
            return await self._process_with_anthropic(query, available_tools)

        except anthropic_module.AuthenticationError:
            print("[Warning] Anthropic authentication failed (invalid or missing API key).")
        except anthropic_module.PermissionDeniedError:
            print("[Warning] Anthropic API access denied.")
        except anthropic_module.APIStatusError as e:
            print(f"[Warning] Anthropic API error {e.status_code}: {e.message}")
        except anthropic_module.APIConnectionError:
            print("[Warning] Could not reach the Anthropic API (connection error).")

        # --- Groq fallback ---
        return await self._process_with_groq(query, available_tools)

    async def converse(self):
        """Start an interactive conversation mode."""
        print("\nEntering conversation mode. Type 'quit' or 'q' to exit.")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() in ("quit", "q"):
                break

            if not query:
                print("Please enter a query")
                continue

            try:
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"Error processing query: {e}")

    async def prompt(self, prompt_name: str):
        """Execute a named prompt template from the MCP server."""
        try:
            prompts_response = await self._get_prompts()
            prompt_obj = next(
                (p for p in prompts_response if p.name == prompt_name), None
            )

            if not prompt_obj:
                print(f"Prompt '{prompt_name}' not found")
                return

            print(prompt_obj)

            arguments = {}
            if prompt_obj.arguments:
                for arg in prompt_obj.arguments:
                    required = "required" if arg.required else "optional"
                    user_input = input(f"{arg.name} ({required}): ").strip()

                    if not user_input and arg.required:
                        print(f"Error: {arg.name} is required")
                        return

                    if user_input:
                        arguments[arg.name] = user_input

            prompt_result = await self.client.get_prompt(prompt_name, arguments=arguments)
            prompt = prompt_result.messages[0].content.text

            response = await self.process_query(prompt)
            print(response)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")

    def _parse_resource(self, resource) -> Any:
        """Parse a FastMCP resource response into a Python object.

        FastMCP may return resource content in several shapes:
          - resource[0].text is a JSON string  → parse it
          - resource[0] has a .data attribute  → use that directly
          - resource itself is already a dict/list → use as-is

        Args:
            resource: Raw resource response from client.read_resource()

        Returns:
            Parsed Python object (dict, list, or str)
        """
        # Try the most common case first: resource[0].text is a JSON string
        item = resource[0] if isinstance(resource, (list, tuple)) else resource

        # Prefer .text (standard MCP TextContent)
        raw = getattr(item, "text", None)

        # Some FastMCP versions expose .data or .contents instead
        if raw is None:
            raw = getattr(item, "data", None)
        if raw is None:
            raw = getattr(item, "contents", None)

        # If we still have nothing, stringify whatever we got
        if raw is None:
            raw = str(item)

        # If it's already a dict/list (FastMCP returned a native object), return directly
        if isinstance(raw, (dict, list)):
            return raw

        # Otherwise try JSON parse, fall back to raw string
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def read_file(self):
        """Read the contents of a file via MCP resource."""
        try:
            file_name = input("Enter file path: ").strip()
            encoded_file_name = quote(file_name, safe="")
            resource = await self.client.read_resource(f"file:///{encoded_file_name}")

            parsed = self._parse_resource(resource)

            # Handle error reported by server
            if isinstance(parsed, dict) and "error" in parsed:
                print(f"Server error: {parsed['error']}")
                return None

            file_content = parsed.get("file_content") if isinstance(parsed, dict) else parsed
            print(f"File Content:\n {file_content}")
            return file_content
        except Exception as e:
            print(f"Error reading file: {e}")

    def _print_dir_listing(self, items: list[dict]):
        """Format and print a directory listing."""
        print("\nDirectory Listing:\n")
        print(f"{'Type':<10} {'Size':>10} {'Modified':<25} {'Name'}")
        print("-" * 70)
        for item in items:
            type_icon = "📁" if item["type"] == "directory" else "📄"
            size = f"{item['size']} B"
            print(f"{type_icon:<2} {item['type']:<8} {size:>10}  {item['modified']:<25} {item['name']}")

    async def read_dir(self):
        """List the contents of the current directory via MCP resource."""
        try:
            resource = await self.client.read_resource("dir://.")
            parsed = self._parse_resource(resource)

            # Debug: uncomment the line below if you hit issues again
            # print(f"[debug] raw resource type={type(parsed)}, value={parsed!r}")

            # Handle server-side errors
            if isinstance(parsed, dict) and "error" in parsed:
                print(f"Server error: {parsed['error']}")
                return

            # FastMCP may return {"items": [...]} or just [...] directly
            if isinstance(parsed, dict):
                dir_list = parsed.get("items")
                if dir_list is None:
                    print(f"Unexpected response shape — keys: {list(parsed.keys())}")
                    print(parsed)
                    return
            elif isinstance(parsed, list):
                dir_list = parsed
            else:
                print(f"Unexpected response type: {type(parsed)}: {parsed}")
                return

            self._print_dir_listing(dir_list)
        except Exception as e:
            print(f"Error reading directory: {e}")

    async def menu(self):
        """Run the main interactive chat loop with menu-driven interface."""
        print("\nMCP Client Started!")
        print("Select from the menu or 'quit'/'q' to exit.")

        menu_actions = {
            "1": lambda: self.prompt("documentation_generator"),
            "2": lambda: self.prompt("code_review"),
            "3": self.read_file,
            "4": self.read_dir,
            "5": self.converse,
            "q": self.quit_action,
            "quit": self.quit_action,
        }

        while True:
            choice = input("""
Select from the Menu
1. Generate Documentation
2. Review Code
3. Read File
4. Read Current Directory
5. Converse with Agent
q. Quit
> """).strip()

            action = menu_actions.get(choice)

            if not action:
                print("Invalid choice. Please try again.")
                continue

            result = await action()
            if result == "quit":
                break

    async def quit_action(self):
        """Signal to exit the client."""
        print("Exiting client...")
        return "quit"

    async def cleanup(self):
        """Clean up resources and close connections."""
        if self.exit_stack:
            await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_path>")
        sys.exit(1)

    client = MCPClient()
    try:
        server_path = sys.argv[1]
        print(f"Connecting to server: {server_path}")
        await client.connect_to_server(server_path)
        await client.menu()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())