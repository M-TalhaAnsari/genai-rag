# required libraries
import socket
import asyncio

# Create the path where files will be stored
import os
def make_dir():
    if os.path.exists(r"ModelContextProtocol\MCP-Server/path"):
        print("Directory already exists")
    else:
        os.makedirs(r"ModelContextProtocol\MCP-Server/path")

PORT = 8000

def test_port(port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

f"Port {PORT} is available: {not test_port()}"



# PRINT INFORMATION ABOUT THE READ/WRITE STREAMS AND SESSION ID
def print_stream_info(read, write, _sid,verbose=False):
    if verbose:
        print(f"Read stream: {read}")
        print(f"Write stream: {write}")
        print(f"Session ID: {_sid}")

# CREATING A CALCULATOR MCP SERVER

from fastmcp import FastMCP
#FastMCP creates the server instances with a name and instruction
mcp = FastMCP(
    name="Calculator MCP Server",
    instructions="""
    This server provides data analysis tools.
    Call get_average() to analyze numerical data.
"""
)
print("mcp object", mcp)


# TOOLS

# tools are functions that the AI agent can call to perform specific tasks. Similar to langchain tools but networked and discoverable
@mcp.tool
def add(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of `a` and `b`.

    Example:
        >>> add(3, 5)
        8
    """
    return a + b


@mcp.tool
def subtract(a: int, b: int) -> int:
    """
    Subtract one integer from another.

    Args:
        a (int): The number to subtract from.
        b (int): The number to subtract.

    Returns:
        int: The result of `a - b`.

    Example:
        >>> subtract(10, 4)
        6
    """
    return a - b


# RESOURCES

# resources are like filing cabinets that ao systems can open to read information. Think of them as "files" or "data sources"

@mcp.resource("file:///endpoint/{name}")
def return_template_document(name: str) -> str:
    """Read a document by name"""
    return f"Document contents of {name}"

@mcp.resource("file://endpoint2/{name}")
def read_document(name: str) -> str:
    """Read a document by name from the path directory"""
    try:
        # Read from the actual file system path
        with open(f"path/{name}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Document '{name}' not found in path directory"
    except Exception as e:
        return f"Error reading document: {str(e)}"


#  PROMPTS

@mcp.prompt(title="Code review")
def review_code(code: str) -> str:
    return f"Please review the following code:\n\n{code}\n\nProvide feedback and suggestions for improvement."


# In-MEMORY TRANSPORT

from fastmcp import Client
client = Client(mcp)

# We create a client to test our MCP server and call its tool remotely
async def call_add_tool(a: int, b: int):
    async with client:
        result = await client.call_tool("add", {"a": a, "b": b})
        return result

# async def main():
#     result = await call_add_tool(5, 3)
#     print(f"Result of add tool: {result.content[0].text}")

# asyncio.run(main())


# RESOURCES

async def call_resource(name):
    async with client:
        result = await client.read_resource(f"file:///endpoint/{name}")
        return result
    


async def call_resource2(name):
    async with client:
        result = await client.read_resource(f"file://endpoint2/{name}")
        return result


#  HTTP TRANSPORT MCP SERVERS

# http transport allows MCP servers to run as web services that clients connect to via URLs. This is ideal for remote servers, cloud deployments, or when we want multiple clients to share the same server instance.

# starting the MCP server as an HTTP service running in the backgrounf. Using the mcp object we call the method mcp.run_http_async()
# 
#  server_task = asyncio.create_task(server_task)
asyncio.create_task(mcp.run_http_async(port = PORT))
print(f"HTTP MCP Server started in background on port {PORT}")

# HTTP TRANSPORT AND CLIENT

# http transport
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
transport_http = StreamableHttpTransport(url = f"http://127.0.0.1:{PORT}/mcp")

http_client = Client(transport_http)
print("http_client object", http_client)

async def test_client_http(client: Client, a:int, b:int) -> int:
    async with client:
        result = await client.call_tool("add", {"a": a, "b": b})
        return result.content[0].text

async def main():
    response = await call_resource("README.txt")
    print(response[0].text)

    response = await call_resource2("examples.txt")
    resource = response[0]

    print(f"uri:      {resource.uri}")
    print(f"mimeType: {resource.mimeType}")
    print(f"meta:     {resource.meta}")
    print(f"text:     {resource.text}")

    response = await test_client_http(http_client, 4, 5)
    print(response)



# STDIO TRANSPORT IN MCP SERVERS
from pathlib import Path
from fastmcp import FastMCP

# Create MCP Server
mcp = FastMCP(
    name="CalculatorMCPServer",
    instructions="""
You are a Calculator MCP Server.

Capabilities:
- Add two numbers.
- Subtract two numbers.
- Multiply two numbers.
- Divide two numbers.
- Read text documents from the documents folder.
- Generate a code review prompt.
"""
)

# --------------------------
# TOOLS
# --------------------------

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b


@mcp.tool
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b


@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divides a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


# --------------------------
# RESOURCE
# --------------------------

DOCUMENT_FOLDER = Path(__file__).parent / "documents"


@mcp.resource("file://documents/{name}")
def read_document(name: str) -> str:
    """
    Reads a text document from the documents folder.
    Example URI:
        file://documents/README.txt
    """
    file_path = DOCUMENT_FOLDER / name

    if not file_path.exists():
        return f"Document '{name}' not found."

    return file_path.read_text(encoding="utf-8")


# --------------------------
# PROMPT
# --------------------------

@mcp.prompt(title="Code Review")
def review_code(code: str) -> str:
    """
    Generates a prompt for reviewing source code.
    """
    return f"""
You are an expert software engineer.

Review the following code and provide:

1. Bugs
2. Performance improvements
3. Readability improvements
4. Security concerns
5. Best practices

Code:

{code}
"""


# --------------------------
# MAIN
# --------------------------

if __name__ == "__main__":
    print("Calculator MCP Server is running...")
    mcp.run()

