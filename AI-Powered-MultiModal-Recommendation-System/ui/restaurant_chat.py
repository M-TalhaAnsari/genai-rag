"""
ui/restaurant_chat.py
-----------------------
Gradio chat interface for the MCP-powered restaurant assistant.

The ReAct agent loop discovers tools from mcp_server.py at startup and
calls them as needed to answer user questions about California restaurants.

LLM stack: Groq (primary) → Gemini (fallback).

Run:
    python ui/restaurant_chat.py
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from fastmcp.client import Client, PythonStdioTransport
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_service" / "mcp_server.py")

SYSTEM_PROMPT = (
    "You are a Connoisseur Companion, an AI assistant that helps users discover "
    "restaurants in California based on their preferences, cuisine types, and dining "
    "vibes. You have access to tools that provide structured restaurant information "
    "and user reviews. Use these tools to answer user queries accurately.\n"
    "When responding, consider the restaurant's cuisine, ambiance, and reviews. "
    "If you cannot find a match, suggest alternatives or ask for more details. "
    "Always provide clear and concise information."
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


def _make_model(tools):
    """Return a model bound to the provided OpenAI-style tool schemas."""
    try:
        return groq_llm.bind_tools(tools)
    except Exception:
        return gemini_llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

async def chat_with_agent(user_message: str, history: list) -> str:
    """Connect to MCP server, discover tools, and run a ReAct loop."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        mcp_tools = await client.list_tools()

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

        model = _make_model(openai_tools)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        for _ in range(10):
            response = await model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                return str(raw)

            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = (
                    " ".join(
                        item.text if hasattr(item, "text") else str(item)
                        for item in result.content
                    )
                    if result.content
                    else "(no result)"
                )
                messages.append(
                    ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                )

    return "I wasn't able to complete that request. Please try again."


# ---------------------------------------------------------------------------
# Gradio event handler
# ---------------------------------------------------------------------------

async def handle_chat(user_message: str, history: list):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking…"},
    ]
    yield history

    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Connoisseur Companion", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Connoisseur Companion\n"
        "Your AI guide to California's restaurant scene. "
        "Ask me about restaurants by name, cuisine, or vibe!"
    )

    chatbot = gr.Chatbot(height=500, type="messages")
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g. "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants", size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot])
    btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot], [chatbot])
    btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot], [chatbot])


if __name__ == "__main__":
    print("Starting Connoisseur Companion…")
    demo.launch(share=True)
