import warnings 
warnings.filterwarnings('ignore')

from langchain_tavily import TavilySearch
from langchain.tools import tool
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(api_key = os.environ.get("GOOGLE_API_KEY", ""), model = "gemini-2.5-flash")
os.environ["TAVILY_API_KEY"] = "Your_TAVILY_API_KEY"

# Initialize the Tavily search tool
search = TavilySearch()

@tool
def search_tool(query: str):
    """
    Search the web for information using Tavily API.

    :param query: The search query string
    :return: Search results related to the query
    """
    return search.invoke(query)

# search_tool.invoke("What is the weather today in Islamabad?")

# Clothing Recommendation Tool

@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on the provided weather description.

    This function examines the input string for specific keywords or temperature indicators 
    (e.g., "snow", "freezing", "rain", "85°F") to suggest appropriate attire. It handles 
    common weather conditions like snow, rain, heat, and cold by providing simple and practical 
    clothing advice.

    :param weather: A brief description of the weather (e.g., "Overcast, 64.9°F")
    :return: A string with clothing recommendations suitable for the weather
    """
    weather = weather.lower()
    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, and boots"
    elif "rain" in weather or "wet" in weather:
        return "Bring a raincoat and waterproof shoes"
    elif "hot" in weather or "85" in weather:
        return "T-shirt, shorts, and sunscreen recommend"
    elif "cold" in weather or "50" in weather:
        return "Wear a warm jacket or sweater"
    
    else:
        return "A light jacket should be fine"
    
tools = [search_tool, recommend_clothing]

tools_by_name = {tool.name:tool for tool in tools}

# Creating the System prompt 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
"""),
    MessagesPlaceholder(variable_name="scratch_pad")
])

# Binding tools to the model
model_react = chat_prompt | model.bind_tools(tools)

# Agent State
# in ReAct agent management is crucial, as the agent must mentain context across multiple reasoning and acting steps

from typing import (Annotated, Sequence, TypedDict)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

state : AgentState = {"messages": []}


# ReAct with Graph
def tool_node(state: AgentState):
    """Execute all tool calls from the last message in the state."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(state:AgentState):
    """Invoke the model wit the current conversation state"""
    response = model_react.invoke({"scratch_pad": state["messages"]})
    return {"messages":[response]}

# Call the reAct-enabled model
# Pass the full conversation context
# return the model's reponse

def should_continue(state: AgentState):
    """Determine whether to continue with tool use or end the converstation"""
    messages = state["messages"]
    last_message = messages[-1]

    # if there is no function call then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
# COnstructing the state graph
from langgraph.graph import StateGraph, END
workflow = StateGraph(AgentState)

# Define the two node we will cyccle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge("tools", "agent")

# Add conditional logic
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)
# Set entry point
workflow.set_entry_point("agent")

# compile the graph
graph = workflow.compile()

# Visualizing the graph
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# Running the agent complete
def print_stream(stream):
    """Helper function for formatting the stream nicely"""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[HumanMessage(content="What's the weather in islamabad, and what i should wear based on the temperature???")]}
print_stream(graph.stream(inputs, stream_mode="values"))

