# Standard library imports
import asyncio
import os
from dotenv import load_dotenv
# Third-party imports for MCP (Model Context Protocol) and LangGraph
from langchain_mcp_adapters.client import MultiServerMCPClient # Connects to MCP servers
from langgraph.prebuilt import create_react_agent # Creates ReAct-style agents
from langgraph.checkpoint.memory import InMemorySaver # Provides conversation memory
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

async def main():
    """
    Main function that sets up and runs an ai agent with access to multiple MCP servers.
    The agent can access Context7 library documnetation and Met museam collections.
    """
    # Configure MCP servers
    # these servers provide tools that the AI agent can use
    client = MultiServerMCPClient(
        {
            # Context7 server - provides access to library documentation
            "context7": {
                "url": "https://mcp.context7.com/mcp",        # Server endpoint
                "transport": "streamable_http",                # Communication protocol
            },
            # Met Museum server - provides access to museum collection data
            "met-museum": {
                "command": "npx",                              # Node.js package runner
                "args": ["-y", "metmuseum-mcp"],              # Install and run met museum MCP
                "transport": "stdio",                         # Communication via stdin/stdout
            }
        }
    )
   
    # initialize the GenAI client with your API key from environment variables
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    tools = await client.get_tools()

    checkpointer = InMemorySaver()  # Memory saver for conversation history

    config = {
    "configurable": {
        "thread_id": "conversation_id"
    }
}

    # ReAct agent
    agent = create_react_agent(
        model=llm, 
        tools=tools, 
        checkpointer=checkpointer,
    )

    response = await agent.ainvoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can access library documentation and museum collections."
            },
            {
                "role": "user",
                "content": "I want to create a new MCP server using the fastmcp python framework. Can you provide me with the necessary steps and resources?"
            }
        ]
    },
    config=config
)

    print(response["messages"][-1].content) # Print the assistant's response
    while True:
        # Display menu options to the user
        choice = input("""
            Menu:

            Ask the agent a question

            Quit
            Enter your choice (1 or 2): """)
        if choice == "1":
            # Get user's question
            print("Your question")
            query = input("> ")
            # Send the user's question to the agent
            # The agent will have access to the full conversation history
            response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                },
                config=config
            )
            # Display the agent's response
            print(response["messages"][-1].content)
        else:
            # Exit the program for any choice other than "1"
            print("Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())