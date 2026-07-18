import asyncio
import logging
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()  

# intialize chat model
async def basic_chat_example():
    # create a chat model instance
    
    llm = ChatModel.from_name(
    "groq:llama-3.3-70b-versatile",
    ChatModelParameters(temperature=0)
)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?")
    ]

    response = await llm.create(messages=messages)

    return response

async def main():
    logging.basicConfig(level=logging.INFO)
    response = await basic_chat_example()
    logging.info(f"Response: {response.messages[-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
