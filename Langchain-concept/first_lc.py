# this code is the demonstrate a simple way of forming a prompt and using it to chain with a model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()

def main():
    print(demosimple.__doc__)
    demosimple()

def demosimple():
    "This function demonstrate a simple use of LCEL (Langchian Expression Language) to create a custom chain with the prompt and model"

    # create the prompt template 
    prompt = ChatPromptTemplate.from_template("Tell me a few key achivements of {name}")

    # create the llm object
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key= os.getenv("GOOGLE_API_KEY"), temperature =0.5)

    # create the chain
    chain = prompt | model  # lcel - langchain expression language

    # invoke (run) the chain - The chat model returns a message
    print(chain.invoke({"name" : "Abraham Lincoln"}).content)

if __name__ == "__main__":
    main()


