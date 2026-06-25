from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain


from langsmith import traceable, Client,trace
client = Client()
api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["GOOGLE_API_KEY"] = str(api_key)
os.environ["LANGCHAIN_API_KEY"] = str(langchain_api_key)
os.environ["LANGCHAIN_TRACING_V2"] = str(langsmith_tracing)
chat_history = []


with trace("Notebook RAG Learning Session", run_type="chain") as run:
    loader = WebBaseLoader("https://www.bahria.edu.pk/Home/AcademicRoadmapDetails?roadmapId=46")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document = text_splitter.split_documents(docs)

    # Core Components
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash") # Note: Adjusted to valid model name
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001") 
    
    # Vector and Retriever Setup
    vector = FAISS.from_documents(document, embeddings) 
    retriever = vector.as_retriever()
    
    # History Aware Retriever Configuration
    history_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)
    
    # Core QA Document Chain Setup
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Final Retrieval Chain Combine
    final_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    # --- TURN 1 ---
    input_1 = "is this contain Introduction to AI?"
    output = final_retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": input_1
    })
    
    # Append History
    chat_history.append(HumanMessage(content=input_1))
    chat_history.append(AIMessage(content=output["answer"]))
    
    # --- TURN 2 ---
    input_2 = "In which semester i can choose it?"
    output_turn_3 = final_retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": input_2
    })
    
    print("Turn 3 Response:", output_turn_3["answer"])