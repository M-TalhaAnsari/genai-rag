# using a webpage and answer the question using gemini api key
import os
from google import genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

loader = WebBaseLoader("https://medium.com/swlh/algorithmic-management-what-is-it-and-whats-next-33ad3429330b")
from dotenv import load_dotenv
load_dotenv()
docs = loader.load()

# The recursivecharacter text splitter split the large text based on specific chunk size
# it does this by using a set of chracters. The default characters provided to it are ["\n\n", "\n", " ",""]

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# llm
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(model ="gemini-embedding-001" )

# FAISS (Facebook ai similarity search) is a library that allows developers to store and search
# documents that are similar to each other
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")


document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)


response = retrieval_chain.invoke(
    {"context": "You are a content writer who is creating a LinkedIn blog for Technology enthusiasts.", 
                                   "input": 
                                   """Please write a blog on the given content that sounds professional. 
                                      The blog should be more than 500 words and well structured in distict chapters.
                                      Use any facts, data or statistics available in the given input. 
                                   """,
                                   })


print(response["answer"])