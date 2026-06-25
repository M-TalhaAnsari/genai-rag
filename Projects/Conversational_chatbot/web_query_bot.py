# The following application will be used for developing the application:
# 1. A vector database storing the Langchain-docs(FAISS)
# 2. Streamit for developing the chat UI
# 3. A conversational Retrieval chain implementing Conversational Memory
import os
import torch

from dotenv import load_dotenv
from langchain_classic.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from tqdm import tqdm
load_dotenv(override=True)

INDEX_DIR = "faiss_index"
    
def upload_web():
    """
    This function does the following:
    1. Read recursively through the given website
    2. Load the pages
    3. Loaded documents are split into chunks using splitter
    4. These chunks are converted into Language embedding and loaded as vectors into a local FAISS vectors database

    """
    
    loader = DirectoryLoader(
        path="langchain-docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"{len(documents)} pages Loaded")

    # split load document into chunks using character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_documents = text_splitter.split_documents(documents=documents)

    print(f"Split into {len(split_documents)} Documents")
 
    # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("Initializing FastEmbed Engine (ONNX C++ runtime)...")
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("Initializing FAISS database with manual batching...")

    batch_size = 16
    db = None
    
    # This loop ensures your tqdm progress bar works flawlessly
    for i in tqdm(range(0, len(split_documents), batch_size), desc="Encoding & Indexing"):
        batch = split_documents[i : i + batch_size]
        
        if db is None:
            # Create the initial index with the first batch
            db = FAISS.from_documents(batch, embeddings)
        else:
            # Add subsequent batches directly to the existing index in memory
            db.add_documents(batch)

    db.save_local(INDEX_DIR)
    print(f"Done. Saved clean FAISS index with {db.index.ntotal} vectors to '{INDEX_DIR}/'")
 

 

    
def faiss_query():
    "This function does the following thing:"
    "1. Load the local FAISS Database"
    "2. Trigger a semantic similarity search using query"
    "3. This reretrieve semantically matching vectors from the DB"

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    query = "How to cal langchain_community"
    docs = new_db.similarity_search(query)

    # print all the extracted vectors from the above query
    for doc in docs:
        print("##---- Page ---##")
        print(doc.metadata['source'])
        print("##---- Content ---##")
        print(doc.page_content)

if __name__ == "__main__":
    # The below code  is executed only once and then commented as the Vector Database is now built and ready for your further 
    # experiments
    upload_web()   
    # The below function is experimental to trigger a semantic search on the Vector DB
    faiss_query()