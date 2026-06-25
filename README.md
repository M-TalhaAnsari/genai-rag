# Generative AI & RAG Projects using LangChain

A collection of Generative AI, Retrieval-Augmented Generation (RAG), and LLM-powered applications built using LangChain, Hugging Face Embeddings, Ollama, and Python.

This repository contains multiple hands-on projects, experiments, and implementations developed while learning modern AI application development. In addition to the course implementations, several components were extended using local LLMs and open-source embedding models.

---

## Technologies Used

- Python
- LangChain
- Ollama
- Hugging Face Embeddings
- FAISS Vector Database
- ChromaDB
- LangSmith
- OpenAI APIs
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Document Loaders
- Text Splitters
- Conversational Memory

---

## Projects

### 1. Conversational Chatbot
A chatbot capable of maintaining conversation context using LangChain memory components.

### 2. CSV & Excel Bot
Interact with CSV and Excel files using natural language queries.

### 3. CV Summarization System
Extracts and summarizes candidate resumes using Large Language Models.

### 4. Invoice Data Extraction
Automatically extracts structured information from invoice documents.

### 5. PDF RAG Question Answering
Upload PDF documents and ask questions based on their contents using Retrieval-Augmented Generation.

### 6. RAG Vector Search
Implemented a vector-based retrieval system capable of:

- Loading documents
- Generating embeddings
- Storing vectors
- Retrieving relevant chunks
- Answering user questions

This project demonstrates the complete RAG pipeline.

### 7. SQL RAG
Natural language querying over structured databases.

---

## Additional Implementations

Beyond the course content, the following enhancements were implemented:

### Hugging Face Embeddings

Used open-source embedding models for semantic document retrieval.

Examples:

- sentence-transformers
- all-MiniLM-L6-v2

### Ollama Integration

Integrated local LLMs through Ollama for private and cost-effective inference.

Examples:

- Llama models
- Mistral models
- Gemma models

This enables running AI applications locally without relying on cloud APIs.

---

## Learning Outcomes

Through these implementations I gained practical experience in:

- Building RAG applications
- Vector databases
- Embedding models
- Prompt engineering
- LangChain chains
- Memory systems
- Local LLM deployment
- Evaluation and tracing with LangSmith
- Document ingestion pipelines

---

## Repository Structure

```text
projects/
    conversational-chatbot/
    csv-excel-bot/
    cv-summary/
    invoice-data-extraction/
    rag-vector-search/

experiments/
    prompt-templates/
    output-parsers/
    conversational-memory/
    document-chains/
    loaders-and-splitters/
    langsmith-tracing/