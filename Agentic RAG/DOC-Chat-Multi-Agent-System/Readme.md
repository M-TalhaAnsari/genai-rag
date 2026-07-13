# DocChat 🐥: Multi-Agent Agentic RAG System
A resilient, production-grade Retrieval-Augmented Generation (RAG) workspace utilizing a stateful multi-agent architecture orchestrated via LangGraph. DocChat leverages Docling for layout-aware document processing, ChromaDB and BM25 for hybrid retrieval, and Gemini 2.5 Flash to drive a deterministic pipeline featuring contextual answering and automated factual verification.

## 🏗️ Core Architecture & Agentic Workflow
DocChat doesn't just guess an answer—it routes, generates, and self-corrects using a deterministic state machine powered by LangGraph:

Code snippet
graph TD
    A[User Input / Query] --> B[RelevanceChecker]
    B -->|NO_MATCH| C[Refuse & Exit]
    B -->|CAN_ANSWER / PARTIAL| D[ResearchAgent]
    D --> E[VerificationAgent]
    E -->|Contradictions / Unsupported| F[Self-Correction Cycle / Refinement]
    E -->|Verified Output| G[Streamlit Interface UI]

## ScreenShot
<img width="1915" height="867" alt="image" src="https://github.com/user-attachments/assets/45a113e8-1055-4da1-97ff-822d4512f0b7" />


## Ingestion Layer (DocumentProcessor): 
Translates messy documents (PDFs, DOCX, TXT, MD) into highly structured Markdown using layout-aware parsing (Docling). Employs content-hash caching (SHA-256) to avoid reprocessing duplicate documents.

## Retrieval Layer (RetrieverBuilder): 
Executes a weighted hybrid search balancing dense semantic embeddings (ChromaDB) with lexical phrase matching (BM25).

## Relevance Gate (RelevanceChecker): 
Triages the retrieved passages, flagging them as CAN_ANSWER, PARTIAL, or NO_MATCH. If no conceptual match exists, the system terminates early to eliminate hallucination.

## Generation Engine (ResearchAgent):
 Generates an exhaustive draft answer strictly grounded within the context boundaries.

## Verification Layer (VerificationAgent): 
Uses strict Pydantic parsing (with_structured_output) to isolate unsupported assertions or hard contradictions, assuring absolute factual integrity.

## 📁 Repository Structure
Plaintext
DOC-Chat-Multi_Agent-System/
|-- .env                         # Environment configurations & API tokens
|-- app.py                       # Main Streamlit Graphical UI Workspace
|-- requirements.txt             # Project dependencies
|-- config/
|   |-- __init__.py
|   |-- constants.py              # Operational limits (file sizes, allowed formats)
|   +-- settings.py              # Pydantic v2 application configuration setup
|-- document_processor/
|   |-- __init__.py
|   +-- file_handler.py          # Docling parser engine & SHA-256 cache controller
|-- retriever/
|   |-- __init__.py
|   +-- builder.py               # ChromaDB + BM25 Hybrid Retrieval constructor
+-- agents/
    |-- __init__.py
    |-- relevance_checker.py             # Context relevance triaging agent
    |-- research_agent.py              # Contextual grounding generation agent
    |-- verification_agent.py          # Pydantic structural alignment & verification agent
    +-- workflow.py              # Stateful LangGraph orchestration engine
### 🛠️ Installation & Setup

1. Clone & Navigate to the Project
Bash
git clone https://github.com/your-username/DOC-Chat-Multi-Agent-System.git
cd DOC-Chat-Multi-Agent-System

2. Configure Your Environment
Create a .env file in the root directory. DocChat relies on Pydantic v2 settings to safely extract properties while completely ignoring extraneous environmental configuration objects:

Code snippet
GOOGLE_API_KEY=AIzaSyYourActualGeminiKeyHere

3. Install Dependencies
Install all required libraries, including layout conversion tools, text splitters, and graph architectures:

Bash
pip install streamlit langchain langchain-community langchain-core langchain-google-genai pydantic pydantic-settings chromadb python-dotenv sentence-transformers rank-bm25 docling langgraph
🚀 Running the Application
Launch the local execution server using Streamlit from the root folder directory:

Bash
streamlit run app.py
Your system will spin up a local interface server, automatically launching a browser window at:
👉 http://localhost:8501

### 🔍 Deep-Dive: Enterprise Feature Highlights
🛡️ Resilient Environments (config/settings.py)
Utilizes Pydantic v2's SettingsConfigDict equipped with extra="ignore". You can keep your system stocked with other API keys (OpenAI, Anthropic, Grok, LangChain Tracing) without ever triggering structural initialization schema validation crashes.

### 📊 Advanced Parsing & Hybrid Cache (document_processor/file_handler.py)
Replaces archaic regex chunkers with layout-aware document structure maps. If a document has already been processed, the system reads its structure instantly from a secure .pkl bin, reducing token overhead and file-processing bottlenecks. It also handles both Streamlit file buffers and local string paths transparently.

### 🛑 Automated Anti-Hallucination Guardrails (agents/verification.py)
Forces Gemini to map its own outputs to a strict Pydantic definition layer before rendering the final result:

Python code
class VerificationSchema(BaseModel):
    supported: str # "YES" or "NO"
    unsupported_claims: List[str] # Isolates rogue data statements
    contradictions: List[str] # Captures mismatches against context data
    relevant: str # Target evaluation tag
    additional_details: str # Deep-dive structural reasoning
