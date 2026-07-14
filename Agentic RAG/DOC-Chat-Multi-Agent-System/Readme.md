# DocChat 🐥: Multi-Agent Agentic RAG System

A resilient, production-grade Retrieval-Augmented Generation (RAG) workspace utilizing a stateful multi-agent architecture orchestrated via **LangGraph**. DocChat leverages **Docling** for layout-aware document processing, **ChromaDB** and **BM25** for hybrid retrieval, and **Gemini 2.5 Flash** to drive a deterministic pipeline featuring contextual answering and automated factual verification.

---

## 🏗️ Core Architecture & Agentic Workflow

DocChat routes, generates, and self-corrects using a deterministic state machine powered by **LangGraph**:

1. **Ingestion Layer (`DocumentProcessor`)**: Translates messy documents (PDFs, DOCX, TXT, MD) into highly structured Markdown using layout-aware parsing (**Docling**). Employs content-hash caching (`SHA-256`) to avoid reprocessing duplicate documents.
2. **Retrieval Layer (`RetrieverBuilder`)**: Executes a weighted hybrid search balancing dense semantic embeddings (**ChromaDB**) with lexical phrase matching (**BM25**).
3. **Relevance Gate (`RelevanceChecker`)**: Triages the retrieved passages, flagging them as `CAN_ANSWER`, `PARTIAL`, or `NO_MATCH`. If no conceptual match exists, the system terminates early to eliminate hallucination.
4. **Generation Engine (`ResearchAgent`)**: Generates an exhaustive draft answer strictly grounded within the context boundaries.
5. **Verification Layer (`VerificationAgent`)**: Uses strict Pydantic parsing (`with_structured_output`) to isolate unsupported assertions or hard contradictions, assuring absolute factual integrity.

---

## 📁 Repository Structure

```text
DOC-Chat-Multi-Agent-System/
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
    |-- relevance_checker.py     # Context relevance triaging agent
    |-- research_agent.py        # Contextual grounding generation agent
    |-- verification_agent.py    # Pydantic structural alignment & verification agent
    +-- workflow.py              # Stateful LangGraph orchestration engine
```

---

## 🛠️ Installation & Setup

```text
DOC-Chat-Multi-Agent-System/
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
    |-- relevance_checker.py     # Context relevance triaging agent
    |-- research_agent.py        # Contextual grounding generation agent
    |-- verification_agent.py    # Pydantic structural alignment & verification agent
    +-- workflow.py              # Stateful LangGraph orchestration engine
```

---

## 🛠️ Installation & Setup

Follow these steps sequentially to configure your local environment.

### 1. System Requirements
Before proceeding, ensure you have the following installed on your host machine:
* **Python 3.10 to 3.12** (Recommended stability range)
* **Git** command line interface tool

### 2. Clone & Navigate to the Project
Open your system terminal or command prompt, clone the remote repository, and shift into the project root directory:
```bash
Follow these steps sequentially to configure your local environment.

### 1. System Requirements
Before proceeding, ensure you have the following installed on your host machine:
* **Python 3.10 to 3.12** (Recommended stability range)
* **Git** command line interface tool

### 2. Clone & Navigate to the Project
Open your system terminal or command prompt, clone the remote repository, and shift into the project root directory:
```bash
git clone https://github.com/your-username/DOC-Chat-Multi-Agent-System.git
cd DOC-Chat-Multi-Agent-System
```

### 3. Create a Virtual Environment (Highly Recommended)
Isolate your application dependencies to avoid local system package conflicts:

* **On Windows (CMD/PowerShell):**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Project Dependencies
Install all required libraries, core frameworks, embedding layers, and multi-agent workflow runtime tools:
```bash
pip install --upgrade pip
```

### 3. Create a Virtual Environment (Highly Recommended)
Isolate your application dependencies to avoid local system package conflicts:

* **On Windows (CMD/PowerShell):**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Project Dependencies
Install all required libraries, core frameworks, embedding layers, and multi-agent workflow runtime tools:
```bash
pip install --upgrade pip
pip install streamlit langchain langchain-community langchain-core langchain-google-genai pydantic pydantic-settings chromadb python-dotenv sentence-transformers rank-bm25 docling langgraph
```

### 5. Configure Your Environment Variables
Create a new file named exactly `.env` in the root folder directory of the workspace. 

DocChat relies on a modern Pydantic v2 validation layer configured to parse strict runtime dependencies while securely filtering out extra development tracking tokens (such as OpenAI, Anthropic, Grok, or LangChain Tracing metrics). Add your Google Gemini API token inside:

```env
GOOGLE_API_KEY=AIzaSyYourActualGeminiKeyHere
```

---

## 🚀 Running the Application

Once installation finishes and your `.env` properties match your deployment tokens, boot the interface instance:

```bash
```

### 5. Configure Your Environment Variables
Create a new file named exactly `.env` in the root folder directory of the workspace. 

DocChat relies on a modern Pydantic v2 validation layer configured to parse strict runtime dependencies while securely filtering out extra development tracking tokens (such as OpenAI, Anthropic, Grok, or LangChain Tracing metrics). Add your Google Gemini API token inside:

```env
GOOGLE_API_KEY=AIzaSyYourActualGeminiKeyHere
```

---

## 🚀 Running the Application

Once installation finishes and your `.env` properties match your deployment tokens, boot the interface instance:

```bash
streamlit run app.py
```

Streamlit will dynamically initialize your local network host engine. A browser window will automatically launch tracking the runtime dashboard configuration:
👉 **http://localhost:8501**

---

## 🔍 Deep-Dive: Enterprise Feature Highlights

### 🛡️ Resilient Environments (`config/settings.py`)
Utilizes Pydantic v2's `SettingsConfigDict` equipped with `extra="ignore"`. You can keep your system stocked with other API keys (OpenAI, Anthropic, Grok, LangChain Tracing) without ever triggering structural initialization schema validation crashes.
```

Streamlit will dynamically initialize your local network host engine. A browser window will automatically launch tracking the runtime dashboard configuration:
👉 **http://localhost:8501**

---

## 🔍 Deep-Dive: Enterprise Feature Highlights

### 🛡️ Resilient Environments (`config/settings.py`)
Utilizes Pydantic v2's `SettingsConfigDict` equipped with `extra="ignore"`. You can keep your system stocked with other API keys (OpenAI, Anthropic, Grok, LangChain Tracing) without ever triggering structural initialization schema validation crashes.

### 📊 Advanced Parsing & Hybrid Cache (`document_processor/file_handler.py`)
Replaces archaic regex chunkers with layout-aware document structure maps via Docling. If a document has already been processed, the system reads its structure instantly from a secure `.pkl` bin, reducing token overhead and file-processing bottlenecks. It also handles both Streamlit file buffers and local string paths transparently.
### 📊 Advanced Parsing & Hybrid Cache (`document_processor/file_handler.py`)
Replaces archaic regex chunkers with layout-aware document structure maps via Docling. If a document has already been processed, the system reads its structure instantly from a secure `.pkl` bin, reducing token overhead and file-processing bottlenecks. It also handles both Streamlit file buffers and local string paths transparently.

### 🛑 Automated Anti-Hallucination Guardrails (`agents/verification_agent.py`)
### 🛑 Automated Anti-Hallucination Guardrails (`agents/verification_agent.py`)
Forces Gemini to map its own outputs to a strict Pydantic definition layer before rendering the final result:

```python
```python
class VerificationSchema(BaseModel):
    supported: str                  # "YES" or "NO"
    unsupported_claims: List[str]   # Isolates rogue data statements
    contradictions: List[str]       # Captures mismatches against context data
    relevant: str                   # Target evaluation tag
    additional_details: str         # Deep-dive structural reasoning
```