# AI-Powered Multimodal Food Recommendation System

A multi-agent RAG system that recommends restaurants and recipes using
multimodal vector retrieval (text + image embeddings), a parallel agent
workflow, and an MCP-based tool server.

---

## Project structure

```
food_recommender/
│
├── data/                          # Raw and processed data (not committed)
│   ├── California-Culinary-Map.txt
│   ├── food_recipes.json
│   ├── synthetic_user_reviews.json
│   ├── structured_restaurant_data.json   ← generated
│   ├── augmented_food_recipe.json         ← generated
│   └── augmented_user_review.json         ← generated
│
├── data_pipeline/
│   ├── ingest_restaurants.py      # Parse raw text → structured JSON (Groq/Gemini)
│   ├── process_multimodal.py      # Caption images in recipes & reviews (Gemini Vision)
│   ├── build_vector_index.py      # Build ChromaDB collections (text + CLIP)
│   └── manage_database.py         # Interactive CLI to add/edit/delete records
│
├── recommendation_engine/
│   ├── agents.py                  # Six specialist agent configs + prompt builder
│   ├── workflow.py                # Hybrid sequential/parallel workflow (Groq/Gemini)
│   └── vector_retrieval.py        # Text & image retrieval + multimodal fusion
│
├── mcp_service/
│   ├── mcp_server.py              # FastMCP server with 3 tools + 1 resource
│   └── mcp_client.py              # MCP client with sampling callback (Groq/Gemini)
│
├── ui/
│   ├── restaurant_chat.py         # MCP-powered restaurant discovery chatbot
│   └── recommendation_chat.py     # Full agent-workflow recommendation chatbot
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

---

## Run order (first time)

```bash
# 1. Structure the raw restaurant text
python data_pipeline/ingest_restaurants.py

# 2. Caption images in recipe and review data
python data_pipeline/process_multimodal.py

# 3. Build the vector index
python data_pipeline/build_vector_index.py
```

---

## Launch the UIs

```bash
# Restaurant discovery chatbot (MCP-powered)
python ui/restaurant_chat.py

# Full multi-agent recommendation chatbot
python ui/recommendation_chat.py
```

---

## CLI database manager

```bash
python data_pipeline/manage_database.py
```

---

## Architecture

```
User query
    │
    ▼
Intent classifier (Groq/Gemini)
    │
    ├─► "restaurant" / "recipe" / "both"
    │       │
    │       ▼
    │   Preference extractor
    │       │
    │       ▼
    │   Agent Workflow
    │   ┌─────────────────────────────────────┐
    │   │ Phase 1: User Profile Generator      │  sequential
    │   │ Phase 2: RAG Retriever               │  sequential
    │   │ Phase 3: ┌─ Trend Analyst ─┐         │  parallel
    │   │          ├─ Style Expert  ─┤         │
    │   │          └─ Nutrition Exp ─┘         │
    │   │ Phase 4: Recommendation Expert       │  sequential
    │   └─────────────────────────────────────┘
    │       │
    │       ▼
    │   Formatted response
    │
    └─► "database" → UI tabs (add/edit records)
```

---

## LLM stack

| Layer | Primary | Fallback |
|-------|---------|----------|
| Text generation | Groq (LLaMA 3.3 70B) | Gemini 3.5 Flash |
| Image captioning | Gemini 1.5 Flash (vision) | Groq text-only |
| Structured output | Gemini (`.with_structured_output`) | — |
