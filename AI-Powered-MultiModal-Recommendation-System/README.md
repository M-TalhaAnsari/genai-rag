# 🍽️ Connoisseur — AI Restaurant Discovery for Pakistan

A production-grade AI backend for restaurant discovery covering **Lahore, Islamabad, Karachi, and Rawalpindi**.

---

## Architecture

```
External Data (OSM + Foursquare + Apify Google Maps)
                    │
                    ▼
             n8n Orchestration
             (scheduled weekly sync)
                    │
                    ▼
           FastAPI Backend (port 8000)
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  PostgreSQL     ChromaDB      BM25
  (Neon)        (vectors)    (keyword)
                    │
                    ▼
           Hybrid Retrieval (RRF)
                    │
                    ▼
         6-Agent Recommendation Workflow
         ┌──────────────────────────────┐
         │ Phase 1 (sequential)         │
         │   Agent 1: Profile Analyser  │
         │   Agent 2: Candidate Filter  │
         │                              │
         │ Phase 2 (parallel)           │
         │   Agent 3: Trend Analyst ──┐ │
         │   Agent 4: Style Expert   ─┤ │
         │   Agent 5: Nutrition      ─┘ │
         │                              │
         │ Phase 3 (sequential)         │
         │   Agent 6: Reranker +        │
         │            Explainer         │
         └──────────────────────────────┘
                    │
                    ▼
          MCP Server (tools for AI clients)
                    │
                    ▼
          Streamlit Frontend (port 8501)
                    │
          User selects restaurant
                    │
                    ▼
          FastAPI → n8n Webhook
          (routes to email / WhatsApp / booking)
```

---

## Project structure

```
connoisseur/
│
├── backend/
│   ├── main.py                  # FastAPI — all endpoints
│   ├── database.py              # Neon PostgreSQL connection
│   ├── models.py                # SQLAlchemy models
│   ├── embedder.py              # Sentence-transformer embeddings
│   ├── vector_store.py          # ChromaDB (2 collections)
│   ├── bm25_store.py            # BM25 keyword index
│   ├── retrieval.py             # RRF hybrid fusion
│   ├── feedback.py              # Thumbs up/down → profile update
│   ├── analytics.py             # Search + feedback analytics
│   ├── contact.py               # Draft message + n8n webhook
│   ├── restaurant_fetcher.py    # OSM + Foursquare (weekly sync)
│   ├── apify_loader.py          # Apify bulk loader (run once)
│   ├── review_summariser.py     # Cautious review summarisation
│   │
│   ├── agents/
│   │   ├── configs.py           # 6 agent definitions + LLM caller
│   │   ├── workflow.py          # Sequential/parallel orchestration
│   │   └── reranker.py          # Standalone reranker utility
│   │
│   └── memory/
│       ├── session.py           # Short-term RAM session memory
│       └── long_term.py         # Persistent PostgreSQL memory
│
├── mcp_service/
│   ├── mcp_server.py            # FastMCP server (5 tools + 1 resource)
│   └── mcp_client.py            # Client + ReAct chat loop
│
├── frontend/
│   └── app.py                   # Streamlit web app
│
├── data/                        # Populated by ingestion (not committed)
│   └── apify_export.json        # Drop your Apify export here
│
├── .env.example
├── requirements.txt
└── run.sh                       # Startup script
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd connoisseur
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
NEON_DATABASE_URL=postgresql+asyncpg://user:password@host/dbname?ssl=require
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
FOURSQUARE_API_KEY=your_foursquare_key        # optional, improves data
N8N_WEBHOOK_URL=https://your-n8n/webhook/xyz  # your n8n webhook URL
```

### 3. Bootstrap data (run once)

**Option A — Apify (recommended, rich data)**
```bash
# 1. Run apify.com/compass/crawler-google-places
#    Search: "restaurants in Lahore Pakistan" × 4 cities
#    Export as JSON → save to data/apify_export.json
# 2. Start the backend
uvicorn backend.main:app --reload
# 3. Load the data
curl -X POST http://localhost:8000/load-apify
# 4. Generate review summaries
curl -X POST http://localhost:8000/summarise-all-reviews
```

**Option B — OSM + Foursquare (free, no manual step)**
```bash
curl -X GET http://localhost:8000/fetch-restaurants > /tmp/data.json
curl -X POST http://localhost:8000/restaurant-sync \
  -H "Content-Type: application/json" \
  -d @/tmp/data.json
```

---

## Running the system

```bash
bash run.sh
```

Choose option 3 to start both backend and frontend.

Or manually:

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
streamlit run frontend/app.py --server.port 8501

# Terminal 3 — MCP server (optional, for AI clients)
python mcp_service/mcp_server.py
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/fetch-restaurants` | Fetch from OSM + Foursquare (n8n calls this) |
| POST | `/restaurant-sync` | Ingest → PostgreSQL + ChromaDB + BM25 |
| POST | `/load-apify` | One-time Apify bulk load |
| GET | `/restaurants` | List with city/cuisine filter + pagination |
| GET | `/restaurants/{id}` | Full detail + reviews |
| GET | `/restaurants/{id}/reviews` | All reviews |
| POST | `/restaurants/{id}/summarise-reviews` | Generate cautious review summary |
| POST | `/summarise-all-reviews` | Batch generate all review summaries |
| GET | `/search?q=...` | Hybrid search (BM25 + ChromaDB + RRF) |
| POST | `/recommend` | Full 6-agent pipeline (SSE streaming) |
| POST | `/feedback` | Thumbs up/down → profile update |
| GET | `/profile/{user_id}` | User preference profile |
| GET | `/memory/{user_id}` | Short + long term memory |
| GET | `/analytics` | Search volume, top queries, feedback stats |
| POST | `/generate-message` | LLM draft contact message |
| POST | `/contact-restaurant` | Send via n8n → email/WhatsApp/booking |
| GET | `/vector-stats` | ChromaDB + BM25 debug |
| GET | `/docs` | Interactive API docs (Swagger) |

---

## n8n workflow

### Sync workflow (weekly schedule)
```
Schedule Trigger (weekly)
    → GET /fetch-restaurants
    → POST /restaurant-sync
```

### Contact workflow (user-triggered)
```
Webhook (POST /contact-restaurant calls this)
    → IF email exists → Send Email node
    → ELSE IF phone exists → WhatsApp node
    → ELSE IF website exists → HTTP Request (booking form)
    → Log result
```

---

## Hallucination handling

Three layers:

1. **Prompt-level**: Every agent prompt says _"use only the data provided — do not invent facts"_
2. **Output validation**: Reranker output is checked — any restaurant name or ID not present in the retrieved candidates is dropped before returning
3. **Review summaries**: Raw reviews are never embedded directly. An LLM writes a 2-sentence cautious summary that only states what multiple reviews agree on, flags fake/polarised reviews, and is shown with a disclaimer

---

## Token usage strategy

| Technique | Where applied | Saving |
|-----------|--------------|--------|
| `json.dumps(x, separators=(',',':'))` | All agent payloads | ~30% vs indent=2 |
| Null field stripping | All agent payloads | ~15% |
| Per-agent field selection | Each agent only gets the fields it needs | ~40% |
| Profile summary capped at 80 words | All agents | Significant |
| Reviews only in reranker, not all 3 parallel agents | Parallel phase | ~60% in Phase 2 |
| Review text capped at 200 chars per review | review_summariser | Prevents overflow |
| Max 8 candidates passed to agents | After filtering | ~35% |
| Parallel Phase 2 | Latency not tokens, but same throughput | 3× faster |

---

## LLM stack

| Use | Primary | Fallback |
|-----|---------|----------|
| Agent calls | Groq (LLaMA 3.3 70B) | Gemini 1.5 Flash |
| Review summaries | Groq (temp=0.1) | Gemini (temp=0.1) |
| Contact messages | Groq (temp=0.7) | Gemini (temp=0.7) |
| Memory summaries | Groq | Gemini |

---

## Data sources

| Source | What it provides | When used |
|--------|-----------------|-----------|
| Apify Google Maps | Name, phone, email, website, rating, reviews, hours, photos, menu | One-time bulk load |
| OpenStreetMap | Name, cuisine, address, phone, website, hours, coordinates | Weekly incremental sync |
| Foursquare | Name, cuisine, rating, price level, hours, coordinates | Weekly incremental sync (gap-filler) |

---

## Context window management

- **Session memory** (`memory/session.py`): RAM, current session only, `deque(maxlen=50)` for messages
- **Long-term memory** (`memory/long_term.py`): PostgreSQL, permanent, last 30 turns passed to LLM
- **Memory summary**: Every 10 turns, LLM rewrites a 100-word summary. Only the summary (not full history) is passed to agents — keeps context short regardless of session length
