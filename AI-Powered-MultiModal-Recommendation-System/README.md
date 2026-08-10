# 🍽️ Connoisseur — AI Restaurant Discovery for Pakistan

Production-grade AI backend for restaurant discovery across **Lahore, Islamabad, Karachi, and Rawalpindi**. Hybrid search, a 6-agent recommendation pipeline, persistent memory, and clickable contact links — no n8n dependency for the user-facing flow.

---

## Architecture

```
Data ingestion                    Storage                      Retrieval
──────────────                    ───────                      ─────────
Apify (bulk, one-time)     →      PostgreSQL (Neon)      →     BM25 (keyword)
OSM + Foursquare (weekly)  →      ChromaDB × 3:           →     Dense (identity)
Google Places (enrichment) →        restaurants                Review sentiment
                                     restaurant_reviews          Image (CLIP)
                                     restaurant_images      →   RRF fusion
                                                                      │
                                                                      ▼
                                                          6-agent recommendation
                                                          workflow (Groq/Gemini)
                                                                      │
                                                                      ▼
                                                          FastAPI (routers + services)
                                                                      │
                                                                      ▼
                                                          Streamlit frontend
                                                          (clickable email/WhatsApp/
                                                           website links per card)
```

**Design note:** contact is handled entirely in-app — clicking a restaurant's email/WhatsApp/website button opens the user's own client with a pre-filled message. No n8n webhook, no server-side sending. n8n is used only for scheduled data sync (`/ingestion/n8n/sync-apify`).

---

## Project structure

```
connoisseur/
├── backend/
│   ├── main.py                    # Thin entry point — mounts routers only
│   │
│   ├── core/
│   │   ├── config.py              # All env vars, one place
│   │   └── database.py            # PostgreSQL async engine + get_db dependency
│   │
│   ├── models/
│   │   ├── db_models.py           # SQLAlchemy tables
│   │   └── schemas.py             # All Pydantic request/response models
│   │
│   ├── routers/                   # Thin — validation + calls into services/
│   │   ├── restaurants.py         # GET /restaurants, /{id}, /{id}/reviews, /{id}/images
│   │   ├── contact_links.py       # GET /restaurants/{id}/contact-links
│   │   ├── search.py              # /search, /search/full, /search/by-review, /search/by-image
│   │   ├── recommend.py           # POST /recommend (SSE streaming)
│   │   ├── feedback.py            # POST /feedback, GET /profile/{user_id}
│   │   ├── memory.py              # GET /memory/{user_id}
│   │   ├── analytics.py           # GET /analytics, /vector-stats
│   │   └── ingestion.py           # All write/sync endpoints, prefixed /ingestion
│   │
│   ├── services/                  # Business logic — no FastAPI imports here
│   │   ├── search_service.py
│   │   ├── recommend_service.py
│   │   ├── feedback_service.py
│   │   ├── memory_service.py
│   │   ├── analytics_service.py
│   │   ├── ingestion_service.py
│   │   ├── contact_service.py     # mailto: / wa.me link generation
│   │   └── enrichment_service.py  # Google Places reviews + CLIP image embedding
│   │
│   ├── agents/
│   │   ├── configs.py             # 6 agent definitions + Groq/Gemini caller
│   │   ├── workflow.py            # Sequential + parallel orchestration
│   │   └── reranker.py            # Standalone reranker utility
│   │
│   ├── memory/                    # (legacy flat modules, superseded by services/memory_service.py)
│   │
│   ├── embedder.py                # Sentence-transformer text embeddings
│   ├── vector_store.py            # ChromaDB — 3 collections (restaurants, reviews, images)
│   ├── bm25_store.py              # BM25 keyword index
│   ├── retrieval.py               # RRF fusion (BM25 + dense)
│   ├── data_enrichment.py         # Cuisine normalisation, name-based inference
│   ├── review_summariser.py       # Cautious LLM review summarisation
│   ├── restaurant_fetcher.py      # OSM + Foursquare fetch
│   ├── apify_loader.py            # Apify JSON → PostgreSQL + ChromaDB + BM25
│   └── apify_automation.py        # Resilient Apify run (survives credit exhaustion)
│
├── mcp_service/
│   ├── mcp_server.py               # FastMCP — 5 tools + 1 resource
│   └── mcp_client.py               # Client + ReAct chat loop
│
├── frontend/
│   └── app.py                      # Streamlit — 4 tabs, clickable contact links
│
├── data/                           # Populated by ingestion (not committed)
│   └── apify_export.json           # Manual Apify export goes here
│
├── requirements.txt
├── .env.example
└── run.sh
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```env
NEON_DATABASE_URL=postgresql+asyncpg://user:pass@host/db?ssl=require
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
APIFY_API_TOKEN=your_apify_token          # apify.com — no card needed
FOURSQUARE_API_KEY=your_foursquare_key    # optional, base search only
GOOGLE_PLACES_API_KEY=your_places_key     # optional — requires card, see note below
```

---

## Bootstrapping data (first time)

### Option A — Apify (recommended, richest data)

```bash
uvicorn backend.main:app --reload

# Automated (resilient — survives mid-run credit exhaustion)
curl -X POST "http://localhost:8000/ingestion/n8n/sync-apify?per_city_limit=50"

# Batch review summaries into ChromaDB
curl -X POST http://localhost:8000/ingestion/summarise-all-reviews
```

The automated sync uses a start-run → poll → fetch-dataset pattern. Even if Apify credits run out mid-scrape, everything scraped before that point is recovered and saved — nothing is lost.

### Option B — OSM + Foursquare (free, no signup friction)

```bash
curl http://localhost:8000/ingestion/fetch-restaurants > /tmp/data.json
curl -X POST http://localhost:8000/ingestion/restaurant-sync \
  -H "Content-Type: application/json" -d @/tmp/data.json
```

Weaker data (no reviews, sparse cuisine tags) — good for filling city coverage between Apify runs.

---

## Enriching with reviews + photos

Reviews and photos aren't fetched by default in every Apify run — they cost extra Apify compute units. Two ways to get them:

**1. Bake them into the Apify sync itself** (`apify_automation.py` actor input already includes `maxReviews: 5` and `maxImages: 10` — no extra cost, same API call).

**2. Enrich existing restaurants after the fact** via Google Places API:

```bash
# Test with 10 first
curl -X POST "http://localhost:8000/ingestion/enrich-reviews?limit=10&embed_images=true"

curl http://localhost:8000/ingestion/enrich-status

# Run on everything if the test looks good
curl -X POST "http://localhost:8000/ingestion/enrich-reviews?embed_images=true"
```

> **Google Places requires a billing card on file**, even for the $200/month free credit — this is Google Cloud policy, not a bug. If your card is rejected, a Wise or Payoneer virtual USD card sometimes passes verification where a local card doesn't. If you can't add billing at all, skip this step — search and recommendations work fine without reviews; just re-run an Apify sync next month with `maxReviews`/`maxImages` set instead.

Each review is summarised cautiously by an LLM (flags mixed/fake-looking review patterns, never invents details) and embedded into `restaurant_reviews`. Each photo is embedded via CLIP into `restaurant_images`, tied to its restaurant by `restaurant_id`.

---

## Running the system

```bash
bash run.sh   # option 3: backend + frontend together
```

Or manually:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run frontend/app.py --server.port 8501
python mcp_service/mcp_server.py   # optional, for MCP clients
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/restaurants` | List, filterable by city/cuisine, paginated |
| GET | `/restaurants/{id}` | Full detail + reviews |
| GET | `/restaurants/{id}/reviews` | Raw reviews |
| GET | `/restaurants/{id}/review-summary` | Cautious LLM quality summary |
| GET | `/restaurants/{id}/images` | Embedded photo URLs |
| GET | `/restaurants/{id}/contact-links` | Clickable email/WhatsApp/website/menu links |
| GET | `/search` | Hybrid BM25 + dense + RRF |
| GET | `/search/full` | Three-signal (+ review sentiment) |
| GET | `/search/by-review` | Review sentiment only |
| GET | `/search/by-image` | CLIP text-to-image search |
| POST | `/recommend` | 6-agent pipeline, SSE streaming |
| POST | `/feedback` | Thumbs up/down → profile update |
| GET | `/profile/{user_id}` | Preference profile |
| GET | `/memory/{user_id}` | Session + long-term memory |
| GET | `/analytics` | Search volume, top queries, feedback stats |
| POST | `/ingestion/restaurant-sync` | Ingest any restaurant list |
| GET | `/ingestion/fetch-restaurants` | OSM + Foursquare fetch |
| POST | `/ingestion/load-apify` | Manual Apify JSON file load |
| POST | `/ingestion/n8n/sync-apify` | Automated, credit-loss-resilient Apify sync |
| POST | `/ingestion/enrich-reviews` | Google Places review + image enrichment |
| GET | `/ingestion/enrich-status` | Enrichment progress |
| POST | `/ingestion/summarise-all-reviews` | Batch review summarisation |
| GET | `/docs` | Swagger UI |

---

## Contact links — how they work

No webhooks, no LLM call, no n8n involvement. Pure link generation:

```
email present   → mailto: href, subject + body pre-filled, opens user's email client
phone present   → wa.me link, message pre-filled, opens WhatsApp
website present → plain URL, opens in new tab
menu_url present → plain URL, opens in new tab
```

Only channels with real data are shown — a restaurant with no email never gets an email button.

---

## Data sources — comparison

| Source | Reviews | Photos | Card required | Cost |
|---|---|---|---|---|
| Apify Google Maps | ✅ (with `maxReviews`) | ✅ (with `maxImages`) | ❌ No | $5/mo free, resets monthly |
| OpenStreetMap | ❌ | ❌ | ❌ No | Free, unlimited |
| Foursquare | ❌ (Premium-only, no free tier) | ❌ (Premium-only) | For base search: no | Base search free, reviews/photos always paid |
| Google Places | ✅ | ✅ | ✅ **Yes, mandatory** | $200/mo free credit |
| TripAdvisor (official) | ✅ | ✅ | ✅ **Yes, mandatory** | 5000 calls/mo free |

**Recommendation:** Apify is the only source that gives reviews + photos without a card. Bake `maxReviews`/`maxImages` into every sync rather than relying on Google Places enrichment as a second step.

---

## Hallucination handling

1. **Prompt level** — every agent prompt: "use only the data provided, do not invent facts"
2. **Output validation** — reranker output checked against actual candidate `restaurant_id`s and names; anything not in the retrieved set is dropped
3. **Review summaries** — LLM only states what multiple reviews agree on; disagreements and polarised/fake-looking rating patterns are flagged explicitly, never silently resolved

---

## Token usage strategy

| Technique | Saving |
|---|---|
| Compact JSON (`separators=(',',':')`) in all agent payloads | ~30% |
| Per-agent field slimming — each agent gets only what it needs | ~40% |
| Reviews passed only to the reranker, not the 3 parallel Phase 2 agents | ~60% in Phase 2 |
| Profile summary capped at 80 words | fixed overhead |
| Max 8 candidates after filtering (was 20) | ~35% |
| Long-term memory: 100-word LLM summary, not full history, sent to agents | constant regardless of conversation length |

---

## Country filtering (Apify)

Text-based location search can match business names containing Pakistani-sounding words anywhere in the world (e.g. "Karachi Food Company" in Texas). Two-layer fix:
1. Actor input: `"countryCode": "PK"`
2. Hard post-fetch validation — any place with `countryCode != "PK"` is rejected regardless of what the actor returns, counted as `skipped_wrong_country` in the sync response

---

## Credit-loss resilience (Apify)

Old approach (`run-sync-get-dataset-items`) blocked until the entire run finished and lost everything if credits ran out mid-scrape. Current approach:
```
Start run (non-blocking) → poll status every 10s → fetch dataset
regardless of final status (SUCCEEDED/FAILED/ABORTED/TIMED-OUT)
```
Apify's dataset retains every item scraped before the run stopped, independent of how the run ended. Partial runs are reported via `is_partial: true` in the response — nothing is silently lost.
