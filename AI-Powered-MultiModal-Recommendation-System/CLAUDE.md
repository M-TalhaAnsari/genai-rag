# CLAUDE.md — Project Context for AI Assistants

This file orients any AI coding assistant (Claude Code, Cursor, etc.) working
on this codebase. Read this before making changes.

---

## What this project is

**Connoisseur** — an AI-powered restaurant discovery backend for Pakistan
(Lahore, Islamabad, Karachi, Rawalpindi). Hybrid search + a 6-agent
recommendation pipeline + persistent memory + clickable contact links.

Not a demo. Real data (Apify/OSM/Foursquare), real embeddings, real
multi-agent orchestration, real production concerns (token budgets,
hallucination guardrails, credit-loss resilience).

---

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI (async) |
| Database | PostgreSQL (Neon, cloud) via SQLAlchemy 2.0 + asyncpg |
| Vector store | ChromaDB (local persistent, 3 collections) |
| Keyword search | BM25 (rank-bm25, disk-persisted pickle) |
| Text embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Image embeddings | CLIP `openai/clip-vit-base-patch32` (512-dim) |
| LLM | Groq (LLaMA 3.3 70B) primary → Gemini 1.5 Flash fallback |
| Frontend | Streamlit |
| MCP | FastMCP (server + client) |
| Orchestration | n8n (scheduled sync only — NOT in the user-facing request path) |
| Data sources | Apify (Google Maps scraper), OpenStreetMap, Foursquare, Google Places (optional enrichment) |

---

## Folder structure and where things live

```
backend/
├── main.py                    # THIN — only creates app + mounts routers. Don't add logic here.
├── core/
│   ├── config.py              # ALL env vars read here via `settings` object. Never os.environ directly elsewhere.
│   └── database.py            # engine, AsyncSessionLocal, Base, get_db() dependency
├── models/
│   ├── db_models.py           # SQLAlchemy tables — the single source of truth for schema
│   └── schemas.py             # ALL Pydantic request/response models — one file, easy to find
├── routers/                   # THIN layer — validate input, call services/, return response.
│   │                           No business logic should live in a router.
│   ├── restaurants.py         # reads: list, detail, reviews, review-summary, images
│   ├── contact_links.py       # GET /restaurants/{id}/contact-links
│   ├── search.py              # /search, /search/full, /search/by-review, /search/by-image
│   ├── recommend.py           # POST /recommend — SSE stream wrapping the agent workflow
│   ├── feedback.py            # POST /feedback, GET /profile/{user_id}
│   ├── memory.py              # GET /memory/{user_id}
│   ├── analytics.py           # GET /analytics, /vector-stats
│   └── ingestion.py           # ALL write/sync endpoints, prefixed /ingestion — kept separate
│                                 from read endpoints deliberately (candidate for auth later)
├── services/                  # Business logic. No `from fastapi import ...` here — keep testable.
│   ├── search_service.py      # hybrid_search / full_search / review_search
│   ├── recommend_service.py   # thin async wrapper around agents/workflow.py
│   ├── feedback_service.py    # save_feedback, recompute_profile, apply_profile_boost
│   ├── memory_service.py      # short-term (RAM) + long-term (Postgres) memory
│   ├── analytics_service.py   # search_logs / feedback aggregation queries
│   ├── ingestion_service.py   # shared insert→embed→BM25 pipeline
│   ├── contact_service.py     # mailto: / wa.me link generation — NO LLM, NO webhook
│   └── enrichment_service.py  # Google Places review+photo enrichment, CLIP image embed
├── agents/
│   ├── configs.py             # 6 agent definitions (role/goal/backstory) + call_agent()
│   ├── workflow.py            # THE ORCHESTRATOR — see "Agent workflow" section below
│   └── reranker.py            # standalone reranker (used outside the full workflow)
├── embedder.py                 # text embedding functions (calls data_enrichment.py for rich text)
├── vector_store.py             # ALL ChromaDB ops — 3 collections, see "Vector store" section
├── bm25_store.py                # BM25 build/search, persisted to bm25_index.pkl
├── retrieval.py                 # RRF fusion of BM25 + dense — the core hybrid_search() function
├── data_enrichment.py           # cuisine normalisation + name-based cuisine inference + rich text builder
├── review_summariser.py         # cautious LLM review summarisation with recency weighting
├── restaurant_fetcher.py        # OSM Overpass + Foursquare fetch (free, weekly sync)
├── apify_loader.py              # normalize_apify_place() + _load_places() — shared by manual & automated load
└── apify_automation.py          # resilient Apify run (start→poll→fetch-dataset, survives credit exhaustion)

mcp_service/
├── mcp_server.py               # FastMCP — exposes search/recommend/feedback as MCP tools
└── mcp_client.py                # client + ReAct ChatGroq/Gemini loop

frontend/
└── app.py                       # Streamlit, 4 tabs, calls the FastAPI backend over HTTP
```

### Golden rules for this codebase
1. **Routers never contain business logic.** If a router function is more than
   ~15 lines of actual logic (not counting docstrings), that logic belongs in
   a service.
2. **Services never import FastAPI.** Keeps them testable and reusable from
   scripts, CLI tools, MCP server, etc.
3. **All Pydantic models live in `models/schemas.py`.** Don't define request
   models inline in routers.
4. **All env vars go through `core/config.py`.** Never `os.environ.get()`
   scattered across files.
5. **Ingestion (writes) is namespaced under `/ingestion/*`** — deliberately
   separated from reads so RBAC can be bolted on later without touching read
   endpoints.

---

## Data flow, end to end

```
1. INGESTION (periodic, via n8n or manual)
   Apify (bulk, richest — reviews+photos if maxReviews/maxImages set)
   OSM + Foursquare (weekly, free, no reviews/photos)
        │
        ▼
   apify_loader.normalize_apify_place()  — maps raw Apify JSON to our schema
   COUNTRY FILTER: reject anything where countryCode != "PK"
   DEDUP KEY: external_id (Google placeId) — NOT (name, city), because
              chain branches share names but have different placeIds
        │
        ▼
   PostgreSQL: Restaurant row + Review rows inserted
        │
        ▼
   ChromaDB "restaurants" collection: rich text embedded (see below)
   ChromaDB "restaurant_reviews": cautious LLM summary embedded (if reviews exist)
   ChromaDB "restaurant_images": CLIP embedding per photo (if enrichment run)
        │
        ▼
   BM25 index rebuilt from full PostgreSQL table

2. SEARCH (user-facing, fast path)
   GET /search
        │
        ▼
   retrieval.hybrid_search()
     ├── bm25_store.search()        — keyword match
     ├── vector_store.search_restaurants()  — dense semantic match
     └── RRF fusion: score = Σ 1/(60 + rank) across both lists
        │
        ▼
   (optional) feedback_service.apply_profile_boost()
     — +0.05 if cuisine in user's liked list, -0.05 if in avoided list
        │
        ▼
   analytics_service.log_search() — fire and forget
        │
        ▼
   Response with rrf_score, dense_rank, sparse_rank per result

3. RECOMMENDATION (user-facing, slow path — 10-30s)
   POST /recommend  →  SSE stream  →  see "Agent workflow" below

4. CONTACT (user-facing, no LLM, no webhook)
   GET /restaurants/{id}/contact-links
        │
        ▼
   contact_service.get_contact_links()
     — builds mailto: href (subject+body pre-filled)
     — builds wa.me href (message pre-filled, phone normalised to E.164)
     — passes through website/menu_url as-is
     — only returns fields that have real data
        │
        ▼
   Frontend renders a button per available channel.
   User clicks → their OWN email/WhatsApp app opens → they hit send.
   Nothing is sent server-side. No n8n involved in this path.
```

---

## Agent workflow — the core of the recommendation pipeline

File: `backend/agents/workflow.py`, orchestrated by `run_recommendation_workflow()`.

### Why 6 agents and why this phase structure

The workflow is deliberately **hybrid sequential/parallel**: some steps
depend on prior output (must run in order), others are independent analyses
that can run concurrently to save wall-clock time.

```
run_recommendation_workflow(query, user_id, profile)
│
├─ PHASE 1 — SEQUENTIAL (each step needs the previous one's output)
│  │
│  ├─ Agent 1: profile_analyser
│  │    Input:  user's stored preference profile (slimmed to only
│  │            preferred_cuisines / avoided_cuisines / preferred_cities /
│  │            feedback_count — NOT the raw preference_vector, that's
│  │            for math not for LLM prompts)
│  │    Output: ≤80-word plain-English personality summary
│  │    Why:    every downstream agent reads this summary instead of
│  │            re-parsing the raw profile — saves tokens 5 times over
│  │
│  └─ Agent 2: candidate_retriever
│       Input:  query + profile_summary + top 15 raw hybrid_search() results
│               (slimmed to restaurant_id/name/cuisine/city/rrf_score only)
│       Output: JSON array, 5-8 filtered candidates with a filter_note
│       HALLUCINATION GUARD: after parsing, we intersect the agent's
│               output restaurant_ids against the ACTUAL retrieved set.
│               Anything not in that set is dropped. If JSON parsing
│               fails entirely, we fall back to the raw top-8 candidates
│               — the agent's opinion is advisory, retrieval is ground truth.
│
├─ PHASE 2 — PARALLEL (ThreadPoolExecutor, 3 workers, run concurrently)
│  │  All three read the SAME filtered_candidates from Phase 1.
│  │  None of them can see each other's output — that's fine, they're
│  │  independent analytical lenses, not a debate.
│  │
│  ├─ Agent 3: trend_analyst
│  │    Input:  candidates slimmed to name/cuisine/city only (no scores,
│  │            no profile — trend analysis doesn't need them)
│  │    Output: 3-4 sentences on which candidates are "trendy" in Pakistan
│  │
│  ├─ Agent 4: style_expert
│  │    Input:  candidates (name/cuisine/city) + profile_summary
│  │    Output: 3-4 sentences matching candidates to user's flavour profile
│  │
│  └─ Agent 5: nutrition_expert
│       Input:  candidates (name/cuisine only) + profile_summary
│       Output: 2-3 sentences flagging dietary conflicts / confirming safety
│
├─ PHASE 3 — SEQUENTIAL (final synthesis, needs everything above)
│  │
│  └─ Agent 6: reranker (aka "Recommendation Expert and Explainer")
│       Input:  query + profile_summary + ALL THREE Phase 2 outputs +
│               candidates ENRICHED with review data (this is the ONLY
│               phase that sees review_summary/review_confidence/
│               review_warning — Phase 2 agents never see reviews,
│               deliberate token-saving decision, see below)
│       Output: JSON array, top 5, each with:
│                 restaurant_id, name, cuisine, city, rrf_score,
│                 reasoning (2-3 sentences, must cite specific signals —
│                 query match, profile fit, trend fit — never generic
│                 filler like "this matches your preferences")
│       HALLUCINATION GUARD (two checks, both must pass):
│         1. restaurant_id must be in the original candidate set
│         2. name (lowercased) must match a candidate name
│         Any recommendation failing either check is silently dropped.
│         If the whole JSON parse fails, fall back to the raw filtered
│         candidates with a generic "matched via hybrid search" reasoning
│         — we NEVER show an LLM-invented restaurant to the user.
│
└─ Returns full state dict — routers/recommend.py slices out
   final_recommendations for the SSE stream, but the state also carries
   profile_summary and filtered_candidates count for a debug block.
```

### Why reviews are only shown to Agent 6, not Phase 2

This was a deliberate token-budget decision, not an oversight. Attaching
`review_summary` + `review_confidence` + `review_warning` to every
candidate for all 3 parallel agents would nearly double Phase 2's token
cost for information those agents don't need (trend/style/nutrition
analysis doesn't require review sentiment). Only the final reranker,
which writes the user-facing reasoning, needs that context — so it's the
only phase that pays for it.

### Why every prompt says "use only the data provided"

This line is in every single agent's system prompt (`agents/configs.py`).
Combined with the two hallucination guards above (candidate-set
validation after Phase 1 and Phase 3), this is the project's three-layer
hallucination defence:
1. Prompt instruction (soft constraint)
2. Post-parse validation against real retrieved IDs (hard constraint)
3. Deterministic fallback to raw retrieval data on any parse failure
   (never silently show nothing, never show something invented)

### LLM calls: Groq → Gemini fallback

Every `call_agent()` invocation tries Groq (LLaMA 3.3 70B, temp=0.7) first
and falls back to Gemini 1.5 Flash on ANY exception (rate limit, timeout,
API error). This fallback is inside `agents/configs.py::call_agent()` —
don't reimplement it per-agent.

### Token optimisation applied throughout workflow.py

| Technique | Where |
|---|---|
| `json.dumps(x, separators=(',',':'))` instead of `indent=2` | every payload |
| `_slim(obj, keys)` — strip to only needed fields, drop `None`s | every candidate list sent to an agent |
| Reviews attached only in Phase 3 | see above |
| `profile_summary` capped at 80 words by the Phase-1 prompt | reused by Phases 2 & 3 instead of re-deriving |
| Max 8 candidates survive Phase 1 filtering | caps everything downstream |
| Long-term memory: 100-word LLM summary, not raw history, ever reaches an agent prompt | `services/memory_service.py::recompute_summary()` |

---

## Database schema (PostgreSQL, `models/db_models.py`)

```
Restaurant
  id, name, cuisine, all_cuisines (JSON), city, area, postal_code
  address, phone, email, website, menu_url
  latitude, longitude
  rating, review_count, price_level, reviews_distribution (JSON — full star breakdown)
  description, opening_hours (JSON), photos (JSON list of URLs), tags (JSON)
  source ("apify"|"osm"|"foursquare"), external_id (Google placeId — PRIMARY DEDUP KEY)
  is_embedded (bool)
  created_at, updated_at
  → relationship: reviews (one-to-many, cascade delete)

Review
  id, restaurant_id (FK), reviewer_name, rating, text,
  published_date, source ("google"|"apify"), created_at

UserFeedback
  id, user_id, restaurant_id, restaurant_name, cuisine, city,
  signal (1=like, -1=dislike), query, created_at

UserProfile
  id, user_id (unique), preferred_cuisines (JSON), avoided_cuisines (JSON),
  preferred_cities (JSON), preference_vector (JSON — mean embedding of
  liked restaurants, used for math not for LLM prompts), feedback_count,
  updated_at

SearchLog
  id, user_id, query, result_count, created_at

ConversationHistory
  id, user_id, role ("user"|"assistant"), content, query, created_at

UserMemorySummary
  id, user_id (unique), summary (LLM-generated, ~100 words),
  turn_count (when last recomputed — triggers every 10 turns), updated_at
```

**Dedup key note:** `Restaurant` dedup is by `external_id` (Google placeId)
when available, falling back to `(name, city)` only when `external_id` is
null (OSM records without a place match). This matters because chain
branches (e.g. two branches of the same cafe in different sectors) share a
name but have different placeIds — deduping on `(name, city)` alone would
have wrongly merged or skipped legitimate distinct locations.

---

## ChromaDB — 3 separate collections (`vector_store.py`)

```
"restaurants"          384-dim (MiniLM), cosine
  One vector per restaurant. Rich embedding text built by
  data_enrichment.build_rich_embedding_text() — NOT just "name is a
  X restaurant in Y". Includes cuisine context knowledge (e.g. "Biryani"
  → "rice, spices, meat, aromatic Pakistani biryani"), extracted area
  from address, availability signals (phone/website/hours), tags.
  Metadata includes normalised cuisine (not raw OSM tags), area,
  has_phone, has_website, rating, lat/lon — enables `where=` filtering.

"restaurant_reviews"   384-dim (MiniLM), cosine
  One vector per restaurant (not per review). Embeds the LLM-generated
  cautious summary from review_summariser.py, NOT raw review text.
  Metadata: confidence (none/low/medium/high based on review count),
  avg_rating, weighted_rating (recency-weighted), polarised (bool),
  burst_detected (bool — many reviews in a short window, fake-review
  signal), disclaimer text, dimensions (JSON — 5 quality axes: food_quality,
  cleanliness, service, menu_variety, vibe, each with signal+summary+recency flag).

"restaurant_images"    512-dim (CLIP ViT-B/32), cosine
  ONE VECTOR PER IMAGE, not per restaurant. ID format: "img_{restaurant_id}_{index}".
  A restaurant with 10 photos = 10 separate rows, all sharing restaurant_id
  in metadata. This is deliberate — embedding all photos into one averaged
  vector would lose the ability to match "rooftop seating" to the ONE photo
  that shows a rooftop, as opposed to averaging it away with 9 unrelated shots.
  Text queries embedded via CLIP's text encoder (embed_text_clip()) —
  CLIP's text and image embeddings share the same 512-dim space, so
  natural language queries match photo content directly.
```

**Never mix embedding spaces.** MiniLM (384-dim, text-only) and CLIP
(512-dim, text+image) are different models with incompatible vector
spaces. Querying the wrong collection with the wrong embedder will
silently return garbage (dimension mismatch errors, or worse, no error
if dimensions happen to coincide but semantics don't).

---

## Retrieval — RRF fusion (`retrieval.py`)

```python
score(doc, rank_in_list) = 1 / (60 + rank_in_list)
```

Documents appearing in both the BM25 list and the dense list accumulate
scores from both — no score normalisation needed, which is RRF's whole
advantage over naive weighted-sum fusion (BM25 scores and cosine
similarity scores live on completely different numeric scales).

`k=60` is the standard constant from the original RRF paper — don't
change it without a reason.

`retrieval.hybrid_search()` is the core function nearly everything else
calls (search_service, agents/workflow.py Phase 1's candidate_retriever,
MCP server's search_restaurants tool).

`search/full` (three-signal) additionally merges review sentiment
results with configurable weights `w_identity`/`w_review` (default
0.6/0.4). Restaurants flagged `has_fake_signals=True` (polarised or
review-burst detected) get their review-signal weight automatically
halved — a defensive measure against fake-review manipulation
influencing ranking.

---

## Data ingestion specifics

### apify_loader.py — field mapping gotchas

Real Apify Google Maps output has `city`, `neighborhood`, `postalCode`
as **separate top-level fields**, NOT nested inside `address` (which is
a plain string, not a dict). An earlier bug assumed `address` was a dict
and every restaurant ended up with `city = "Pakistan"` as a result —
fixed, but worth knowing if you see this pattern again in new data
sources.

`categories` (plural, list) holds ALL cuisine tags a place has — a
restaurant can legitimately be tagged Steak House + Chinese + Italian +
Seafood + Sushi simultaneously. We store all of them in `all_cuisines`
(JSON) but pick `categories[0]` as the primary `cuisine` field.

`menu` (not `menuUrl`) is the correct field name for the menu link.

`additionalInfo` is a goldmine for embedding-text richness — it's a dict
of category → list of `{feature: true}` dicts (Atmosphere, Highlights,
Offerings, Dining options, Crowd, Service options). We flatten the
priority categories into a tags list via
`apify_loader._extract_tags_from_additional_info()`.

`reviewsTags` gives structured dish mentions with counts (e.g.
`{"title": "tomahawk steak", "count": 2}`) — used to build a fallback
`description` when Apify's own `description` field is null (which it
almost always is for Google Maps places).

### Country filtering

Text-based location search (`"restaurants in Lahore Pakistan"`) can match
business names containing Pakistani-sounding words anywhere in the world
— e.g. "Karachi Food Company" in Texas, "Karahi Point" in Quebec. Two
layers of defence:
1. Actor input: `"countryCode": "PK"`
2. Hard validation in `apify_loader._load_places()`: any place with
   `countryCode` present and `!= "PK"` is rejected, counted separately
   as `skipped_wrong_country` in the response.

### apify_automation.py — credit-loss resilience

The naive approach (`run-sync-get-dataset-items`) blocks until the ENTIRE
actor run finishes and only returns data at the end — if Apify credits
run out mid-scrape, that call fails and everything already scraped is
lost. Fixed by decoupling the run from the dataset:

```
1. POST /v2/acts/{actor}/runs           — start, don't wait, get run_id + dataset_id
2. Poll GET /v2/actor-runs/{run_id}     — every 10s, up to 10 min
3. GET /v2/datasets/{dataset_id}/items  — fetch REGARDLESS of final run status
```

Step 3 works even if the run's final status is FAILED or ABORTED —
Apify's dataset retains every item already pushed to it independent of
how the run ultimately ended. Response includes `is_partial: true` when
this happens, so the caller knows to re-run later for the rest.

### Reviews/photos are opt-in per Apify call, not automatic

`maxReviews` and `maxImages` must be explicitly set in the actor input
(`apify_automation.py`) or Apify returns empty `reviews: []` and
`imageUrls: []` — this is not an extra API call or extra cost, just a
parameter that was initially missing.

### enrichment_service.py — Google Places as a secondary path

For restaurants already in the DB (from Apify without reviews, or from
OSM/Foursquare which never have reviews), `enrich_restaurants()` uses
the stored `external_id` (placeId) to call Google Places Details
directly — no re-scraping the restaurant itself. For OSM records lacking
a placeId, does a Text Search first to find one.

**Known constraint:** Google Places API requires a billing card on file
even for the $200/month free credit — this is Google Cloud policy, not
a bug in our code. If unavailable, Apify (with maxReviews/maxImages set)
is the only reviews+photos source that requires no card at all.

---

## Memory system (`services/memory_service.py`)

Two layers, deliberately different lifetimes:

**Short-term (RAM)** — plain Python dict of deques, keyed by user_id.
Cleared on server restart. `MAX_MESSAGES=50`. Swap to Redis for
multi-server deployments — the interface (`add_query`, `add_message`,
`get_session_summary`, etc.) would stay identical.

**Long-term (PostgreSQL)** — every conversation turn saved permanently
to `ConversationHistory`. Every 10th turn triggers
`recompute_summary()`, which asks the `profile_analyser` agent to
rewrite a ≤100-word summary from the last 30 turns. This summary — NOT
the raw history — is what gets attached to agent prompts elsewhere,
which is why context size stays constant no matter how long a user's
history grows.

---

## Feedback → personalisation loop (`services/feedback_service.py`)

```
User clicks 👍/👎
    → UserFeedback row inserted
    → recompute_profile() runs immediately:
        - liked/avoided cuisine lists built by frequency (Counter.most_common)
        - a cuisine liked MORE RECENTLY overrides an earlier dislike of
          the same cuisine (net_liked / net_avoided logic)
        - preference_vector = mean of embed_restaurant() for every liked
          restaurant, L2-normalised — used for potential future vector-space
          personalisation, not currently used in ranking directly
    → future /search and /recommend calls with the same user_id
      immediately reflect the update (apply_profile_boost: ±0.05 to
      rrf_score per matching/avoided cuisine)
```

---

## Contact links (`services/contact_service.py`) — NOT n8n-based

Originally designed around n8n webhooks routing to email/WhatsApp/booking.
Replaced with pure client-side link generation after determining the n8n
round-trip added complexity without value for this use case:

```python
email present   → mailto: href, url-encoded subject+body, opens in new tab
phone present   → normalise_phone() to E.164 → wa.me/{number}?text={message}
website present → pass through as-is
menu_url present → pass through as-is
```

`normalise_phone()` handles Pakistani number formats: `+92 311 1100317`,
`0311-1100317`, `03111100317` all normalise to `923111100317` for the
wa.me link. Message text is a template (`build_contact_message()`), not
an LLM call — deliberate, since a template is good enough here and saves
tokens for something this simple. Only channels with real, non-null data
are returned — frontend renders a button only for populated fields.

n8n is NOT part of this path anymore. It's retained only for
`/ingestion/n8n/sync-apify` — scheduled, backend-to-backend, not
user-facing.

---

## MCP server (`mcp_service/mcp_server.py`)

Exposes 5 tools + 1 resource over stdio transport, so any MCP client
(Claude Desktop, this project's own `mcp_client.py`, or other agents)
can call the backend without going through the REST API:

```
search_restaurants     → retrieval.hybrid_search()
get_recommendations    → full 6-agent workflow (10-30s, warn callers)
submit_feedback         → feedback_service.save_feedback()
get_user_profile         → feedback_service.get_profile()
get_analytics             → analytics_service.get_analytics()
resource: restaurant://stats → live PostgreSQL + ChromaDB counts
```

Sync FastMCP tool handlers wrap async service calls via a `_run()` helper
that bridges asyncio inside a sync context (thread pool executor pattern)
— necessary because FastMCP tool functions are synchronous by
convention but everything underneath is async SQLAlchemy.

---

## Environment variables (`core/config.py`)

```
NEON_DATABASE_URL        # required — PostgreSQL connection string
GROQ_API_KEY              # required — primary LLM
GOOGLE_API_KEY             # required — Gemini fallback LLM
APIFY_API_TOKEN            # required for Apify sync — no card needed, $5/mo free
FOURSQUARE_API_KEY         # optional — base place search only, reviews/photos are Premium-only (paid)
GOOGLE_PLACES_API_KEY      # optional — requires a billing card, $200/mo free credit
N8N_WEBHOOK_URL             # legacy, not currently used by any active endpoint
```

---

## Common commands

```bash
# Run everything
bash run.sh                                    # option 3 = backend + frontend

# Manual
uvicorn backend.main:app --reload
streamlit run frontend/app.py
python mcp_service/mcp_server.py

# Ingest data
curl -X POST "http://localhost:8000/ingestion/n8n/sync-apify?per_city_limit=50"
curl -X POST http://localhost:8000/ingestion/summarise-all-reviews
curl -X POST "http://localhost:8000/ingestion/enrich-reviews?limit=10&embed_images=true"

# Check state
curl http://localhost:8000/vector-stats
curl http://localhost:8000/ingestion/enrich-status
curl http://localhost:8000/ingestion/n8n/apify-status
```

---

## Things NOT to do

- Don't add business logic to routers — put it in services/.
- Don't call `os.environ` directly — go through `core/config.py`.
- Don't mix MiniLM and CLIP embeddings — they're different vector spaces.
- Don't dedupe restaurants by `(name, city)` when `external_id` is
  available — chain branches will collide.
- Don't attach review data to Phase 2 agents in the workflow — it's a
  deliberate token-saving exclusion, only Phase 3 (reranker) needs it.
- Don't remove the hallucination guards in `workflow.py` (candidate-set
  validation after Phase 1 and Phase 3) even if they seem redundant with
  the prompt instructions — they're the hard constraint, prompts are
  only the soft one.
- Don't re-introduce `run-sync-get-dataset-items` for Apify — it loses
  data on credit exhaustion. Always use the start→poll→fetch-dataset
  pattern in `apify_automation.py`.
- Don't route contact messages through n8n or an LLM — `contact_service.py`
  is intentionally template-based and client-side only.
