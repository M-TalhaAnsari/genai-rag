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
│   ├── config.py               # ALL env vars read here via `settings` object. Never os.environ directly elsewhere.
│   ├── database.py              # engine, AsyncSessionLocal, Base, get_db() dependency
│   ├── security.py              # JWT encode/decode, password hashing, get_current_user, require_admin/require_user
│   └── redis_client.py          # one shared async Redis pool — sessions, rate limits, every short-lived token
├── models/
│   ├── db_models.py             # SQLAlchemy tables — the single source of truth for schema
│   └── schemas.py                # ALL Pydantic request/response models — one file, easy to find
├── routers/                     # THIN layer — validate input, call services/, return response.
│   │                              No business logic should live in a router.
│   ├── restaurants.py            # reads: list, detail, reviews, review-summary, images
│   ├── contact_links.py          # GET /restaurants/{id}/contact-links
│   ├── search.py                 # /search, /search/full, /search/by-review, /search/by-image
│   ├── recommend.py              # POST /recommend — SSE stream wrapping the agent workflow
│   ├── feedback.py               # POST /feedback, GET /profile/{user_id}
│   ├── memory.py                 # GET /memory/{user_id}
│   ├── analytics.py              # GET /analytics, /vector-stats
│   ├── ingestion.py              # ALL write/sync endpoints, prefixed /ingestion — kept separate
│   │                                from read endpoints deliberately (this IS the auth boundary now:
│   │                                every /ingestion/* route requires require_admin)
│   └── auth/                     # split once auth crossed ~10 endpoints — see
│       ├── __init__.py            #   backend/services/auth/CLAUDE.md for the full design
│       ├── local.py               # register/login/verify-email/refresh/logout/me/deactivate/
│       │                            sessions/2FA/password-reset — everything except Google
│       ├── google.py              # Google OAuth (Authorization Code flow) + account linking
│       └── auth.md                # session-by-session build notes — history, not current-state truth;
│                                     read CLAUDE.md files for current state
├── services/                     # Business logic. No `from fastapi import ...` here — keep testable.
│   ├── search_service.py         # hybrid_search / full_search / review_search
│   ├── recommend_service.py      # thin async wrapper around agents/workflow.py
│   ├── feedback_service.py       # save_feedback, recompute_profile, apply_profile_boost
│   ├── memory_service.py         # short-term (RAM) + long-term (Postgres) memory
│   ├── analytics_service.py      # search_logs / feedback aggregation queries
│   ├── ingestion_service.py      # shared insert→embed→BM25 pipeline
│   ├── contact_service.py        # mailto: / wa.me link generation — NO LLM, NO webhook
│   ├── enrichment_services.py    # Google Places review+photo enrichment, CLIP image embed (PAID path)
│   ├── image_embedding_service.py # NEW — free local-CLIP image embedding using photo URLs
│   │                                 already in Postgres (Apify-sourced). Same CLIP model and
│   │                                 same ChromaDB collection as enrichment_services.py's image
│   │                                 step; only difference is where the URL came from and cost.
│   └── auth/                     # see backend/services/auth/CLAUDE.md — errors, core, google_oauth,
│                                     email, rate_limit, audit, totp (7 files, split by concern)
├── agents/
│   ├── llm.py                  # AGENT_CONFIGS dict (role/model per agent) + call_agent() — this is
│   │                              what CLAUDE.md used to call "configs.py"; that file doesn't exist,
│   │                              the configs live inline in llm.py alongside call_agent() itself
│   ├── workflow.py              # THE ORCHESTRATOR — see "Agent workflow" section below
│   └── reranker.py              # standalone, currently UNUSED alternative to workflow.py's Phase 3 —
│                                    nothing calls it; workflow.py calls call_agent("reranker", ...) directly
├── retrieving/
│   ├── embedder.py               # text embedding functions (calls data_enrichment.py for rich text)
│   ├── vector_store.py            # ALL ChromaDB ops — 3 collections, see "Vector store" section.
│   │                                 Also where the CLIP model itself lives (_get_clip(), lazy-loaded)
│   ├── bm25_store.py               # BM25 build/search, persisted to bm25_index.pkl
│   ├── retrieval.py                 # RRF fusion of BM25 + dense — the core hybrid_search() function
│   └── data_enrichment.py            # cuisine normalisation + name-based cuisine inference + rich text builder
└── data_loader/
    ├── review_summariser.py       # cautious LLM review summarisation with recency weighting
    ├── restaurant_fetcher.py      # OSM Overpass + Foursquare fetch (free, weekly sync)
    ├── apify_loader.py            # normalize_apify_place() + _load_places()/_load_places_fill_missing()
    └── apify_fetcher.py           # resilient Apify run (start→poll→fetch-dataset, survives credit
                                       exhaustion) — this is the file referred to elsewhere as "apify
                                       automation"; it was renamed from apify_automation.py at some point
                                       without every doc being updated, hence this note. ALSO still
                                       contains generate_contact_messages/generate_whatsapp_url/
                                       generate_gmail_compose_url/_get_restaurant_contact_method — an
                                       older LLM-based contact-message approach, fully superseded by
                                       services/contact_service.py's template-based one. Dead code,
                                       nothing calls these four. _get_restaurant_contact_method is also
                                       broken (calls dict-style .get() on a SQLAlchemy ORM object). Safe
                                       to delete; flagged twice now (once in auth.md's own history, once
                                       here) without being removed — check with the project owner before
                                       assuming it's still wanted before deleting.

mcp_service/                      # OPTIONAL — not mounted into main.py, run as its own process.
├── mcp_server.py                  # FastMCP — exposes search/recommend/feedback as MCP tools.
│                                     See mcp_service/CLAUDE.md.
└── mcp_client.py                  # client + ReAct ChatGroq/Gemini loop

frontend/                         # See frontend/CLAUDE.md for the full login/auth wiring.
├── app.py                         # Streamlit, 5 tabs (added Account), login-gated via render_login()
├── login_page.py                  # Login/register/Google/2FA/password-reset/account-linking screens
└── api_client.py                  # authed_request() — attaches Bearer token, auto-refreshes on 401
```

### Golden rules for this codebase
1. **Routers never contain business logic.** If a router function is more than
   ~15 lines of actual logic (not counting docstrings), that logic belongs in
   a service.
2. **Services never import FastAPI.** Keeps them testable and reusable from
   scripts, CLI tools, MCP server, etc. The one intentional exception in
   spirit (not letter) is `services/auth/*` raising a plain `AuthError` —
   routers translate that to the right HTTP status, so even auth services
   stay FastAPI-free.
3. **All Pydantic models live in `models/schemas.py`.** Don't define request
   models inline in routers.
4. **All env vars go through `core/config.py`.** Never `os.environ.get()`
   scattered across files. (This was violated once — `GOOGLE_PLACES_API_KEY`
   in `enrichment_services.py` — and has been fixed; don't reintroduce it.)
5. **Ingestion (writes) is namespaced under `/ingestion/*`** and every route
   there requires `require_admin` — this is now enforced, not aspirational.
6. **Auth business logic never lives outside `services/auth/`.** Routers in
   `routers/auth/` should only validate input, call a `services/auth/*`
   function, and translate `AuthError`/`RateLimitError` to HTTP. See
   `backend/services/auth/CLAUDE.md`.

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

This line is in every single agent's system prompt (`agents/llm.py`).
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
API error). This fallback is inside `agents/llm.py::call_agent()` —
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

### apify_fetcher.py — credit-loss resilience

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
(`apify_fetcher.py`) or Apify returns empty `reviews: []` and
`imageUrls: []` — this is not an extra API call or extra cost, just a
parameter that was initially missing.

### enrichment_services.py — Google Places as a secondary path

For restaurants already in the DB (from Apify without reviews, or from
OSM/Foursquare which never have reviews), `enrich_restaurants()` uses
the stored `external_id` (placeId) to call Google Places Details
directly — no re-scraping the restaurant itself. For OSM records lacking
a placeId, does a Text Search first to find one.

**Known constraint:** Google Places API requires a billing card on file
even for the $200/month free credit — this is Google Cloud policy, not
a bug in our code. If unavailable, Apify (with maxReviews/maxImages set)
is the only reviews+photos source that requires no card at all.

### image_embedding_service.py — the free counterpart, for photos you already have

`enrichment_services.py`'s image embedding only ever runs as a side
effect of a paid Google Places Details call. But Apify's `photos`
column already contains direct, hotlinkable image URLs — no
`photo_reference` token, no Google API call needed to resolve one.
`image_embedding_service.embed_restaurant_photos()` downloads those
URLs directly and calls the exact same `vector_store.upsert_restaurant_image()`
/ CLIP model as the Google path — same collection, same vector space,
zero marginal cost beyond bandwidth and local compute. Exposed as
`POST /ingestion/embed-restaurant-images`. Idempotent — a restaurant is
skipped once every URL in its `photos` list has a matching
`image_index` already in ChromaDB, so it's safe to call again after
every new Apify sync.

Don't confuse this with a "second, different implementation" — it's
the same `_get_clip()` / `embed_image_from_bytes()` in `vector_store.py`
either way. The two services just differ in **where the URL comes
from** (already in Postgres vs. fetched fresh from Google) and
**what it costs** (free vs. ~$0.017/restaurant).

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

Optional, standalone — NOT mounted into `main.py`, run as its own
process (`python mcp_service/mcp_server.py`) only if you want MCP tool
access (Claude Desktop, this project's own `mcp_client.py`, or other
agents). The main app runs fully without it. See `mcp_service/CLAUDE.md`
for the full picture, including a real bug that was in here for a
while: it imported `Restaurant` from `models.schemas` (a Pydantic
module with no such class) instead of `models.db_models` (the actual
SQLAlchemy ORM class `select(Restaurant)` needs) — meaning this file
could not even be imported, let alone run, until that was fixed. If
you're reading an old commit or a stale clone, check that import first
before debugging anything else in this file.

Exposes 5 tools + 1 resource over stdio transport:

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

## Authentication & authorization (`services/auth/`, `routers/auth/`)

Full design detail, Redis key namespaces, and the reasoning behind
every security choice: **`backend/services/auth/CLAUDE.md`**. Summary
for orientation:

- Self-hosted (no third-party auth provider) — email/password +
  Google OAuth (Authorization Code flow) both issue the same JWT
  access/refresh token pair. Admin is a `role` column on the same
  `User`/JWT system, not a separate API key or service.
- `/ingestion/*` requires `require_admin`; nothing else in the app
  requires auth at the router level (search/recommend stay public by
  design) except the `/auth/*` routes that explicitly need a session
  (`/auth/me`, `/auth/sessions`, `/auth/2fa/*`, `/auth/deactivate`).
- Also covers: password reset, account deactivation, session/device
  listing + revocation, TOTP 2FA, Redis-backed rate limiting on
  register/resend-verification/forgot-password, login lockout after 5
  failed attempts, and an append-only `audit_logs` table.
- The frontend's login/register/2FA/linking screens and the
  `authed_request()` pattern every other frontend call goes through
  are covered in `frontend/CLAUDE.md`, not here.

---

## Environment variables (`core/config.py`)

Full reference with explanations: `.env.example` — that file is
generated from this module's actual `Settings` class, so trust the
code over any doc if they ever drift.

```
# Required
NEON_DATABASE_URL          # PostgreSQL connection string
JWT_SECRET_KEY              # generate: python -c "import secrets; print(secrets.token_urlsafe(64))"

# At least one required (Groq tried first, Gemini as fallback)
GROQ_API_KEY
GOOGLE_API_KEY

# Required for Apify ingestion; everything below is optional and
# degrades gracefully without it
APIFY_API_TOKEN             # no card needed, $5/mo free
FOURSQUARE_API_KEY          # base place search only, reviews/photos are Premium-only (paid)
GOOGLE_PLACES_API_KEY       # requires a billing card, $200/mo free credit
GOOGLE_CLIENT_ID            # Google sign-in
GOOGLE_CLIENT_SECRET
REDIS_URL                    # defaults to redis://localhost:6379/0
JWT_ACCESS_EXPIRE_MINUTES     # default 30
JWT_REFRESH_EXPIRE_DAYS        # default 7
BACKEND_URL                     # default http://localhost:8000 — where emailed links point
FRONTEND_URL                     # default http://localhost:8501
SMTP_HOST / PORT / USER / PASSWORD   # unset → verification/reset links print to console instead
N8N_WEBHOOK_URL                        # legacy, not currently used by any active endpoint
```

---

## Common commands

```bash
# Run everything
bash run.sh                                    # option 3 = backend + frontend

# Manual
uvicorn backend.main:app --reload
streamlit run frontend/app.py
python mcp_service/mcp_server.py               # optional

# Get an admin token first — every /ingestion/* call below needs it
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"yourpassword123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Ingest data
curl -X POST "http://localhost:8000/ingestion/n8n/sync-apify?per_city_limit=50" -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:8000/ingestion/summarise-all-reviews -H "Authorization: Bearer $TOKEN"
curl -X POST "http://localhost:8000/ingestion/embed-restaurant-images?limit=10" -H "Authorization: Bearer $TOKEN"   # free, local CLIP
curl -X POST "http://localhost:8000/ingestion/enrich-reviews?limit=10&embed_images=true" -H "Authorization: Bearer $TOKEN"  # paid, Google Places

# Check state (also admin-gated)
curl http://localhost:8000/ingestion/enrich-status -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/ingestion/n8n/apify-status -H "Authorization: Bearer $TOKEN"

# Check state (public)
curl http://localhost:8000/vector-stats
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
  pattern in `apify_fetcher.py`.
- Don't route contact messages through n8n or an LLM — `contact_service.py`
  is intentionally template-based and client-side only.
- Don't add password/session/token logic anywhere outside `services/auth/`
  — routers only validate input and translate exceptions. See
  `backend/services/auth/CLAUDE.md`'s own "don't" list for auth-specific
  pitfalls (exception ordering, Redis key collisions, etc.) — they're
  detailed enough to deserve their own file rather than duplicating here.
- Don't call `requests`/`httpx` directly from a Streamlit script body
  without a try/except — a body-level call executes on every rerun of
  every tab, so one network hiccup can crash the whole page, not just
  the feature that made the call. See `frontend/CLAUDE.md`.