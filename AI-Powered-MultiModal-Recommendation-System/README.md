# 🍽️ Connoisseur — AI Restaurant Discovery for Pakistan

Production-grade AI backend for restaurant discovery across **Lahore, Islamabad, Karachi, and Rawalpindi**. Hybrid search, a 6-agent recommendation pipeline, persistent memory, and clickable contact links.

> **This README supersedes the original project README for everything related to data ingestion.** The Apify sync pipeline was substantially rewritten to fix real bugs found during actual scraping runs. See `HANDOFF.md` for the full story; this file documents the corrected, current behavior only.

---

## Architecture

```
Data ingestion                    Storage                      Retrieval
──────────────                    ───────                      ─────────
Apify (bulk + backfill)   →      PostgreSQL (Neon)      →     BM25 (keyword)
OSM + Foursquare (weekly) →      ChromaDB × 3:           →     Dense (identity)
Google Places (fallback)  →        restaurants                Review sentiment
                                     restaurant_reviews          Image (CLIP, deferred)
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
```

---

## Project structure

```
connoisseur/
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py          # pool_pre_ping + pool_recycle=180 — see "Neon connection notes" below
│   │   └── security.py          # JWT auth + role-based access (require_admin, require_user)
│   ├── models/
│   │   ├── db_models.py         # includes ApifyRawSnapshot + ApifyRunLog (new)
│   │   └── schemas.py
│   ├── routers/
│   │   ├── auth.py               # register/login/refresh/verify-email/Google OAuth
│   │   └── ingestion.py          # see "Ingestion endpoints" below — significantly changed
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── auth_service.py
│   │   ├── google_oauth_service.py
│   │   ├── email_service.py
│   │   └── enrichment_service.py # Google Places fallback — manual, opt-in only
│   ├── data_loader/
│   │   ├── apify_automation.py   # rewritten — search-term builder, snapshot-to-Neon, run logging
│   │   ├── apify_loader.py       # rewritten — batched fill-missing loader, error isolation
│   │   ├── restaurant_fetcher.py # OSM + Foursquare (unchanged)
│   │   └── review_summariser.py  # unchanged, still manual/deferred
│   ├── retrieving/
│   │   ├── vector_store.py
│   │   └── bm25_store.py
│   └── agents/                   # unchanged, out of scope for this handoff
├── mcp_service/
├── frontend/
└── alembic/versions/
    ├── 0001_add_auth_tables.py
    ├── 0002_google_oauth.py
    └── 0003_email_verification.py
```

---

## Environment variables

```env
NEON_DATABASE_URL=postgresql+asyncpg://user:pass@host/db?ssl=require
GROQ_API_KEY=...
GOOGLE_API_KEY=...
APIFY_API_TOKEN=...                # apify.com — no card needed, $5/mo free tier
FOURSQUARE_API_KEY=...             # optional
GOOGLE_PLACES_API_KEY=...          # optional — requires billing card, fallback only

# Auth
JWT_SECRET_KEY=...
JWT_ALGORITHM=HS256
JWT_ACCESS_EXPIRE_MINUTES=30       # short-lived — see "Auth notes" below
JWT_REFRESH_EXPIRE_DAYS=7
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_URL=http://localhost:8501
SMTP_HOST=                          # optional — unset means verification links print to console
```

---

## First-time setup

```powershell
# 1. Install deps
pip install -r requirements.txt

# 2. Apply auth migrations
alembic upgrade head

# 3. Create ingestion tables (not yet in alembic — ApifyRawSnapshot, ApifyRunLog)
python -c "
import asyncio
from backend.core.database import engine, Base
from backend.models import db_models

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(main())
"

# 4. Start backend
uvicorn backend.main:app --reload
```

---

## Getting an admin token (PowerShell)

```powershell
# Register
$body = @{ email = "you@example.com"; password = "yourpassword123" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/auth/register" -Method Post -ContentType "application/json" -Body $body

# Verify email — token is printed in the uvicorn console if SMTP isn't configured
$verifyBody = @{ token = "PASTE_TOKEN_HERE" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/auth/verify-email" -Method Post -ContentType "application/json" -Body $verifyBody

# Promote to admin directly in Neon
python -c "
import asyncio
from sqlalchemy import update
from backend.core.database import AsyncSessionLocal
from backend.models.db_models import User, UserRole

async def main():
    async with AsyncSessionLocal() as db:
        await db.execute(update(User).where(User.email == 'you@example.com').values(role=UserRole.admin))
        await db.commit()

asyncio.run(main())
"

# Log in
$loginBody = @{ email = "you@example.com"; password = "yourpassword123" } | ConvertTo-Json
$tokens = Invoke-RestMethod -Uri "http://localhost:8000/auth/login" -Method Post -ContentType "application/json" -Body $loginBody
$headers = @{ Authorization = "Bearer $($tokens.access_token)" }
```

**Access tokens expire fast (`JWT_ACCESS_EXPIRE_MINUTES`).** If a long-running ingestion call returns `"Could not validate credentials"`, that's a fully separate, unrelated failure from anything happening on Apify's side — just re-run the login block above and retry. See `HANDOFF.md` for why token expiry can never interrupt an in-flight Apify scrape.

---

## Ingestion endpoints (rewritten)

### Check before you spend anything

```powershell
# Free — current DB volume, zero Apify cost
Invoke-RestMethod -Uri "http://localhost:8000/ingestion/n8n/apify-status" -Headers $headers

# Free — which restaurants are missing reviews/photos/etc.
Invoke-RestMethod -Uri "http://localhost:8000/ingestion/restaurants-missing-data" -Headers $headers

# Free — per-field completeness counts
Invoke-RestMethod -Uri "http://localhost:8000/ingestion/enrich-status" -Headers $headers

# Free — history of past runs, incl. duplicate ratio per search term
Invoke-RestMethod -Uri "http://localhost:8000/ingestion/n8n/run-history" -Headers $headers
```

### The main sync endpoint

```
POST /ingestion/n8n/sync-apify
```

| Param | Default | Notes |
|---|---|---|
| `cities` | all 4 | comma-separated |
| `per_city_limit` | 15 | applies **per search term**, not per city — matters a lot in `cuisine-specific`/`neighborhood` modes |
| `search_mode` | `broad` | `broad \| cafes \| fastfood \| bakeries \| cuisine-specific \| neighborhood` |
| `max_reviews` | 5 | tuned value — see `HANDOFF.md` for why 5, not 3 |
| `max_images` | 3 | **URLs only** — no image bytes are downloaded or stored anywhere yet |
| `fill_missing` | `true` | when `true`, existing restaurants matched by `external_id`/name+city get **any empty field filled in** (not just reviews/photos) instead of being skipped |

Example — targeted backfill of a specific city/neighborhood, cheap and high new-restaurant yield:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ingestion/n8n/sync-apify?cities=Lahore&search_mode=neighborhood&per_city_limit=5&max_reviews=5&max_images=3&fill_missing=true" `
  -Method Post -Headers $headers
```

**`search_mode` reference — use to avoid re-scraping the same top results:**
- `broad` — saturates fastest; avoid repeating on the same cities
- `cafes` / `fastfood` / `bakeries` — different Google category, surfaces a different result set
- `cuisine-specific` — one search per cuisine keyword × city (10 cuisines × N cities = 10N terms — watch `per_city_limit`)
- `neighborhood` — one search per known neighborhood × city (**highest yield for genuinely new restaurants**, since it resets Google's ranking window entirely instead of re-querying the same city-wide top-N)

### Response fields worth checking

```json
{
  "message": "...",
  "run_status": "SUCCEEDED | FAILED | ABORTED | TIMED-OUT",
  "is_partial": false,
  "snapshot_id": 2,
  "raw_places_fetched": 200,
  "inserted": 120,
  "filled_existing": 80,
  "reviews_added": 995,
  "skipped_wrong_country": 0,
  "errors": []
}
```

`is_partial: true` means Apify's run didn't finish cleanly (commonly: credits ran out mid-scrape) — **the response still reflects everything actually recovered and saved**. Nothing scraped before the cutoff is lost. See `HANDOFF.md` §"Credit-loss resilience, confirmed working" for the real incident this was validated against.

### Manual, opt-in fallback (Google Places)

```
POST /ingestion/enrich-reviews?limit=10&embed_images=true
```

Only relevant now for restaurants sourced from OSM/Foursquare (which never carry reviews/photos) or from a partial Apify run. Requires a billing card (Google Cloud policy) and costs ~$0.017/restaurant. **Not part of the default flow anymore** — Apify with `maxReviews`/`maxImages` set is the primary and cheaper source for everything.

---

## Neon connection notes

```python
# backend/core/database.py
engine = create_async_engine(
    settings.NEON_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=180,   # tightened after a real incident — see HANDOFF.md
)
```

Large ingestion runs (200+ places, hundreds of review inserts) held a single DB session open long enough to hit Neon's idle-connection timeout mid-request, producing a `500` on the *response* even though every actual write had already committed successfully. The loader is now batched (25 places per session — see `_load_places_fill_missing` in `apify_loader.py`) specifically to keep each session short-lived.

**If you ever see a `500` with an `asyncpg.exceptions.InterfaceError` / "cannot call Transaction.rollback(): the underlying connection is closed"** immediately after log lines showing `"Fill-missing done: ..."` — that is a teardown-phase error, not a data-loss error. Check `/ingestion/enrich-status` before assuming anything was lost.

---

## Image handling — current state

`maxImages` in the Apify actor input controls how many **image URLs** come back per place (`photos` column, JSON list of strings) — it does **not** download actual image bytes. This matches the schema already in place. Actual image embedding into the `restaurant_images` ChromaDB collection (CLIP-based) is a **deferred, separate step**: fetch bytes from the stored URLs, run CLIP locally, upsert — decoupled entirely from ingestion, zero additional API cost, can be run whenever convenient. Not yet implemented as of this handoff.

---

## Review summarization — deferred by design

Reviews are inserted into Postgres during ingestion (`Review` table), but LLM-based summarization into ChromaDB (`review_summariser.py`, `/ingestion/summarise-all-reviews`) is intentionally **not** run automatically after every sync. Run it manually once you're ready:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ingestion/summarise-all-reviews" -Method Post -Headers $headers
```

---

## Hallucination handling, token strategy, agent workflow

Unchanged from the original design — see `HANDOFF.md` if you need the full multi-agent recommendation pipeline explained; it wasn't touched in this round of work.

---

## Data sources — comparison

| Source | Reviews | Photos (URLs) | Card required | Cost |
|---|---|---|---|---|
| Apify Google Maps | ✅ (`maxReviews`) | ✅ (`maxImages`) | ❌ No | $5/mo free, resets monthly |
| OpenStreetMap | ❌ | ❌ | ❌ No | Free, unlimited |
| Foursquare | ❌ (Premium-only) | ❌ (Premium-only) | No (base search) | Base free, reviews/photos paid |
| Google Places | ✅ | ✅ | ✅ Yes | $200/mo free credit |