"""
backend/apify_automation.py
------------------------------
Fully automated Apify data pipeline — NO manual export/import step.

This replaces the manual workflow (run actor on Apify website → download
JSON → save to data/apify_export.json → POST /load-apify) with a single
API call that n8n triggers on a schedule.

HOW IT WORKS
-------------
1. n8n calls POST /n8n/sync-apify (on whatever schedule you want —
   weekly, daily, etc.)
2. This endpoint calls the Apify API directly using APIFY_API_TOKEN
3. Apify runs the Google Maps scraper actor SYNCHRONOUSLY and returns
   the scraped restaurants directly in the response (no polling needed)
4. We normalise the data (same logic as apify_loader.py — reused, not
   duplicated) and insert into PostgreSQL + ChromaDB + BM25
5. Review summaries are generated automatically for new restaurants

SETUP REQUIRED
---------------
1. Sign up at apify.com (free, $5/month credits, no card)
2. Get your API token: Settings → Integrations → API token
3. Add to .env:  APIFY_API_TOKEN=your_token_here

n8n workflow (simple):
    Schedule Trigger (weekly)
        → POST http://your-server:8000/n8n/sync-apify
        → (optional) Slack/Email notification with the response summary

No manual file download. No manual upload. Fully automated.
"""

from fastapi import APIRouter
from sqlalchemy import select

from backend.db.database import AsyncSessionLocal
from backend.model.models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store
from backend.data_loader.apify_loader import normalize_apify_place, _load_places # reuse existing logic


from n8n.backend.contact import _run_apify_scraper_resilient
router = APIRouter(prefix="/n8n", tags=["n8n automation"])



APIFY_ACTOR_ID = "compass~crawler-google-places"   # Google Maps scraper

DEFAULT_CITIES = ["Lahore", "Islamabad", "Karachi", "Rawalpindi"]
DEFAULT_PER_CITY_LIMIT = 50


APIFY_ACTOR_ID = "compass~crawler-google-places"

RECOMMENDED_INCREMENTAL_LIMIT = 15   # scheduled weekly re-sync — cheaper

@router.post("/sync-apify")
async def sync_apify(
    cities: str = None,
    per_city_limit: int = RECOMMENDED_INCREMENTAL_LIMIT
):
    """
    Run through n8n

    Query params:
      cities:          comma-separated city names, default: all 4 cities
      per_city_limit:  max restaurants per city.
                        Recommended: 50 for your FIRST bootstrap run,
                        10-15 for regular scheduled re-syncs (cheaper —
                        Apify re-scrapes the same top-rated places every
                        time regardless of what you already have stored).

    Returns a summary including:
      run_status:               Apify's final run status
      is_partial:                True if credits ran out / run didn't finish
      raw_places_fetched:        how many places Apify actually returned
      inserted / skipped:        new vs already-in-database
      skipped_wrong_country:     places rejected for not being in Pakistan
      new_vs_duplicate_ratio:    helps you tune per_city_limit over time
    """
    city_list = (
        [c.strip() for c in cities.split(",") if c.strip()]
        if cities else DEFAULT_CITIES
    )

    print(f"[n8n-apify] Starting resilient sync for cities: {city_list}")

    raw_places, run_info = await _run_apify_scraper_resilient(city_list, per_city_limit)

    if not raw_places:
        return {
            "message":     "No places recovered from Apify.",
            "run_status":  run_info["status"],
            "run_message": run_info["message"],
            "cities":      city_list,
            "inserted":    0,
            "skipped":     0,
        }

    # Load into PostgreSQL + ChromaDB + BM25 (shared logic with manual loader)
    result = await _load_places(raw_places)

    total_processed = result["inserted"] + result["skipped"]
    duplicate_ratio = (
        round(result["skipped"] / total_processed * 100, 1)
        if total_processed > 0 else 0.0
    )

    return {
        "message":                 "Apify sync complete" if not run_info["is_partial"]
                                    else "Apify sync partially completed — credits or run may have been interrupted, but recovered data was saved.",
        "run_status":               run_info["status"],
        "is_partial":                run_info["is_partial"],
        "run_message":               run_info["message"],
        "cities":                    city_list,
        "raw_places_fetched":        len(raw_places),
        "new_vs_duplicate_ratio":    f"{duplicate_ratio}% were already in database",
        **result,
    }


# ── Status check ─────────────────────────────────────────────────────────

@router.get("/apify-status")
async def apify_status():
    """Quick health check — current data volume without triggering a sync."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Restaurant))
        restaurants = result.scalars().all()

    total = len(restaurants)
    by_source = {}
    by_country_flag = {"pk_verified": 0, "unverified": 0}

    for r in restaurants:
        src = r.source or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

    return {
        "total_restaurants": total,
        "by_source":         by_source,
        "chroma_vectors":    vector_store.collection_count(),
        "bm25_index_exists": bm25_store.index_exists(),
    }
