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

import os
import httpx
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from backend.db.database import AsyncSessionLocal
from backend.model.models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store
from backend.data_loader.apify_loader import normalize_apify_place   # reuse existing logic

router = APIRouter(prefix="/n8n", tags=["n8n automation"])



APIFY_ACTOR_ID = "compass~crawler-google-places"   # Google Maps scraper
APIFY_RUN_SYNC_URL = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"

DEFAULT_CITIES = ["Lahore", "Islamabad", "Karachi", "Rawalpindi"]
DEFAULT_PER_CITY_LIMIT = 50


@router.post("/sync-apify")
async def sync_apify(
    cities: str = None,
    per_city_limit: int = DEFAULT_PER_CITY_LIMIT
):
    """
    Fully automated Apify sync — no manual file download/upload.

    Call this from n8n on a schedule. It:
      1. Triggers the Apify Google Maps scraper via API
      2. Waits for results (synchronous — no polling needed)
      3. Deduplicates against existing PostgreSQL records
      4. Inserts new restaurants + their reviews
      5. Embeds new restaurants into ChromaDB
      6. Rebuilds the BM25 index
      7. Generates review summaries for new restaurants

    Query params:
      cities:          comma-separated city names, e.g. "Lahore,Karachi"
                        defaults to all 4 target cities
      per_city_limit:  max restaurants per city (Apify caps at ~50-100
                        efficiently), default 50

    Returns a summary dict — safe to call repeatedly, duplicates are skipped.

    n8n setup:
        Schedule Trigger (weekly) → HTTP Request node:
          Method: POST
          URL: http://your-server:8000/n8n/sync-apify
    """
    city_list = (
        [c.strip() for c in cities.split(",") if c.strip()]
        if cities else DEFAULT_CITIES
    )

    print(f"[n8n-apify] Starting sync for cities: {city_list}")

    # Step 1: Call Apify (this is the slow part — several minutes)
    raw_places = await _run_apify_scraper(city_list, per_city_limit)
    print(f"[n8n-apify] Apify returned {len(raw_places)} raw places")

    if not raw_places:
        return {
            "message": "Apify returned no results.",
            "cities": city_list,
            "inserted": 0,
            "skipped": 0,
        }

    # Step 2: Normalise + insert (reuses apify_loader's proven logic)
    inserted, skipped, reviews_inserted, embedded = 0, 0, 0, 0
    errors = []

    async with AsyncSessionLocal() as db:
        newly_inserted: list[tuple[Restaurant, list[dict]]] = []

        for place in raw_places:
            try:
                restaurant_data, reviews = normalize_apify_place(place)
            except Exception as e:
                errors.append(str(e))
                continue

            name = restaurant_data.get("name", "").strip()
            city = restaurant_data.get("city", "").strip()

            if not name:
                skipped += 1
                continue

            result = await db.execute(
                select(Restaurant).where(
                    Restaurant.name == name,
                    Restaurant.city == city
                )
            )
            if result.scalars().first():
                skipped += 1
                continue

            record = Restaurant(**restaurant_data, is_embedded=False)
            db.add(record)
            newly_inserted.append((record, reviews))
            inserted += 1

        await db.commit()

        # Step 3: reviews + embeddings for newly inserted only
        for record, reviews in newly_inserted:
            await db.refresh(record)

            for rv in reviews:
                db.add(Review(restaurant_id=record.id, **rv))
                reviews_inserted += 1

            try:
                vector_store.upsert_restaurant(
                    restaurant_id=record.id,
                    name=record.name,
                    cuisine=record.cuisine,
                    city=record.city,
                    address=record.address,
                    description=record.description,
                    tags=record.tags,
                    opening_hours=record.opening_hours,
                    phone=record.phone,
                    website=record.website,
                    rating=record.rating,
                    latitude=record.latitude,
                    longitude=record.longitude,
                )
                record.is_embedded = True
                embedded += 1
            except Exception as e:
                errors.append(f"embed {record.name}: {e}")

        await db.commit()

        # Step 4: rebuild BM25 from full table
        all_result = await db.execute(select(Restaurant))
        all_restaurants = all_result.scalars().all()
        bm25_store.build_index([
            {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
            for r in all_restaurants
        ])

        # Step 5: auto-summarise reviews for newly inserted restaurants
        summarised = 0
        if newly_inserted:
            from backend.review_summariser import summarise_reviews
            from backend.vector_store import upsert_review_summary

            for record, reviews in newly_inserted:
                if not reviews:
                    continue
                try:
                    summary_data = summarise_reviews(
                        restaurant_name=record.name,
                        cuisine=record.cuisine,
                        reviews=reviews
                    )
                    upsert_review_summary(
                        restaurant_id=record.id,
                        restaurant_name=record.name,
                        cuisine=record.cuisine,
                        city=record.city,
                        **summary_data
                    )
                    summarised += 1
                except Exception as e:
                    errors.append(f"summarise {record.name}: {e}")

    print(
        f"[n8n-apify] Done: {inserted} inserted, {skipped} skipped, "
        f"{reviews_inserted} reviews, {embedded} embedded, {summarised} summarised"
    )

    return {
        "message":             "Apify sync complete",
        "cities":               city_list,
        "raw_places_fetched":   len(raw_places),
        "inserted":             inserted,
        "skipped":              skipped,
        "reviews_inserted":     reviews_inserted,
        "embedded":              embedded,
        "reviews_summarised":   summarised,
        "errors":               errors[:10],
    }


# ── Status check (n8n can poll this to verify data freshness) ──────────────

@router.get("/apify-status")
async def apify_status():
    """
    Quick health check for the n8n workflow — shows current data volume
    without triggering a new sync. Useful for n8n IF conditions
    (e.g. only alert if restaurant count hasn't grown in X days).
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Restaurant))
        restaurants = result.scalars().all()

    total = len(restaurants)
    by_source = {}
    for r in restaurants:
        src = r.source or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

    return {
        "total_restaurants": total,
        "by_source":         by_source,
        "chroma_vectors":    vector_store.collection_count(),
        "bm25_index_exists": bm25_store.index_exists(),
    }
