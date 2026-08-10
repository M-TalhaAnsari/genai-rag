"""
backend/routers/ingestion.py
------------------------------
All data ingestion endpoints — kept separate from read/search endpoints.

POST /ingestion/restaurant-sync         — ingest any restaurant list → PostgreSQL + ChromaDB + BM25
GET  /ingestion/fetch-restaurants       — fetch from OSM + Foursquare (n8n calls this)
POST /ingestion/load-apify              — one-time manual Apify JSON file load
POST /ingestion/summarise-all-reviews   — batch generate review summaries
POST /ingestion/n8n/sync-apify          — fully automated Apify sync (n8n calls this on schedule)
GET  /ingestion/n8n/apify-status        — data volume check without triggering a sync

Note on access control:
  These endpoints write to the database and should eventually sit behind
  admin auth (API key or JWT). For now they are open — add a dependency
  like `Depends(verify_admin_key)` when you implement RBAC.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.models.db_models import Restaurant, Review
from backend.models.schemas import RestaurantRequest, SyncResponse
from backend.retrieving import vector_store, bm25_store

from sqlalchemy import func
from backend.models.db_models import Review as ReviewModel
from backend.retrieving.vector_store import (review_collection_count,
                                             upsert_review_summary, get_review_summary
                                            )

from backend.data_loader.restaurant_fetcher import fetch_all_cities, TARGET_CITIES
from backend.data_loader.apify_loader import load_apify_export, _load_places
from backend.data_loader.review_summariser import summarise_reviews as _summarise

from backend.data_loader.apify_fetcher import _run_apify_scraper_resilient

from backend.services.enrichment_services import enrich_restaurants

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


# ── Fetch from OSM + Foursquare (n8n weekly sync) ──────────────────────────

@router.get("/fetch-restaurants")
def fetch_restaurants(
    cities: str | None = Query(
        default=None,
        description="Comma-separated city names. Default: Lahore, Islamabad, Karachi, Rawalpindi"
    )
):
    """
    Fetch real restaurant data from OSM + Foursquare.
    n8n calls this, then POSTs the response to /ingestion/restaurant-sync.

    Free, no Apify credits used.
    OSM: no key required. Foursquare: FOURSQUARE_API_KEY (free 100k/month).
    """
    

    city_list = (
        [c.strip() for c in cities.split(",") if c.strip()]
        if cities else TARGET_CITIES
    )
    try:
        restaurants = fetch_all_cities(cities=city_list)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Fetch failed: {e}"
        )
    return {"cities_fetched": city_list, "total": len(restaurants), "restaurants": restaurants}


# ── Restaurant sync (receives data from n8n or direct POST) ────────────────

@router.post("/restaurant-sync", response_model=SyncResponse)
async def restaurant_sync(
    request: RestaurantRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a list of restaurants into PostgreSQL + ChromaDB + BM25.

    Called by n8n after GET /ingestion/fetch-restaurants.
    Also callable directly with any restaurant data.

    Deduplicates by (name, city) — safe to call repeatedly.
    Only newly inserted restaurants are embedded — skipped ones are untouched.
    """
    saved, skipped, newly_inserted = [], [], []

    for item in request.restaurants:
        result = await db.execute(
            select(Restaurant).where(
                Restaurant.name == item.name,
                Restaurant.city == item.city
            )
        )
        if result.scalars().first():
            skipped.append(item.name)
            continue

        record = Restaurant(**item.model_dump(), is_embedded=False)
        db.add(record)
        newly_inserted.append(record)
        saved.append(item.name)

    await db.commit()

    embedded_count = 0
    for record in newly_inserted:
        await db.refresh(record)
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
            embedded_count += 1
        except Exception as e:
            print(f"[embed error] {record.name}: {e}")

    await db.commit()

    if newly_inserted:
        all_result = await db.execute(select(Restaurant))
        all_recs = all_result.scalars().all()
        bm25_store.build_index([
            {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
            for r in all_recs
        ])

    return SyncResponse(
        message="Sync complete",
        inserted_count=len(saved),
        skipped_count=len(skipped),
        embedded_count=embedded_count,
        inserted=saved,
        skipped=skipped
    )


# ── One-time manual Apify load ─────────────────────────────────────────────

@router.post("/load-apify")
async def load_apify():
    """
    One-time bulk load from data/apify_export.json.

    Steps:
    1. Download Apify Google Maps Scraper export as JSON
    2. Save to data/apify_export.json
    3. Call this endpoint

    For automated/scheduled loading, use POST /ingestion/n8n/sync-apify instead.
    """

    result = await load_apify_export()
    if "error" in result:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])
    return result


# ── Batch review summarisation ─────────────────────────────────────────────

@router.post("/summarise-all-reviews")
async def summarise_all_reviews(db: AsyncSession = Depends(get_db)):
    """
    Generate cautious review summaries for all restaurants that have
    reviews but no summary yet in ChromaDB.

    Run this once after /load-apify to populate the review_summaries collection.
    Already-summarised restaurants are skipped automatically.
    """

    r_result  = await db.execute(select(Restaurant))
    restaurants = r_result.scalars().all()

    done, skipped, errors = 0, 0, []

    for restaurant in restaurants:
        if get_review_summary(restaurant.id):
            skipped += 1
            continue

        rv_result = await db.execute(
            select(Review).where(Review.restaurant_id == restaurant.id)
        )
        reviews = rv_result.scalars().all()

        if not reviews:
            skipped += 1
            continue

        try:
            review_dicts = [
                {"rating": r.rating, "text": r.text,
                 "published_date": r.published_date, "source": r.source}
                for r in reviews
            ]
            summary_data = _summarise(
                restaurant_name=restaurant.name,
                cuisine=restaurant.cuisine,
                reviews=review_dicts
            )
            upsert_review_summary(
                restaurant_id=restaurant.id,
                restaurant_name=restaurant.name,
                cuisine=restaurant.cuisine,
                city=restaurant.city,
                **summary_data
            )
            done += 1
        except Exception as e:
            errors.append(f"{restaurant.name}: {e}")

    return {"summarised": done, "skipped": skipped, "errors": errors[:10]}


# ── n8n automated Apify sync ───────────────────────────────────────────────

@router.post("/n8n/sync-apify")
async def n8n_sync_apify(
    cities: str | None = Query(default=None),
    per_city_limit: int = Query(default=15)
):
    """
    Fully automated, credit-loss-resilient Apify sync triggered by n8n.

    Flow:
      1. Starts Apify actor run (non-blocking)
      2. Polls status every 10 seconds
      3. Fetches dataset regardless of how the run ended
         (credits exhausted mid-run → partial data still saved)
      4. Filters to Pakistan only (countryCode=PK)
      5. Inserts new restaurants → PostgreSQL + ChromaDB + BM25 + review summaries

    per_city_limit:
      50  → first bootstrap run
      15  → scheduled weekly re-sync (cheaper, catches new openings)
    """

    city_list = (
        [c.strip() for c in cities.split(",") if c.strip()]
        if cities
        else ["Lahore", "Islamabad", "Karachi", "Rawalpindi"]
    )

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

    result = await _load_places(raw_places)

    total_processed = result["inserted"] + result["skipped"]
    duplicate_ratio = (
        round(result["skipped"] / total_processed * 100, 1)
        if total_processed > 0 else 0.0
    )

    return {
        "message": (
            "Apify sync complete" if not run_info["is_partial"]
            else "Partial sync — credits may have run out. Recovered data was saved."
        ),
        "run_status":             run_info["status"],
        "is_partial":             run_info["is_partial"],
        "run_message":            run_info["message"],
        "cities":                 city_list,
        "raw_places_fetched":     len(raw_places),
        "new_vs_duplicate_ratio": f"{duplicate_ratio}% already in database",
        **result,
    }


@router.get("/n8n/apify-status")
async def n8n_apify_status(db: AsyncSession = Depends(get_db)):
    """
    Quick data volume check — n8n can poll this to verify freshness
    without triggering a new sync.
    """
    result = await db.execute(select(Restaurant))
    restaurants = result.scalars().all()

    by_source = {}
    for r in restaurants:
        src = r.source or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

    return {
        "total_restaurants": len(restaurants),
        "by_source":         by_source,
        "chroma_vectors":    vector_store.collection_count(),
        "bm25_index_exists": bm25_store.index_exists(),
    }


# ── Google Places enrichment ───────────────────────────────────────────────

@router.post("/enrich-reviews")
async def enrich_reviews(
    limit: int | None = Query(
        default=50,
        description="Max restaurants to enrich per run. Start small to test. None = all."
    ),
    only_without_reviews: bool = Query(
        default=True,
        description="Skip restaurants that already have reviews. Recommended: True."
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Enrich existing restaurants with reviews + photos from Google Places API.

    Uses the placeId (external_id) already stored from Apify to call
    Google Places Details — one call per restaurant, same response contains
    both reviews and photo references.

    For OSM-sourced restaurants without a placeId, does a Text Search first
    to find the correct place, then fetches details.

    Cost: ~$0.017 per restaurant. 600 restaurants ≈ $10.20 total.
    Free tier: $200/month credit. Well within budget.

    Required env var: GOOGLE_PLACES_API_KEY
    Get a free key: console.cloud.google.com → APIs → Places API

    Run order:
      1. POST /ingestion/enrich-reviews?limit=10   ← test with 10 first
      2. Check the reviews table in Neon
      3. POST /ingestion/enrich-reviews?limit=None ← run on all if test looks good
    """

    try:
        result = await enrich_restaurants(
            db=db,
            only_without_reviews=only_without_reviews,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    return result


@router.get("/enrich-status")
async def enrich_status(db: AsyncSession = Depends(get_db)):
    """
    Check how many restaurants have been enriched with reviews.
    Run this before and after /enrich-reviews to see progress.

    Returns:
      total_restaurants        — all restaurants in PostgreSQL
      with_reviews             — restaurants that have at least one review
      without_reviews          — restaurants still missing reviews
      total_reviews            — total review rows in the reviews table
      with_photos              — restaurants that have photo URLs stored
      review_summaries_in_chroma — restaurants summarised in ChromaDB
    """
    

    r_result = await db.execute(select(Restaurant))
    all_restaurants = r_result.scalars().all()
    total = len(all_restaurants)

    have_photos = sum(1 for r in all_restaurants if r.photos)

    review_result = await db.execute(
        select(ReviewModel.restaurant_id).distinct()
    )
    enriched_ids = set(review_result.scalars().all())

    total_reviews_result = await db.execute(
        select(func.count()).select_from(ReviewModel)
    )
    total_reviews = total_reviews_result.scalar() or 0

    return {
        "total_restaurants":         total,
        "with_reviews":              len(enriched_ids),
        "without_reviews":           total - len(enriched_ids),
        "total_reviews":             total_reviews,
        "with_photos":               have_photos,
        "review_summaries_in_chroma": review_collection_count(),
    }
