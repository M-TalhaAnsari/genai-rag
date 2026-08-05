"""
backend/apify_loader.py
------------------------
Loads a one-time Apify Google Maps Scraper export into PostgreSQL + ChromaDB + BM25.

HOW TO GET YOUR APIFY DATA
---------------------------
1. Go to https://apify.com and sign up (free — $5/month credits, no card needed)
2. Open: https://apify.com/compass/crawler-google-places
3. Click "Try for free"
4. Set search terms (one per city):
     "restaurants in Lahore Pakistan"
     "restaurants in Islamabad Pakistan"
     "restaurants in Karachi Pakistan"
     "restaurants in Rawalpindi Pakistan"
5. Set maxCrawledPlacesPerSearch = 50 (200 total, fits in free $5 credits)
6. Run the actor
7. When done: Dataset → Export → JSON
8. Save the file as: data/apify_export.json
9. Run: python -m backend.apify_loader

The $5 free credit covers this one-time scrape (~200-500 restaurants).
After that, OSM + Foursquare handle weekly incremental updates.

WHAT APIFY PROVIDES PER RESTAURANT
------------------------------------
name, address, phone, website, rating, reviewCount, priceLevel,
totalScore, categories, openingHours, imageUrls, description,
latitude, longitude, reviews (up to 5 per place), url, placeId
"""

import json
import os
import asyncio
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import AsyncSessionLocal, engine, Base
from backend.model.models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store

APIFY_EXPORT_PATH = "data/apify_export.json"


def _extract_city(place: dict) -> str:
    """
    Extract city from Apify address fields.
    Apify returns address as a dict with city, state, country fields.
    """
    addr = place.get("address", {})
    if isinstance(addr, dict):
        return (
            addr.get("city")
            or addr.get("neighborhood")
            or addr.get("state")
            or "Pakistan"
        ).strip()
    # Sometimes address is a plain string
    return "Pakistan"


def _extract_cuisine(place: dict) -> str:
    """Extract primary cuisine from Apify categories list."""
    categories = place.get("categories", [])
    if not categories:
        return "Restaurant"
    # Filter out generic terms
    skip = {"restaurant", "food", "establishment", "point_of_interest"}
    for cat in categories:
        if isinstance(cat, str) and cat.lower() not in skip:
            return cat.title()
    return categories[0].title() if categories else "Restaurant"


def _extract_price_level(place: dict) -> str | None:
    """Convert Apify priceLevel integer (1-4) to $ symbols."""
    level = place.get("priceLevel") or place.get("price")
    if isinstance(level, int):
        return "$" * max(1, min(4, level))
    if isinstance(level, str) and level.startswith("$"):
        return level
    return None


def _extract_hours(place: dict) -> str | None:
    """Serialize opening hours to JSON string."""
    hours = place.get("openingHours") or place.get("popularTimesHistogram")
    if not hours:
        return None
    if isinstance(hours, (dict, list)):
        return json.dumps(hours)
    return str(hours)


def _extract_photos(place: dict) -> str | None:
    """Serialize photo URLs to JSON string."""
    images = place.get("imageUrls") or place.get("images") or []
    if not images:
        return None
    # Keep max 10 photos to avoid huge DB rows
    return json.dumps(images[:10])


def _extract_tags(place: dict) -> str | None:
    """Build a tags list from Apify amenities and highlights."""
    tags = []
    for key in ("amenities", "highlights", "serviceOptions"):
        val = place.get(key)
        if isinstance(val, dict):
            tags.extend(k for k, v in val.items() if v is True)
        elif isinstance(val, list):
            tags.extend(val)
    return json.dumps(tags) if tags else None


def normalize_apify_place(place: dict) -> tuple[dict, list[dict]]:
    """
    Convert one raw Apify Google Maps place into:
      - restaurant_data dict (matches Restaurant model fields)
      - reviews list of dicts (matches Review model fields)
    """
    city = _extract_city(place)

    restaurant_data = {
        "name":          (place.get("title") or place.get("name") or "").strip(),
        "cuisine":       _extract_cuisine(place),
        "city":          city,
        "address":       place.get("address") if isinstance(place.get("address"), str)
                         else place.get("street") or place.get("addressStreet"),
        "phone":         place.get("phone") or place.get("phoneUnformatted"),
        "website":       place.get("website"),
        "menu_url":      place.get("menuUrl"),
        "latitude":      place.get("location", {}).get("lat") if isinstance(place.get("location"), dict)
                         else place.get("lat"),
        "longitude":     place.get("location", {}).get("lng") if isinstance(place.get("location"), dict)
                         else place.get("lng"),
        "rating":        place.get("totalScore") or place.get("rating"),
        "review_count":  place.get("reviewsCount") or place.get("reviewCount"),
        "price_level":   _extract_price_level(place),
        "description":   place.get("description"),
        "opening_hours": _extract_hours(place),
        "photos":        _extract_photos(place),
        "tags":          _extract_tags(place),
        "source":        "apify",
        "external_id":   place.get("placeId") or place.get("id"),
    }

    # Extract up to 5 reviews per place
    raw_reviews = place.get("reviews") or []
    reviews = []
    for r in raw_reviews[:5]:
        if not isinstance(r, dict):
            continue
        reviews.append({
            "reviewer_name":  r.get("name") or r.get("reviewerName"),
            "rating":         r.get("stars") or r.get("rating"),
            "text":           r.get("text") or r.get("reviewText"),
            "published_date": r.get("publishedAtDate") or r.get("date"),
            "source":         "google",
        })

    return restaurant_data, reviews


async def load_apify_export(filepath: str = APIFY_EXPORT_PATH) -> dict:
    """
    Read the Apify JSON export and load all records into:
      - PostgreSQL (Restaurant + Review rows)
      - ChromaDB (embeddings for new restaurants)
      - BM25 index (rebuilt after all inserts)

    Returns a summary dict.
    """
    if not os.path.exists(filepath):
        return {
            "error": f"File not found: {filepath}. "
                     "Export your Apify dataset as JSON and save it there."
        }

    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        return {"error": "Expected a JSON array at the top level of the Apify export."}

    print(f"[apify] Loaded {len(raw_data)} places from {filepath}")

    inserted = 0
    skipped = 0
    reviews_inserted = 0
    embedded = 0
    errors = []

    async with AsyncSessionLocal() as db:
        newly_inserted: list[Restaurant] = []

        for place in raw_data:
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

            # Deduplicate by name + city
            result = await db.execute(
                select(Restaurant).where(
                    Restaurant.name == name,
                    Restaurant.city == city
                )
            )
            existing = result.scalars().first()

            if existing:
                skipped += 1
                continue

            record = Restaurant(**restaurant_data, is_embedded=False)
            db.add(record)
            newly_inserted.append((record, reviews))
            inserted += 1

        await db.commit()

        # Refresh to get auto-assigned IDs, then insert reviews + embed
        for record, reviews in newly_inserted:
            await db.refresh(record)

            # Insert reviews
            for rv in reviews:
                db.add(Review(restaurant_id=record.id, **rv))
                reviews_inserted += 1

            # Embed into ChromaDB
            try:
                vector_store.upsert_restaurant(
                    restaurant_id=record.id,
                    name=record.name,
                    cuisine=record.cuisine,
                    city=record.city
                )
                record.is_embedded = True
                embedded += 1
            except Exception as e:
                errors.append(f"embed {record.name}: {e}")

        await db.commit()

        # Rebuild BM25 from full table
        all_result = await db.execute(select(Restaurant))
        all_restaurants = all_result.scalars().all()
        bm25_store.build_index([
            {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
            for r in all_restaurants
        ])

    print(f"[apify] Done: {inserted} inserted, {skipped} skipped, "
          f"{reviews_inserted} reviews, {embedded} embedded")

    # Auto-trigger review summarisation for all newly inserted restaurants
    # that have reviews — this populates the ChromaDB review_summaries collection
    summarised = 0
    if newly_inserted:
        from backend.data_loader.review_summariser import summarise_reviews
        from backend.retrieving.vector_store import upsert_review_summary

        async with AsyncSessionLocal() as db:
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

    print(f"[apify] Review summaries generated: {summarised}")

    return {
        "inserted":           inserted,
        "skipped":            skipped,
        "reviews_inserted":   reviews_inserted,
        "embedded":           embedded,
        "reviews_summarised": summarised,
        "errors":             errors[:10],
    }


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        # Ensure tables exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        result = await load_apify_export()
        print(json.dumps(result, indent=2))

    asyncio.run(main())
