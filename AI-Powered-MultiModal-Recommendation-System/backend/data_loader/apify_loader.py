"""
backend/apify_loader.py
------------------------
Loads Apify Google Maps Scraper data into PostgreSQL + ChromaDB + BM25.

"""

import json
import os

from sqlalchemy import select

from backend.core.database import AsyncSessionLocal, engine, Base
from backend.models.db_models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store

APIFY_EXPORT_PATH = "data/apify_export.json"


# ── Field extraction helpers ────────────────────────────────────────────────

def _extract_price_level(price_raw) -> str | None:
    """Convert Apify price field to $ symbols. Handles int, string, or None."""
    if price_raw is None:
        return None
    if isinstance(price_raw, int):
        return "$" * max(1, min(4, price_raw))
    if isinstance(price_raw, str):
        if price_raw.startswith("$"):
            return price_raw
        # Apify sometimes returns "Rs 500–1000" style — keep as-is, capped
        return price_raw[:20]
    return None


def _extract_all_cuisines(place: dict) -> list[str]:
    """
    Extract ALL cuisine categories
    (e.g. "Steak house", "Chinese restaurant", "Seafood restaurant").
    """
    categories = place.get("categories") or []
    if not categories:
        primary = place.get("categoryName")
        return [primary] if primary else []

    cleaned = []
    seen = set()
    for c in categories:
        if not isinstance(c, str):
            continue
        c_clean = c.strip()
        key = c_clean.lower()
        if c_clean and key not in seen:
            seen.add(key)
            cleaned.append(c_clean)
    return cleaned[:6]   # cap — a restaurant tagged with 10+ categories is noise


def _extract_tags_from_additional_info(place: dict) -> list[str]:
    """
    Flatten additionalInfo into a list of feature tags.
    additionalInfo structure:
      {"Atmosphere": [{"Cozy": true}, {"Romantic": true}], "Highlights": [...], ...}

    """
    additional = place.get("additionalInfo") or {}
    if not isinstance(additional, dict):
        return []

    priority_categories = [
        "Atmosphere", "Highlights", "Offerings",
        "Dining options", "Crowd", "Popular for", "Service options"
    ]

    tags = []
    for category in priority_categories:
        items = additional.get(category, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                for feature, is_true in item.items():
                    if is_true and feature not in tags:
                        tags.append(feature)

    return tags[:20]   # cap for token safety downstream


def _extract_top_dishes(place: dict) -> list[str]:
    """
    Extract frequently-mentioned dish names from reviewsTags.
  
    """
    review_tags = place.get("reviewsTags") or []
    if not review_tags:
        return []

    sorted_tags = sorted(
        [t for t in review_tags if isinstance(t, dict) and t.get("count", 0) >= 2],
        key=lambda t: t.get("count", 0),
        reverse=True
    )
    return [t["title"] for t in sorted_tags[:8] if t.get("title")]


def _extract_hours(place: dict) -> str | None:
    """Serialize openingHours list to JSON string."""
    hours = place.get("openingHours")
    if not hours:
        return None
    return json.dumps(hours)


def _extract_photos(place: dict) -> str | None:
    """Combine main imageUrl + imageUrls list, capped."""
    urls = []
    main_image = place.get("imageUrl")
    if main_image:
        urls.append(main_image)
    extra = place.get("imageUrls") or []
    if isinstance(extra, list):
        urls.extend(extra[:9])
    return json.dumps(urls[:10]) if urls else None


def _extract_reviews_distribution(place: dict) -> str | None:
    """Serialize the full star-rating breakdown — used for fake review detection."""
    dist = place.get("reviewsDistribution")
    if not dist or not isinstance(dist, dict):
        return None
    return json.dumps({
        "one_star":   dist.get("oneStar", 0),
        "two_star":   dist.get("twoStar", 0),
        "three_star": dist.get("threeStar", 0),
        "four_star":  dist.get("fourStar", 0),
        "five_star":  dist.get("fiveStar", 0),
    })


def _build_description(place: dict, top_dishes: list[str], all_cuisines: list[str]) -> str | None:
    """
    Apify's 'description' field is almost always null for Google Maps places.
    Build a useful description from categories + top-mentioned dishes instead.
    """
    raw_desc = place.get("description")
    if raw_desc and len(raw_desc.strip()) > 10:
        return raw_desc.strip()[:400]

    parts = []
    if all_cuisines:
        parts.append(f"Serves {', '.join(all_cuisines[:3])}.")
    if top_dishes:
        parts.append(f"Known for: {', '.join(top_dishes[:5])}.")

    return " ".join(parts) if parts else None


# ── Main normalisation function ─────────────────────────────────────────────

def normalize_apify_place(place: dict) -> tuple[dict, list[dict]]:
    """
    Convert one raw Apify Google Maps place into:
      - restaurant_data dict (matches Restaurant model fields)
      - reviews list of dicts (matches Review model fields)

    """
    name = (place.get("title") or "").strip()
    city = (place.get("city") or "").strip()
    area = (place.get("neighborhood") or "").strip() or None

    all_cuisines = _extract_all_cuisines(place)
    primary_cuisine = all_cuisines[0] if all_cuisines else (place.get("categoryName") or "Restaurant")

    top_dishes = _extract_top_dishes(place)
    tags = _extract_tags_from_additional_info(place)
    combined_tags = list(dict.fromkeys(tags + top_dishes))   # merge, dedupe, keep order

    location = place.get("location") or {}

    restaurant_data = {
        "name":          name,
        "cuisine":       primary_cuisine,
        "all_cuisines":  json.dumps(all_cuisines) if all_cuisines else None,
        "city":          city,
        "area":          area,
        "postal_code":   place.get("postalCode"),
        "address":       place.get("address"),
        "phone":         place.get("phone") or place.get("phoneUnformatted"),
        "website":       place.get("website"),
        "menu_url":      place.get("menu"),              # NOTE: field is "menu" not "menuUrl"
        "latitude":      location.get("lat"),
        "longitude":     location.get("lng"),
        "rating":        place.get("totalScore"),
        "review_count":  place.get("reviewsCount"),
        "price_level":   _extract_price_level(place.get("price")),
        "description":   _build_description(place, top_dishes, all_cuisines),
        "opening_hours": _extract_hours(place),
        "photos":        _extract_photos(place),
        "tags":          json.dumps(combined_tags) if combined_tags else None,
        "reviews_distribution": _extract_reviews_distribution(place),
        "source":        "apify",
        "external_id":   place.get("placeId"),
    }

    # Reviews — only present if actor input requested them (maxReviews > 0)
    raw_reviews = place.get("reviews") or []
    reviews = []
    for r in raw_reviews[:10]:
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


# ── Load pipeline ────────────────────────────────────────────────────────────

async def load_apify_export(filepath: str = APIFY_EXPORT_PATH) -> dict:
    """
    Read the Apify JSON export and load all records into:
      PostgreSQL (Restaurant + Review rows) → ChromaDB → BM25 index.

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

    return await _load_places(raw_data)


async def _load_places(raw_data: list[dict]) -> dict:
    """Shared loading logic — used by both manual file load and automated API load."""
    print(f"[apify] Processing {len(raw_data)} places")

    inserted, skipped, reviews_inserted, embedded = 0, 0, 0, 0
    skipped_wrong_country = 0
    errors = []

    async with AsyncSessionLocal() as db:
        newly_inserted: list[tuple[Restaurant, list[dict]]] = []

        for place in raw_data:
            if place.get("permanentlyClosed") or place.get("temporarilyClosed"):
                skipped += 1
                continue

            country_code = place.get("countryCode")
            if country_code and country_code != "PK":
                skipped_wrong_country += 1
                continue

            try:
                restaurant_data, reviews = normalize_apify_place(place)
                name = restaurant_data.get("name", "").strip()
                external_id = restaurant_data.get("external_id")

                if not name:
                    skipped += 1
                    continue

                if external_id:
                    result = await db.execute(
                        select(Restaurant).where(Restaurant.external_id == external_id)
                    )
                else:
                    result = await db.execute(
                        select(Restaurant).where(
                            Restaurant.name == name,
                            Restaurant.city == restaurant_data.get("city", "")
                        )
                    )

                if result.scalars().first():
                    skipped += 1
                    continue

                record = Restaurant(**restaurant_data, is_embedded=False)
                db.add(record)
                newly_inserted.append((record, reviews))
                inserted += 1

            except Exception as e:
                errors.append(f"place error ({place.get('title', '?')}): {e}")
                continue

        await db.commit()

        # Refresh, insert reviews, embed
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

        # Rebuild BM25 from full table
        all_result = await db.execute(select(Restaurant))
        all_restaurants = all_result.scalars().all()
        bm25_store.build_index([
            {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
            for r in all_restaurants
        ])

        # Auto-summarise reviews for newly inserted restaurants that have them
        summarised = 0
        if newly_inserted:
            from backend.data_loader.review_summariser import summarise_reviews
            from backend.retrieving.vector_store import upsert_review_summary

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

    print(f"[apify] Done: {inserted} inserted, {skipped} skipped, "
          f"{skipped_wrong_country} wrong-country rejected, "
          f"{reviews_inserted} reviews, {embedded} embedded, {summarised} summarised")

    return {
        "inserted":               inserted,
        "skipped":                skipped,
        "skipped_wrong_country":  skipped_wrong_country,
        "reviews_inserted":       reviews_inserted,
        "embedded":               embedded,
        "reviews_summarised":     summarised,
        "errors":                 errors[:10],
    }


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        result = await load_apify_export()
        print(json.dumps(result, indent=2))

    asyncio.run(main())
