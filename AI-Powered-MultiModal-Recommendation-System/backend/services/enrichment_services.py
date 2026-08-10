"""
backend/services/enrichment_service.py
----------------------------------------
Enriches existing restaurants with reviews + photos from Google Places API,
then embeds everything into ChromaDB — each item tied to its restaurant_id.

WHAT THIS DOES PER RESTAURANT
-------------------------------
1. Google Places Details call (one call, free tier)
   → reviews (up to 5)
   → photo references (up to 10)

2. Reviews → PostgreSQL reviews table
   → LLM generates cautious summary
   → Summary embedded → ChromaDB restaurant_reviews collection
   → Tied to restaurant by restaurant_id

3. Photos → download each image URL
   → CLIP embeds each image (512-dim)
   → Each image stored in ChromaDB restaurant_images collection
   → Tied to restaurant by restaurant_id + image_index
   → URL also written to restaurants.photos for display

WHY SEPARATE EMBEDDING PER IMAGE (not one per restaurant)
-----------------------------------------------------------
A restaurant might have a photo of the menu, one of the rooftop,
one of the food plating. Embedding all into one vector loses that.
Separate embeddings mean "rooftop view" finds the rooftop photo,
not just restaurants that happen to have a rooftop photo somewhere.

COST
-----
Google Places Details: ~$0.017 per restaurant
$200 free/month → 11,700 calls free → 600 restaurants ≈ $10.20

Required env var: GOOGLE_PLACES_API_KEY
"""

import asyncio
import io
import json
import os

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import Restaurant, Review
from backend.retrieving.vector_store import (
    upsert_review_summary,
    get_review_summary,
    upsert_restaurant_image,
    image_collection_count,
)

PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_SEARCH_URL  = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PHOTO_URL_BASE     = "https://maps.googleapis.com/maps/api/place/photo"
DETAIL_FIELDS      = "reviews,photos"
RATE_LIMIT_RPS     = 10     # requests per second — well within Google limits
MAX_PHOTOS         = 10     # images to embed per restaurant


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GOOGLE_PLACES_API_KEY not set. "
            "Get a free key: console.cloud.google.com → APIs → Places API"
        )
    return key


# ── Google Places helpers ───────────────────────────────────────────────────

def _build_photo_url(photo_reference: str, api_key: str, max_width: int = 800) -> str:
    return (
        f"{PHOTO_URL_BASE}"
        f"?maxwidth={max_width}"
        f"&photo_reference={photo_reference}"
        f"&key={api_key}"
    )


async def _find_place_id(
    client: httpx.AsyncClient,
    restaurant: Restaurant,
    api_key: str,
) -> str | None:
    """Text Search to find placeId for OSM restaurants that have none."""
    try:
        resp = await client.get(
            PLACES_SEARCH_URL,
            params={"query": f"{restaurant.name} {restaurant.city} Pakistan", "key": api_key},
            timeout=10.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0].get("place_id") if results else None
    except Exception as e:
        print(f"[enrich] Text search failed for {restaurant.name}: {e}")
        return None


async def _fetch_place_details(
    client: httpx.AsyncClient,
    place_id: str,
    api_key: str,
) -> dict:
    """Fetch reviews + photo references for one placeId."""
    try:
        resp = await client.get(
            PLACES_DETAILS_URL,
            params={"place_id": place_id, "fields": DETAIL_FIELDS, "key": api_key},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json().get("result", {})
    except Exception as e:
        print(f"[enrich] Details API error for {place_id}: {e}")
        return {}


async def _download_image(
    client: httpx.AsyncClient,
    url: str,
) -> bytes | None:
    """Download image bytes from a URL. Returns None on failure."""
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[enrich] Image download failed: {e}")
        return None


# ── Per-restaurant enrichment ───────────────────────────────────────────────

async def _enrich_one_restaurant(
    db: AsyncSession,
    client: httpx.AsyncClient,
    restaurant: Restaurant,
    api_key: str,
    embed_images: bool,
) -> dict:
    """
    Full enrichment pipeline for one restaurant:
      1. Get placeId (or use stored one)
      2. Fetch reviews + photo references
      3. Insert reviews → PostgreSQL
      4. Generate review summary → embed → ChromaDB restaurant_reviews
      5. Download each photo → CLIP embed → ChromaDB restaurant_images
      6. Store photo URLs in restaurants.photos

    Returns a summary dict for reporting.
    """
    result = {
        "name":           restaurant.name,
        "reviews_added":  0,
        "photos_stored":  0,
        "images_embedded": 0,
        "summarised":     False,
        "error":          None,
    }

    # Step 1: ensure we have a placeId
    place_id = restaurant.external_id
    if not place_id:
        place_id = await _find_place_id(client, restaurant, api_key)
        if place_id:
            restaurant.external_id = place_id
            await db.commit()

    if not place_id:
        result["error"] = "No placeId found"
        return result

    # Step 2: fetch details from Google
    details = await _fetch_place_details(client, place_id, api_key)
    if not details:
        result["error"] = "Empty response from Places API"
        return result

    # Step 3: insert reviews into PostgreSQL
    raw_reviews = details.get("reviews", [])
    review_dicts = []
    for r in raw_reviews:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        review_obj = Review(
            restaurant_id=restaurant.id,
            reviewer_name=r.get("author_name"),
            rating=r.get("rating"),
            text=text[:1000],
            published_date=r.get("relative_time_description", ""),
            source="google",
        )
        db.add(review_obj)
        review_dicts.append({
            "reviewer_name":  r.get("author_name"),
            "rating":         r.get("rating"),
            "text":           text[:1000],
            "published_date": r.get("relative_time_description", ""),
            "source":         "google",
        })
        result["reviews_added"] += 1

    await db.commit()

    # Step 4: generate review summary → embed → ChromaDB restaurant_reviews
    # Each summary is keyed by restaurant_id so it's always tied to the right place
    if review_dicts and not get_review_summary(restaurant.id):
        try:
            from backend.data_loader.review_summariser import summarise_reviews
            summary_data = summarise_reviews(
                restaurant_name=restaurant.name,
                cuisine=restaurant.cuisine,
                reviews=review_dicts,
            )
            upsert_review_summary(
                restaurant_id=restaurant.id,
                restaurant_name=restaurant.name,
                cuisine=restaurant.cuisine,
                city=restaurant.city,
                **summary_data,
            )
            result["summarised"] = True
        except Exception as e:
            print(f"[enrich] Summarise failed for {restaurant.name}: {e}")

    # Step 5: photos — download + CLIP embed + store in ChromaDB restaurant_images
    raw_photos  = details.get("photos", [])[:MAX_PHOTOS]
    photo_urls  = []

    for idx, photo in enumerate(raw_photos):
        ref = photo.get("photo_reference")
        if not ref:
            continue

        url = _build_photo_url(ref, api_key)
        photo_urls.append(url)

        if embed_images:
            image_bytes = await _download_image(client, url)
            if image_bytes:
                try:
                    # Each image stored as img_{restaurant_id}_{idx}
                    # metadata.restaurant_id ties it back to the restaurant
                    upsert_restaurant_image(
                        restaurant_id=restaurant.id,
                        image_index=idx,
                        image_url=url,
                        image_bytes=image_bytes,
                        name=restaurant.name,
                        cuisine=restaurant.cuisine,
                        city=restaurant.city,
                    )
                    result["images_embedded"] += 1
                except Exception as e:
                    print(f"[enrich] CLIP embed failed {restaurant.name} img {idx}: {e}")

    # Step 6: write photo URLs to restaurants.photos for display
    if photo_urls:
        restaurant.photos = json.dumps(photo_urls)
        await db.commit()
        result["photos_stored"] = len(photo_urls)

    return result


# ── Batch enrichment entry point ────────────────────────────────────────────

async def enrich_restaurants(
    db: AsyncSession,
    only_without_reviews: bool = True,
    limit: int | None = None,
    embed_images: bool = True,
) -> dict:
    """
    Batch enrichment: fetch reviews + photos from Google Places,
    embed both into ChromaDB, each tied to their restaurant_id.

    Args:
        db:                   async DB session
        only_without_reviews: skip restaurants already in reviews table
        limit:                max restaurants to process. Start with 10 to test.
        embed_images:         set False to skip CLIP embedding (faster, saves memory)

    Returns:
        Full summary with per-field counts and any errors.
    """
    api_key = _get_api_key()

    result = await db.execute(select(Restaurant))
    all_restaurants = result.scalars().all()

    if only_without_reviews:
        have_reviews = await db.execute(
            select(Review.restaurant_id).distinct()
        )
        enriched_ids = {r for r in have_reviews.scalars().all()}
        to_process = [r for r in all_restaurants if r.id not in enriched_ids]
    else:
        to_process = list(all_restaurants)

    if limit:
        to_process = to_process[:limit]

    print(f"[enrich] Processing {len(to_process)} restaurants "
          f"(images={'yes' if embed_images else 'no'})")

    total_reviews   = 0
    total_photos    = 0
    total_images_embedded = 0
    total_summarised = 0
    errors          = []

    async with httpx.AsyncClient() as client:
        for i, restaurant in enumerate(to_process):
            try:
                r = await _enrich_one_restaurant(
                    db, client, restaurant, api_key, embed_images
                )
                total_reviews          += r["reviews_added"]
                total_photos           += r["photos_stored"]
                total_images_embedded  += r["images_embedded"]
                total_summarised       += int(r["summarised"])

                if r["error"]:
                    errors.append(f"{restaurant.name}: {r['error']}")

            except Exception as e:
                errors.append(f"{restaurant.name}: {e}")

            # Rate limit: 10 requests/second
            if (i + 1) % RATE_LIMIT_RPS == 0:
                await asyncio.sleep(1.0)

            if (i + 1) % 50 == 0:
                print(f"[enrich] {i+1}/{len(to_process)} done...")

    return {
        "message":              "Enrichment complete",
        "processed":            len(to_process),
        "reviews_added":        total_reviews,
        "photos_stored":        total_photos,
        "images_embedded":      total_images_embedded,
        "review_summaries":     total_summarised,
        "image_vectors_total":  image_collection_count(),
        "errors":               errors[:20],
    }
