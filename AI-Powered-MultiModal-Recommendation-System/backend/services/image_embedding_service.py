"""
backend/services/image_embedding_service.py
----------------------------------------------
Batch CLIP-embeds photos restaurants ALREADY have into the
restaurant_images ChromaDB collection — the free, local counterpart
to enrichment_services.py's Google-Places-based image embedding.

WHY THIS IS SEPARATE FROM enrichment_services.py
--------------------------------------------------
Apify's `photos` column already contains direct, hotlinkable image
URLs (no `photo_reference` token, no Google Places API call needed to
resolve one) — most restaurants in this dataset got their photos this
way. enrichment_services.py's image embedding only ever runs as a
side effect of its Google Places Details call, which costs
~$0.017/restaurant and requires a billing-enabled API key. This job
re-uses the exact same CLIP model and the exact same
upsert_restaurant_image() / ChromaDB collection — the only difference
is where the URL came from and that this path costs nothing beyond
bandwidth and local CPU/GPU time.

Safe to re-run repeatedly: a restaurant is skipped once every URL in
its `photos` list already has a matching image_index in ChromaDB, so
this can be called again after every new Apify sync without
re-downloading or re-embedding anything already done.
"""

import asyncio
import json

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import Restaurant
from backend.retrieving.vector_store import (
    upsert_restaurant_image,
    get_restaurant_images,
    image_collection_count,
)

RATE_LIMIT_RPS = 5              # polite pace against whatever CDN these URLs point at
MAX_PHOTOS_PER_RESTAURANT = 10  # matches enrichment_services.py's MAX_PHOTOS for consistency


async def _download_image(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[image_embed] Download failed for {url}: {e}")
        return None


async def embed_restaurant_photos(db: AsyncSession, limit: int | None = None) -> dict:
    """
    For every restaurant with a non-empty `photos` column, download
    and CLIP-embed each URL not already present in the
    restaurant_images collection.

    Args:
        db:    async DB session
        limit: max restaurants to process this run. None = all.
               Start small (10-20) to confirm CLIP loads and embeds
               correctly before running against the full dataset —
               the first call in a process will download the CLIP
               model (~600MB) and load it onto CPU/GPU, which is the
               slowest part of a first run.

    Returns a summary dict with per-field counts and up to 20 errors.
    """
    result = await db.execute(
        select(Restaurant).where(Restaurant.photos.isnot(None))
    )
    all_restaurants = [r for r in result.scalars().all() if r.photos]

    if limit:
        all_restaurants = all_restaurants[:limit]

    processed = 0
    images_embedded = 0
    already_done = 0
    errors: list[str] = []

    async with httpx.AsyncClient() as client:
        for i, restaurant in enumerate(all_restaurants):
            try:
                urls = json.loads(restaurant.photos)
            except (json.JSONDecodeError, TypeError):
                errors.append(f"{restaurant.name}: photos column is not valid JSON")
                continue

            if not urls:
                continue

            urls = urls[:MAX_PHOTOS_PER_RESTAURANT]
            existing_indices = {img["image_index"] for img in get_restaurant_images(restaurant.id)}

            if len(existing_indices) >= len(urls):
                already_done += 1
                continue

            processed += 1
            for idx, url in enumerate(urls):
                if idx in existing_indices:
                    continue

                image_bytes = await _download_image(client, url)
                if not image_bytes:
                    continue

                try:
                    upsert_restaurant_image(
                        restaurant_id=restaurant.id,
                        image_index=idx,
                        image_url=url,
                        image_bytes=image_bytes,
                        name=restaurant.name,
                        cuisine=restaurant.cuisine,
                        city=restaurant.city,
                    )
                    images_embedded += 1
                except Exception as e:
                    errors.append(f"{restaurant.name} img {idx}: {e}")

            if (i + 1) % RATE_LIMIT_RPS == 0:
                await asyncio.sleep(1.0)
            if (i + 1) % 50 == 0:
                print(f"[image_embed] {i + 1}/{len(all_restaurants)} restaurants processed...")

    return {
        "message":                     "Image embedding complete",
        "restaurants_with_photos":     len(all_restaurants),
        "restaurants_processed":       processed,
        "restaurants_already_complete": already_done,
        "images_embedded":             images_embedded,
        "image_vectors_total":         image_collection_count(),
        "errors":                      errors[:20],
    }