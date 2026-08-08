"""
backend/services/ingestion_service.py
---------------------------------------
All data ingestion logic: insert restaurants, embed, build BM25.

Used by routers/ingestion.py.
The actual Apify API calls live in apify_automation.py.
The apify data normalisation lives in apify_loader.py.
This service handles the shared pipeline that both use.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store


async def sync_restaurants(
    db: AsyncSession,
    restaurants: list[dict],
) -> dict:
    """
    Insert a list of restaurant dicts into PostgreSQL + ChromaDB + BM25.

    Deduplicates by (name, city) — safe to call repeatedly.
    Only newly inserted restaurants are embedded.

    Args:
        db:          async DB session
        restaurants: list of dicts matching RestaurantInput schema

    Returns:
        {"inserted": [...], "skipped": [...], "embedded_count": int}
    """
    saved, skipped, newly_inserted = [], [], []

    for item in restaurants:
        name = (item.get("name") or "").strip()
        city = (item.get("city") or "").strip()
        if not name:
            skipped.append(name)
            continue

        result = await db.execute(
            select(Restaurant).where(
                Restaurant.name == name,
                Restaurant.city == city,
            )
        )
        if result.scalars().first():
            skipped.append(name)
            continue

        record = Restaurant(**{
            k: v for k, v in item.items()
            if hasattr(Restaurant, k)
        }, is_embedded=False)
        db.add(record)
        newly_inserted.append(record)
        saved.append(name)

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
            print(f"[ingestion] embed error — {record.name}: {e}")

    await db.commit()

    # Rebuild BM25 from full table after every sync
    if newly_inserted:
        all_result = await db.execute(select(Restaurant))
        all_recs   = all_result.scalars().all()
        bm25_store.build_index([
            {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
            for r in all_recs
        ])

    return {
        "inserted":      saved,
        "skipped":       skipped,
        "embedded_count": embedded_count,
    }


async def summarise_new_restaurants(
    db: AsyncSession,
    newly_inserted: list[tuple],
) -> int:
    """
    Generate cautious review summaries for a batch of newly inserted restaurants.
    Called automatically after Apify loads — skips restaurants with no reviews.

    Args:
        newly_inserted: list of (Restaurant record, reviews list) tuples

    Returns:
        Number of summaries successfully generated.
    """
    from backend.data_loader.review_summariser import summarise_reviews
    from backend.retrieving.vector_store import upsert_review_summary

    count = 0
    for record, reviews in newly_inserted:
        if not reviews:
            continue
        try:
            summary_data = summarise_reviews(
                restaurant_name=record.name,
                cuisine=record.cuisine,
                reviews=reviews,
            )
            upsert_review_summary(
                restaurant_id=record.id,
                restaurant_name=record.name,
                cuisine=record.cuisine,
                city=record.city,
                **summary_data,
            )
            count += 1
        except Exception as e:
            print(f"[ingestion] summarise error — {record.name}: {e}")

    return count


async def rebuild_bm25(db: AsyncSession) -> int:
    """Rebuild the BM25 index from the full restaurants table. Returns record count."""
    result = await db.execute(select(Restaurant))
    all_recs = result.scalars().all()
    bm25_store.build_index([
        {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
        for r in all_recs
    ])
    return len(all_recs)