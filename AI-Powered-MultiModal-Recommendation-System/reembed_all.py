"""
reembed_all.py
---------------
Re-embeds ALL restaurants in PostgreSQL using the new rich embedding text.

Run this ONCE after updating embedder.py and data_enrichment.py to fix
the existing sparse embeddings in ChromaDB.

What changes:
  BEFORE: "Nando's is a Chicken restaurant in Islamabad."
  AFTER:  "Nando's is a peri-peri grilled chicken restaurant in Islamabad,
            Agha Khan Road area. Known for: peri peri, grilled chicken,
            spicy, portuguese. Phone contact available."

Also fixes cuisine normalisation in ChromaDB metadata:
  BEFORE: cuisine = "Restaurant"
  AFTER:  cuisine = "Biryani" (or whatever was inferred from the name)

Run from project root:
    python reembed_all.py
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()


def ok(msg):   print(f"  \033[92m✅ {msg}\033[0m")
def fail(msg): print(f"  \033[91m❌ {msg}\033[0m")
def info(msg): print(f"  \033[94mℹ  {msg}\033[0m")


async def reembed_all():
    print("\n\033[1m🔄 Re-embedding all restaurants with rich text\033[0m\n")

    from backend.db.database import AsyncSessionLocal, engine, Base
    from backend.model.models import Restaurant
    from backend.retrieving.vector_store import upsert_restaurant, collection_count
    from backend.retrieving.data_enrichment import (
        normalise_cuisine, build_rich_embedding_text
    )
    from sqlalchemy import select

    # Load all restaurants
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Restaurant))
        restaurants = result.scalars().all()

    total = len(restaurants)
    print(f"Found {total} restaurants to re-embed\n")

    # Show before/after examples first
    print("BEFORE → AFTER examples:")
    print("─" * 60)
    from backend.retrieving.data_enrichment import build_rich_embedding_text, normalise_cuisine
    for r in restaurants[:5]:
        old = f"{r.name} is a {r.cuisine} restaurant located in {r.city}."
        new = build_rich_embedding_text(
            name=r.name, cuisine=r.cuisine, city=r.city,
            address=r.address, description=r.description,
            tags=r.tags, opening_hours=r.opening_hours,
            phone=r.phone, website=r.website, rating=r.rating
        )
        clean_c = normalise_cuisine(r.cuisine, r.name)
        print(f"\n  Name: {r.name}")
        print(f"  Cuisine: '{r.cuisine}' → '{clean_c}'")
        print(f"  OLD: {old[:80]}")
        print(f"  NEW: {new[:120]}")
    print("\n" + "─" * 60 + "\n")

    input("Press Enter to proceed with re-embedding all restaurants...")

    # Re-embed all
    done = 0
    errors = []
    t0 = time.time()

    for i, r in enumerate(restaurants):
        try:
            upsert_restaurant(
                restaurant_id=r.id,
                name=r.name,
                cuisine=r.cuisine,
                city=r.city,
                address=r.address,
                description=r.description,
                tags=r.tags,
                opening_hours=r.opening_hours,
                phone=r.phone,
                website=r.website,
                rating=r.rating,
                latitude=r.latitude,
                longitude=r.longitude,
            )
            done += 1

            if (i + 1) % 50 == 0:
                elapsed = round(time.time() - t0, 1)
                rate = round(done / elapsed, 1)
                print(f"  {i+1}/{total} done ({rate}/sec)...")

        except Exception as e:
            errors.append(f"{r.name}: {e}")

    elapsed = round(time.time() - t0, 1)

    print()
    ok(f"Re-embedded {done}/{total} restaurants in {elapsed}s")
    if errors:
        for e in errors[:5]:
            fail(f"Error: {e}")

    # Rebuild BM25 with normalised cuisine
    print("\nRebuilding BM25 index...")
    from backend.retrieving.bm25_store import build_index
    from backend.retrieving.data_enrichment import normalise_cuisine

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Restaurant))
        all_recs = result.scalars().all()

    build_index([
        {
            "id":      r.id,
            "name":    r.name,
            "cuisine": normalise_cuisine(r.cuisine, r.name),  # use normalised
            "city":    r.city,
            "address": r.address or ""
        }
        for r in all_recs
    ])
    ok(f"BM25 rebuilt from {len(all_recs)} restaurants")

    # Verify with test searches
    print("\nVerification searches:")
    print("─" * 40)

    from backend.retrieving.retrieval import hybrid_search

    test_queries = [
        "biryani Islamabad",
        "coffee cafe Islamabad",
        "Chinese food",
        "burger fast food",
        "Pakistani desi food",
    ]

    for q in test_queries:
        results = hybrid_search(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for r in results:
            print(
                f"    [{r['rrf_score']:.4f}] {r['name']} "
                f"— {r['cuisine']} ({r['city']})"
            )

    print(f"\n\033[92m✅ Done. Total vectors in ChromaDB: {collection_count()}\033[0m")
    print("\nNow run the server:")
    print("  bash run.sh")


if __name__ == "__main__":
    asyncio.run(reembed_all())
