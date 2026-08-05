"""
test_pipeline.py
-----------------
End-to-end pipeline test. Run from the project root:

    python test_pipeline.py

Tests (in order):
  1. PostgreSQL connection + table creation
  2. OSM fetch for ONE city (Islamabad, small bbox)
  3. Insert fetched restaurants into PostgreSQL
  4. Generate embeddings
  5. Store in ChromaDB
  6. Build BM25 index
  7. Run a test search across all three signals
  8. Print summary

No pytest needed — plain Python, reads your .env automatically.
"""

import asyncio
import sys
import os
import time

from backend.db.service import get_all_restaurants
# ── Make sure project root is on path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── Colour helpers (works on Windows too if colorama installed) ─────────────
def ok(msg):   print(f"  \033[92m✅ {msg}\033[0m")
def fail(msg): print(f"  \033[91m❌ {msg}\033[0m")
def info(msg): print(f"  \033[94mℹ  {msg}\033[0m")
def head(msg): print(f"\n\033[1m{'─'*55}\n{msg}\n{'─'*55}\033[0m")


# ══════════════════════════════════════════════════════════════
# STEP 1 — PostgreSQL connection + table creation
# ══════════════════════════════════════════════════════════════

async def test_database():
    head("STEP 1 — PostgreSQL connection")

    try:
        from backend.db.database import engine, Base
        from backend.model.models import Restaurant, Review, UserFeedback, UserProfile, SearchLog
        from backend.memory.long_term import ConversationHistory, UserMemorySummary

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        ok("Connected to Neon PostgreSQL")
        ok("All tables created / verified")
        return True

    except Exception as e:
        fail(f"Database connection failed: {e}")
        info("Check NEON_DATABASE_URL in your .env file")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 2 — OSM fetch (Islamabad only, small test)
# ══════════════════════════════════════════════════════════════

def test_osm_fetch():
    head("STEP 2 — OSM fetch (Islamabad)")

    try:
        from backend.data_loader.restaurant_fetcher import fetch_city_osm
        t0 = time.time()
        restaurants = fetch_city_osm("Islamabad")
        elapsed = round(time.time() - t0, 1)

        if not restaurants:
            fail("OSM returned 0 restaurants — check internet connection")
            return []

        ok(f"Fetched {len(restaurants)} restaurants in {elapsed}s")

        # Show first 3
        for r in restaurants[:3]:
            info(f"  {r['name']} | {r['cuisine']} | {r['city']}")
            if r.get('phone'):
                info(f"    phone: {r['phone']}")
            if r.get('website'):
                info(f"    website: {r['website']}")

        # Basic validation
        missing_name = [r for r in restaurants if not r.get('name')]
        if missing_name:
            fail(f"{len(missing_name)} restaurants have no name — will be skipped")
        else:
            ok("All records have names")

        return restaurants

    except Exception as e:
        fail(f"OSM fetch failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════
# STEP 3 — Insert into PostgreSQL
# ══════════════════════════════════════════════════════════════

async def test_db_insert(restaurants: list) -> list:
    head("STEP 3 — Insert into PostgreSQL")

    if not restaurants:
        fail("No restaurants to insert — skipping")
        return []

    try:
        from backend.db.database import AsyncSessionLocal
        from backend.model.models import Restaurant
        from sqlalchemy import select

        inserted_records = []
        saved, skipped = [], []

        async with AsyncSessionLocal() as db:
            for item in restaurants:
                name = item.get("name", "").strip()
                city = item.get("city", "").strip()

                if not name:
                    continue

                # Check duplicate
                result = await db.execute(
                    select(Restaurant).where(
                        Restaurant.name == name,
                        Restaurant.city == city
                    )
                )
                existing = result.scalars().first()

                if existing:
                    skipped.append(name)
                    inserted_records.append(existing)
                    continue

                record = Restaurant(
                    name=name,
                    cuisine=item.get("cuisine", "Restaurant"),
                    city=city,
                    address=item.get("address"),
                    phone=item.get("phone"),
                    website=item.get("website"),
                    menu_url=item.get("menu_url"),
                    latitude=item.get("latitude"),
                    longitude=item.get("longitude"),
                    opening_hours=item.get("opening_hours"),
                    source=item.get("source", "openstreetmap"),
                    external_id=item.get("external_id"),
                    is_embedded=False
                )
                db.add(record)
                saved.append(name)
                inserted_records.append(record)

            await db.commit()

            # Refresh to get IDs
            for record in inserted_records:
                if record.id is None:
                    await db.refresh(record)

        ok(f"Inserted: {len(saved)} new restaurants")
        if skipped:
            info(f"Skipped (already exist): {len(skipped)}")

        # Show sample with IDs
        for r in inserted_records[:3]:
            info(f"  ID={r.id} | {r.name} | {r.cuisine}")

        return inserted_records

    except Exception as e:
        fail(f"DB insert failed: {e}")
        import traceback; traceback.print_exc()
        return []


# ══════════════════════════════════════════════════════════════
# STEP 4 — Generate embeddings
# ══════════════════════════════════════════════════════════════

def test_embeddings(records: list) -> bool:
    head("STEP 4 — Generate embeddings")

    if not records:
        fail("No records to embed — skipping")
        return False

    try:
        from backend.retrieving.embedder import embed_restaurant, build_restaurant_text

        sample = records[0]
        t0 = time.time()

        text = build_restaurant_text(
            sample.name, sample.cuisine, sample.city,
            sample.description if hasattr(sample, 'description') else None
        )
        vector = embed_restaurant(sample.name, sample.cuisine, sample.city)
        elapsed = round((time.time() - t0) * 1000, 1)

        ok(f"Model loaded and embedding generated in {elapsed}ms")
        info(f"  Text: '{text[:80]}...'")
        info(f"  Vector dimensions: {len(vector)}")
        info(f"  First 5 values: {[round(v, 4) for v in vector[:5]]}")

        if len(vector) != 384:
            fail(f"Expected 384 dimensions, got {len(vector)}")
            return False

        ok("Embedding dimensions correct (384)")
        return True

    except Exception as e:
        fail(f"Embedding failed: {e}")
        info("Run: pip install sentence-transformers")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 5 — Store in ChromaDB
# ══════════════════════════════════════════════════════════════

async def test_chromadb(records: list) -> bool:
    head("STEP 5 — Store in ChromaDB")

    if not records:
        fail("No records to store — skipping")
        return False

    try:
        from backend.retrieving.vector_store import upsert_restaurant, collection_count
        from backend.db.database import AsyncSessionLocal
        from backend.model.models import Restaurant
        from sqlalchemy import select

        embedded_ids = []
        errors = []

        for record in records:
            if not record.id:
                continue
            try:
                upsert_restaurant(
                    restaurant_id=record.id,
                    name=record.name,
                    cuisine=record.cuisine,
                    city=record.city,
                    description=getattr(record, 'description', None),
                    tags=getattr(record, 'tags', None)
                )
                embedded_ids.append(record.id)
            except Exception as e:
                errors.append(f"{record.name}: {e}")

        # Mark as embedded in PostgreSQL
        if embedded_ids:
            async with AsyncSessionLocal() as db:
                for rid in embedded_ids:
                    result = await db.execute(
                        select(Restaurant).where(Restaurant.id == rid)
                    )
                    r = result.scalars().first()
                    if r:
                        r.is_embedded = True
                await db.commit()

        total_in_chroma = collection_count()

        ok(f"Embedded {len(embedded_ids)} restaurants into ChromaDB")
        ok(f"Total vectors in ChromaDB: {total_in_chroma}")

        if errors:
            for e in errors[:3]:
                fail(f"Embed error: {e}")

        return len(embedded_ids) > 0

    except Exception as e:
        fail(f"ChromaDB store failed: {e}")
        info("Run: pip install chromadb")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 6 — Build BM25 index
# ══════════════════════════════════════════════════════════════

async def test_bm25(records: list) -> bool:
    head("STEP 6 — Build BM25 index")

    if not records:
        fail("No records — skipping")
        return False

    try:
        from backend.retrieving.bm25_store import build_index, index_exists
        from backend.db.database import AsyncSessionLocal
        from backend.model.models import Restaurant
        from sqlalchemy import select

        # Build from full DB table (not just test records)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Restaurant))
            all_restaurants = result.scalars().all()

        build_index([
            {"id": r.id, "name": r.name,
             "cuisine": r.cuisine, "city": r.city}
            for r in all_restaurants
        ])

        ok(f"BM25 index built from {len(all_restaurants)} restaurants")
        ok(f"Index saved to disk: {index_exists()}")
        return True

    except Exception as e:
        fail(f"BM25 build failed: {e}")
        info("Run: pip install rank-bm25")
        return False


# ══════════════════════════════════════════════════════════════
# STEP 7 — Test search
# ══════════════════════════════════════════════════════════════

def test_search() -> bool:
    head("STEP 7 — Test search")

    try:
        from backend.retrieving.retrieval import hybrid_search
        from backend.retrieving.vector_store import search_restaurants
        from backend.retrieving.bm25_store import search_restaurants as bm25_search

        query = "restaurant Islamabad"

        # Dense search
        dense = search_restaurants(query, top_k=3)
        ok(f"Dense (ChromaDB): {len(dense)} results")
        for r in dense[:2]:
            info(f"  [{r['score']:.3f}] {r['name']} — {r['cuisine']}")

        # Sparse search
        sparse = bm25_search(query, top_k=3)
        ok(f"Sparse (BM25): {len(sparse)} results")
        for r in sparse[:2]:
            info(f"  [{r['score']:.3f}] {r['name']} — {r['cuisine']}")

        # Hybrid RRF
        hybrid = hybrid_search(query, top_k=5)
        ok(f"Hybrid RRF: {len(hybrid)} results")
        for r in hybrid[:3]:
            info(
                f"  [{r['rrf_score']:.4f}] {r['name']} — {r['cuisine']} "
                f"(dense rank: {r['dense_rank']}, sparse rank: {r['sparse_rank']})"
            )

        return len(hybrid) > 0

    except Exception as e:
        fail(f"Search failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

def print_summary(results: dict):
    head("TEST SUMMARY")

    all_passed = all(results.values())

    for step, passed in results.items():
        status = "\033[92m PASS\033[0m" if passed else "\033[91m FAIL\033[0m"
        print(f"  {status}  {step}")

    print()
    if all_passed:
        print("\033[92m🎉 All tests passed. Your pipeline is working.\033[0m")
        print("\nNext steps:")
        print("  1. Run the backend:  uvicorn backend.main:app --reload")
        print("  2. Run the frontend: streamlit run frontend/app.py")
        print("  3. Load full data:   POST http://localhost:8000/load-apify")
    else:
        print("\033[91m⚠️  Some tests failed. Fix the errors above before running the server.\033[0m")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    print("\n\033[1m🍽️  Connoisseur — Pipeline Test\033[0m")
    print("Testing: DB → OSM fetch → insert → embed → ChromaDB → BM25 → search")

    results = {}

    # # Step 1: Database
    # results["1. PostgreSQL connection"] = await test_database()
    # if not results["1. PostgreSQL connection"]:
    #     print_summary(results)
    #     return

    # # Step 2: OSM fetch
    # restaurants = test_osm_fetch()
    # results["2. OSM fetch"] = len(restaurants) > 0

    # # Step 3: DB insert
    # records = await test_db_insert(restaurants)
    # results["3. PostgreSQL insert"] = len(records) > 0

    records = await get_all_restaurants()

    # Step 4: Embeddings
    results["4. Embedding generation"] = test_embeddings(records)
    if not results["4. Embedding generation"]:
        print_summary(results)
        return
    
    

    # Step 5: ChromaDB
    results["5. ChromaDB store"] = await test_chromadb(records)

    # Step 6: BM25
    results["6. BM25 index build"] = await test_bm25(records)

    # Step 7: Search
    results["7. Hybrid search"] = test_search()

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
