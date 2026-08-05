"""
backend/vector_store.py
------------------------
ChromaDB operations for two collections:

  1. "restaurants"        — primary identity embedding (name, cuisine, city, description)
  2. "restaurant_reviews" — cautious review summary embedding (sanitised, not raw)

WHY TWO COLLECTIONS:
  Mixing review sentiment with identity embeddings degrades search quality.
  A query for "biryani Lahore" should match on cuisine/location,
  not on whether someone left a bad review about parking.
  Keeping them separate means we can weight them differently during search.
"""

import chromadb
from chromadb.config import Settings
from backend.embedder import embed_restaurant, embed_text, embed_review_summary

_client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(anonymized_telemetry=False)
)

# Collection 1: restaurant identity
_restaurants = _client.get_or_create_collection(
    name="restaurants",
    metadata={"hnsw:space": "cosine"}
)

# Collection 2: review summaries (separate — not mixed with identity)
_reviews = _client.get_or_create_collection(
    name="restaurant_reviews",
    metadata={"hnsw:space": "cosine"}
)


# ── Restaurant identity ────────────────────────────────────────────────────

def upsert_restaurant(
    restaurant_id: int,
    name: str,
    cuisine: str,
    city: str,
    description: str = None,
    tags: str = None
) -> None:
    """
    Embed and store a restaurant's identity.
    Using upsert — safe to call multiple times on the same ID.
    """
    from backend.embedder import build_restaurant_text
    text      = build_restaurant_text(name, cuisine, city, description, tags)
    embedding = embed_restaurant(name, cuisine, city, description, tags)

    _restaurants.upsert(
        ids=[str(restaurant_id)],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "restaurant_id": restaurant_id,
            "name":          name,
            "cuisine":       cuisine,
            "city":          city
        }]
    )


def search_restaurants(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic search over restaurant identities.
    Returns list of dicts with restaurant_id, name, cuisine, city, score.
    """
    query_embedding = embed_text(query)

    count = _restaurants.count()
    if count == 0:
        return []

    results = _restaurants.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["metadatas", "distances"]
    )

    hits = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({
            "restaurant_id": meta["restaurant_id"],
            "name":          meta["name"],
            "cuisine":       meta["cuisine"],
            "city":          meta["city"],
            "score":         round(1.0 - distance, 4)
        })
    return hits


def collection_count() -> int:
    return _restaurants.count()


# ── Review summaries ───────────────────────────────────────────────────────

def upsert_review_summary(
    restaurant_id: int,
    restaurant_name: str,
    cuisine: str,
    city: str,
    summary: str,
    confidence: str,
    avg_rating: float | None,
    polarised: bool,
    disclaimer: str,
    review_count: int
) -> None:
    """
    Embed and store a restaurant's cautious review summary.
    Called after summarise_reviews() produces a validated summary.
    """
    embedding = embed_review_summary(restaurant_name, cuisine, city, summary)

    _reviews.upsert(
        ids=[f"review_{restaurant_id}"],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{
            "restaurant_id": restaurant_id,
            "name":          restaurant_name,
            "cuisine":       cuisine,
            "city":          city,
            "confidence":    confidence,
            "avg_rating":    avg_rating or 0.0,
            "polarised":     int(polarised),   # ChromaDB stores bool as int
            "disclaimer":    disclaimer,
            "review_count":  review_count
        }]
    )


def search_by_review_sentiment(query: str, top_k: int = 10) -> list[dict]:
    """
    Search restaurants by their review summary content.
    Used as an optional third signal in hybrid search.

    Returns results with a disclaimer field — always shown to users.
    """
    query_embedding = embed_text(query)
    count = _reviews.count()
    if count == 0:
        return []

    results = _reviews.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["metadatas", "distances", "documents"]
    )

    hits = []
    for meta, distance, doc in zip(
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0]
    ):
        hits.append({
            "restaurant_id": meta["restaurant_id"],
            "name":          meta["name"],
            "cuisine":       meta["cuisine"],
            "city":          meta["city"],
            "review_summary": doc,
            "confidence":    meta["confidence"],
            "avg_rating":    meta["avg_rating"] or None,
            "polarised":     bool(meta["polarised"]),
            "disclaimer":    meta["disclaimer"],
            "review_count":  meta["review_count"],
            "score":         round(1.0 - distance, 4)
        })
    return hits


def review_collection_count() -> int:
    return _reviews.count()


def get_review_summary(restaurant_id: int) -> dict | None:
    """
    Fetch the stored review summary for one restaurant by ID.
    Returns None if no summary has been generated yet.
    """
    try:
        result = _reviews.get(
            ids=[f"review_{restaurant_id}"],
            include=["metadatas", "documents"]
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        doc  = result["documents"][0]
        return {
            "review_summary": doc,
            "confidence":     meta["confidence"],
            "avg_rating":     meta["avg_rating"] or None,
            "polarised":      bool(meta["polarised"]),
            "disclaimer":     meta["disclaimer"],
            "review_count":   meta["review_count"]
        }
    except Exception:
        return None
