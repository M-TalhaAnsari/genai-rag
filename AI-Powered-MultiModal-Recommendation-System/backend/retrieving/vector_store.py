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
from backend.retrieving.embedder import embed_restaurant, embed_text, embed_review_summary

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
    address: str = None,
    description: str = None,
    tags: str = None,
    opening_hours: str = None,
    phone: str = None,
    website: str = None,
    rating: float = None,
    latitude: float = None,
    longitude: float = None,
) -> None:
    """
    Build rich embedding text, embed it, and store in ChromaDB.
    Metadata includes normalised cuisine and area for filtering.
    """
    from backend.retrieving.data_enrichment import normalise_cuisine, extract_area

    clean_cuisine = normalise_cuisine(cuisine, name)
    area = extract_area(address or "", city) or ""

    embedding, text = embed_restaurant(
        name=name, cuisine=cuisine, city=city,
        address=address, description=description,
        tags=tags, opening_hours=opening_hours,
        phone=phone, website=website, rating=rating
    )

    _restaurants.upsert(
        ids=[str(restaurant_id)],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "restaurant_id":    restaurant_id,
            "name":             name,
            "cuisine":          clean_cuisine,   # normalised
            "cuisine_raw":      cuisine,          # original for reference
            "city":             city,
            "area":             area,
            "has_phone":        int(bool(phone)),
            "has_website":      int(bool(website)),
            "rating":           float(rating) if rating else 0.0,
            "latitude":         float(latitude) if latitude else 0.0,
            "longitude":        float(longitude) if longitude else 0.0,
        }]
    )


def search_restaurants(
    query: str,
    top_k: int = 10,
    where: dict = None
) -> list[dict]:
    """
    Semantic search over restaurant identities.

    Args:
        query:  Natural language search query
        top_k:  Number of results
        where:  Optional ChromaDB metadata filter, e.g.:
                {"cuisine": "Chinese"}
                {"city": "Islamabad", "has_phone": 1}

    Returns list of dicts with restaurant_id, name, cuisine, city, area, score.
    """
    query_embedding = embed_text(query)

    count = _restaurants.count()
    if count == 0:
        return []

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        min(top_k, count),
        "include":          ["metadatas", "distances"]
    }
    if where:
        kwargs["where"] = where

    results = _restaurants.query(**kwargs)

    hits = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({
            "restaurant_id": meta["restaurant_id"],
            "name":          meta["name"],
            "cuisine":       meta["cuisine"],
            "city":          meta["city"],
            "area":          meta.get("area", ""),
            "has_phone":     bool(meta.get("has_phone", 0)),
            "has_website":   bool(meta.get("has_website", 0)),
            "rating":        meta.get("rating") or None,
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
    weighted_rating: float | None = None,
    polarised: bool = False,
    burst_detected: bool = False,
    has_fake_signals: bool = False,
    disclaimer: str = "",
    review_count: int = 0,
    most_recent_review: str | None = None,
    oldest_review: str | None = None,
    dimensions: dict = None,
    **kwargs   # absorb any extra fields from summarise_reviews output
) -> None:
    """
    Embed and store a restaurant's review summary with full quality metadata.
    Called after summarise_reviews() produces a validated summary.
    """
    import json as _json
    embedding = embed_review_summary(restaurant_name, cuisine, city, summary)

    _reviews.upsert(
        ids=[f"review_{restaurant_id}"],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{
            "restaurant_id":      restaurant_id,
            "name":               restaurant_name,
            "cuisine":            cuisine,
            "city":               city,
            "confidence":         confidence,
            "avg_rating":         avg_rating or 0.0,
            "weighted_rating":    weighted_rating or 0.0,
            "polarised":          int(polarised),
            "burst_detected":     int(burst_detected),
            "has_fake_signals":   int(has_fake_signals),
            "disclaimer":         disclaimer,
            "review_count":       review_count,
            "most_recent_review": most_recent_review or "",
            "oldest_review":      oldest_review or "",
            "dimensions":         _json.dumps(dimensions or {})
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
        import json as _json
        return {
            "review_summary":     doc,
            "confidence":         meta["confidence"],
            "avg_rating":         meta["avg_rating"] or None,
            "weighted_rating":    meta.get("weighted_rating") or None,
            "polarised":          bool(meta.get("polarised", 0)),
            "burst_detected":     bool(meta.get("burst_detected", 0)),
            "has_fake_signals":   bool(meta.get("has_fake_signals", 0)),
            "disclaimer":         meta.get("disclaimer", ""),
            "review_count":       meta.get("review_count", 0),
            "most_recent_review": meta.get("most_recent_review") or None,
            "oldest_review":      meta.get("oldest_review") or None,
            "dimensions":         _json.loads(meta.get("dimensions", "{}"))
        }
    except Exception:
        return None
