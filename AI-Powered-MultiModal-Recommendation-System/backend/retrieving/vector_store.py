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
import json as _json
from chromadb.config import Settings
import torch
from transformers import CLIPModel, CLIPProcessor

from backend.retrieving.embedder import embed_restaurant, embed_text, embed_review_summary
from backend.retrieving.data_enrichment import normalise_cuisine, extract_area

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


# ── Restaurant images (CLIP embeddings) ────────────────────────────────────
# Third collection — image embeddings via CLIP (512-dim, cosine similarity).
# Separate from text embeddings because CLIP and sentence-transformer
# embeddings live in different vector spaces and cannot be mixed.
#
# Each document represents ONE image from a restaurant.
# Multiple images per restaurant → multiple rows, all with same restaurant_id.
# This lets a single image query match the best photo from each restaurant.

_images = _client.get_or_create_collection(
    name="restaurant_images",
    metadata={"hnsw:space": "cosine"}
)

# CLIP model — loaded once, shared across all image embedding calls
_clip_model     = None
_clip_processor = None
_clip_device    = None


def _get_clip():
    """Lazy-load CLIP so it doesn't slow startup if images aren't used."""
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        _clip_device    = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_clip_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        _clip_model.eval()
    return _clip_model, _clip_processor, _clip_device


def embed_image_from_bytes(image_bytes: bytes) -> list[float]:
    """
    Embed one image (raw bytes) using CLIP.
    Returns a 512-dim normalised float list for ChromaDB.
    """
    import torch
    import io
    from PIL import Image

    model, processor, device = _get_clip()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        feats  = model.get_image_features(**inputs)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().tolist()


def embed_text_clip(text: str) -> list[float]:
    """
    Embed a text query using CLIP's text encoder.
    Used so text queries can match against image embeddings.
    CLIP text and image embeddings share the same 512-dim space.
    """
    import torch

    model, processor, device = _get_clip()

    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        feats  = model.get_text_features(**inputs)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().tolist()


def upsert_restaurant_image(
    restaurant_id: int,
    image_index: int,
    image_url: str,
    image_bytes: bytes,
    name: str,
    cuisine: str,
    city: str,
) -> None:
    """
    Embed one restaurant image and store it in ChromaDB.

    Each image gets a unique ID: f"img_{restaurant_id}_{image_index}"
    Multiple images per restaurant are stored as separate rows —
    all tied back to the same restaurant_id in metadata.

    Args:
        restaurant_id: PostgreSQL ID of the restaurant
        image_index:   Position of this image in the restaurant's gallery (0-based)
        image_url:     URL where the image can be retrieved (stored for display)
        image_bytes:   Raw downloaded image bytes (used for embedding)
        name/cuisine/city: Restaurant metadata — stored for search result display
    """
    embedding = embed_image_from_bytes(image_bytes)

    _images.upsert(
        ids=[f"img_{restaurant_id}_{image_index}"],
        embeddings=[embedding],
        documents=[image_url],          # store URL as the document text
        metadatas=[{
            "restaurant_id": restaurant_id,
            "image_index":   image_index,
            "image_url":     image_url,
            "name":          name,
            "cuisine":       cuisine,
            "city":          city,
        }]
    )


def search_by_image_text(query: str, top_k: int = 10) -> list[dict]:
    """
    Search restaurant images using a text query via CLIP.

    CLIP's shared embedding space means text queries can match images
    semantically — "rooftop with city view" finds photos of rooftop restaurants
    even if that text appears nowhere in the metadata.

    Returns one result per IMAGE (not per restaurant).
    Use deduplicate=True to get one result per restaurant instead.
    """
    count = _images.count()
    if count == 0:
        return []

    query_embedding = embed_text_clip(query)

    results = _images.query(
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
            "image_url":     meta["image_url"],
            "image_index":   meta["image_index"],
            "score":         round(1.0 - distance, 4),
        })
    return hits


def search_by_image_text_deduped(query: str, top_k: int = 10) -> list[dict]:
    """
    Same as search_by_image_text but returns one result per restaurant.
    Keeps the highest-scoring image per restaurant.
    """
    raw = search_by_image_text(query, top_k=top_k * 3)
    seen: dict[int, dict] = {}
    for hit in raw:
        rid = hit["restaurant_id"]
        if rid not in seen or hit["score"] > seen[rid]["score"]:
            seen[rid] = hit
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def image_collection_count() -> int:
    return _images.count()


def get_restaurant_images(restaurant_id: int) -> list[dict]:
    """
    Fetch all stored image embeddings for one restaurant.
    Returns list of {image_url, image_index, score=None}.
    """
    try:
        result = _images.get(
            where={"restaurant_id": restaurant_id},
            include=["metadatas", "documents"]
        )
        if not result["ids"]:
            return []
        return [
            {
                "image_url":   meta["image_url"],
                "image_index": meta["image_index"],
            }
            for meta in result["metadatas"]
        ]
    except Exception:
        return []
