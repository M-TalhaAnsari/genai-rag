"""
backend/embedder.py
--------------------
Embedding functions. Uses data_enrichment.py to build
semantically rich text before encoding.

Model: all-MiniLM-L6-v2 (384 dimensions, free, CPU-fast)
"""

from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    """Embed any text string. Returns list[float] for ChromaDB."""
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_restaurant(
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
) -> tuple[list[float], str]:
    """
    Build rich embedding text then embed it.

    Returns:
        (vector, text) — vector for ChromaDB, text for debugging/storage
    """
    from backend.retrieving.data_enrichment import build_rich_embedding_text

    text = build_rich_embedding_text(
        name=name, cuisine=cuisine, city=city,
        address=address, description=description,
        tags=tags, opening_hours=opening_hours,
        phone=phone, website=website, rating=rating
    )
    return embed_text(text), text


def embed_review_summary(
    restaurant_name: str,
    cuisine: str,
    city: str,
    review_summary: str
) -> list[float]:
    """Embed a review summary for the review_summaries collection."""
    from backend.retrieving.data_enrichment import normalise_cuisine
    clean_cuisine = normalise_cuisine(cuisine, restaurant_name)
    text = (
        f"{restaurant_name} ({clean_cuisine}, {city}). "
        f"Customer feedback: {review_summary}"
    )
    return embed_text(text)
