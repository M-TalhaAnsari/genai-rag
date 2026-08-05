"""
backend/embedder.py
--------------------
Converts restaurant records into dense vector embeddings.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast on CPU (~10ms per record)
  - Free, no API key needed

TWO EMBEDDING STRATEGIES
--------------------------
1. Restaurant identity embedding (name + cuisine + city)
   Used for: fast hybrid search
   Stored in: ChromaDB collection "restaurants"

2. Review summary embedding (sanitised aggregate of reviews)
   Used for: sentiment-aware recommendation
   Stored in: ChromaDB collection "restaurant_reviews"
   
   WHY SUMMARIES NOT RAW REVIEWS:
   - Reviews are mixed (fake, biased, outdated)
   - Raw review text is noisy and long
   - We ask an LLM to write a CAUTIOUS 2-sentence summary
     that captures only the consistent patterns across reviews
   - The summary is embedded, not the raw text
   - During search we surface the summary with a disclaimer
"""

from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_restaurant_text(name: str, cuisine: str, city: str,
                           description: str = None, tags: str = None) -> str:
    """
    Build the primary embedding text for a restaurant.
    Includes description and tags if available for richer semantic matching.
    """
    parts = [f"{name} is a {cuisine} restaurant located in {city}."]
    if description:
        parts.append(description[:300])   # cap to avoid bloat
    if tags:
        import json
        try:
            tag_list = json.loads(tags)
            if tag_list:
                parts.append("Tags: " + ", ".join(tag_list[:10]))
        except Exception:
            pass
    return " ".join(parts)


def build_review_summary_text(
    restaurant_name: str,
    cuisine: str,
    city: str,
    review_summary: str
) -> str:
    """
    Build the text that gets embedded for a restaurant's review summary.
    This is what we store in the "restaurant_reviews" collection.
    """
    return (
        f"{restaurant_name} ({cuisine}, {city}). "
        f"Customer feedback summary: {review_summary}"
    )


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns list[float] for ChromaDB."""
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_restaurant(name: str, cuisine: str, city: str,
                     description: str = None, tags: str = None) -> list[float]:
    """Primary restaurant embedding — used by vector_store.upsert_restaurant."""
    text = build_restaurant_text(name, cuisine, city, description, tags)
    return embed_text(text)


def embed_review_summary(restaurant_name: str, cuisine: str,
                         city: str, review_summary: str) -> list[float]:
    """Review summary embedding — used by vector_store.upsert_review_summary."""
    text = build_review_summary_text(restaurant_name, cuisine, city, review_summary)
    return embed_text(text)
