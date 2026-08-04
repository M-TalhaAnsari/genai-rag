"""
backend/embedder.py
--------------------
Converts restaurant records into dense vector embeddings.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast on CPU (~10ms per record)
  - Free, no API key needed
  - Downloads once and caches locally

We build one text string per restaurant that captures all searchable
fields. This is what gets embedded and stored in ChromaDB.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once at module import — not on every request
_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_restaurant_text(name: str, cuisine: str, city: str) -> str:
    """
    Construct the text that will be embedded for a restaurant.

    We include all three fields so that semantic search can match
    queries like "spicy Japanese food in Rawalpindi" correctly.
    """
    return f"{name} is a {cuisine} restaurant located in {city}."


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string and return a plain Python list.
    ChromaDB expects list[float], not a numpy array.
    """
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_restaurant(name: str, cuisine: str, city: str) -> list[float]:
    """
    Build the restaurant text and embed it in one call.
    This is the main function used by vector_store.py.
    """
    text = build_restaurant_text(name, cuisine, city)
    return embed_text(text)
