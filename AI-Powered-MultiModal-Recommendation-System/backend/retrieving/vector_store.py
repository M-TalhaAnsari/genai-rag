"""
backend/vector_store.py
------------------------
All ChromaDB operations live here.

ChromaDB is used as a local persistent vector database.
It stores the dense embeddings for each restaurant so we
can do semantic (meaning-based) search.

Storage location: ./chroma_data/ (created automatically)
Collection name: restaurants

Each document stored in ChromaDB has:
  - id       : str(restaurant.id from PostgreSQL)
  - embedding: list[float] from embedder.py
  - document : the raw text that was embedded (for debugging)
  - metadata : name, cuisine, city (returned in search results)
"""

import chromadb
from chromadb.config import Settings
from backend.retrieving.embedder import embed_restaurant, embed_text

# Persistent client — data survives restarts
# On Hugging Face Spaces, point this to /data/chroma_data
_client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(anonymized_telemetry=False)
)

# Get or create the collection once
_collection = _client.get_or_create_collection(
    name="restaurants",
    metadata={"hnsw:space": "cosine"}   # cosine similarity for text
)


def upsert_restaurant(
    restaurant_id: int,
    name: str,
    cuisine: str,
    city: str
) -> None:
    """
    Generate an embedding for this restaurant and store it in ChromaDB.

    Using upsert (not add) means running this twice on the same
    restaurant_id will update it rather than create a duplicate.
    ChromaDB ids must be strings.
    """
    from backend.retrieving.embedder import build_restaurant_text

    text = build_restaurant_text(name, cuisine, city)
    embedding = embed_restaurant(name, cuisine, city)

    _collection.upsert(
        ids=[str(restaurant_id)],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "restaurant_id": restaurant_id,
            "name": name,
            "cuisine": cuisine,
            "city": city
        }]
    )


def search_restaurants(query: str, top_k: int = 10) -> list[dict]:
    """
    Embed the query and find the top_k most semantically similar restaurants.

    Returns a list of dicts, each containing:
      - restaurant_id (int)
      - name, cuisine, city (str)
      - score (float, 0–1, higher = more similar)

    ChromaDB returns distances (lower = closer for cosine).
    We convert: score = 1 - distance
    """
    query_embedding = embed_text(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count() or 1),
        include=["metadatas", "distances"]
    )

    hits = []
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for meta, distance in zip(metadatas, distances):
        hits.append({
            "restaurant_id": meta["restaurant_id"],
            "name": meta["name"],
            "cuisine": meta["cuisine"],
            "city": meta["city"],
            "score": round(1.0 - distance, 4)
        })

    return hits


def collection_count() -> int:
    """Return how many restaurants are currently in ChromaDB."""
    return _collection.count()
