"""
backend/bm25_store.py
----------------------
BM25 keyword (sparse) search over restaurants.

BM25 is the algorithm behind traditional search engines like Elasticsearch.
It scores documents by exact term frequency — good at matching specific
words that semantic search might miss (e.g. a restaurant name like
"Burger Lab" or a city like "Islamabad").

We persist the index to disk as a pickle file so it survives restarts
and does not need to be rebuilt from scratch every time.

Index file: ./bm25_index.pkl

The index is rebuilt completely whenever new restaurants are added.
Because the dataset is small (hundreds to low thousands of restaurants),
a full rebuild takes under 1 second and is simpler than an incremental update.
"""

import pickle
import os
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = "./bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Good enough for restaurant names, cuisines, and cities.
    """
    return text.lower().split()


def build_index(restaurants: list[dict]) -> None:
    """
    Build a BM25 index from a list of restaurant dicts and save to disk.

    Each restaurant dict must have: id, name, cuisine, city.

    Called by /restaurant-sync after new records are inserted so the
    BM25 index always reflects what is in PostgreSQL.
    """
    if not restaurants:
        return

    corpus = []
    metadata = []

    for r in restaurants:
        text = f"{r['name']} {r['cuisine']} {r['city']}"
        corpus.append(_tokenize(text))
        metadata.append({
            "restaurant_id": r["id"],
            "name": r["name"],
            "cuisine": r["cuisine"],
            "city": r["city"]
        })

    bm25 = BM25Okapi(corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata}, f)


def search_restaurants(query: str, top_k: int = 10) -> list[dict]:
    """
    Search the BM25 index for restaurants matching the query.

    Returns a list of dicts with:
      - restaurant_id, name, cuisine, city
      - score (float, raw BM25 score — higher = better match)

    Returns empty list if no index exists yet.
    """
    if not os.path.exists(BM25_INDEX_PATH):
        return []

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    bm25: BM25Okapi = data["bm25"]
    metadata: list[dict] = data["metadata"]

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair each score with its metadata, sort descending
    scored = sorted(
        zip(scores, metadata),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, meta in scored[:top_k]:
        if score > 0:   # skip zero-score (no keyword overlap at all)
            results.append({
                "restaurant_id": meta["restaurant_id"],
                "name": meta["name"],
                "cuisine": meta["cuisine"],
                "city": meta["city"],
                "score": round(float(score), 4)
            })

    return results


def index_exists() -> bool:
    return os.path.exists(BM25_INDEX_PATH)
