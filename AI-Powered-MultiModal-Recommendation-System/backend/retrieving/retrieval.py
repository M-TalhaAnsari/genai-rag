"""
backend/retrieval.py
---------------------
Hybrid search: combines dense (ChromaDB) + sparse (BM25) results
using Reciprocal Rank Fusion (RRF).

Why hybrid search?
  - Dense search (embeddings) is great at semantic meaning.
    "Best ramen place" finds Japanese restaurants even if they
    don't use the word "ramen".
  - Sparse search (BM25) is great at exact keywords.
    "Burger Lab" finds that exact name even if semantically
    other burger places are more "similar".
  - RRF combines both ranked lists into one final ranking
    without needing to normalise scores across different scales.

RRF formula (per document):
  rrf_score = sum over each list of: 1 / (k + rank)

  k=60 is the standard constant (from the original RRF paper).
  A document ranked #1 in both lists gets the highest combined score.
  A document appearing in only one list still gets partial credit.
"""

from backend.retrieving import vector_store, bm25_store

RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Run dense + sparse search and merge results with RRF.

    Args:
        query:  The user's natural language search query.
        top_k:  How many final results to return.

    Returns:
        List of restaurant dicts, ordered by RRF score descending.
        Each dict contains:
          - restaurant_id, name, cuisine, city
          - rrf_score      (final combined score)
          - dense_rank     (rank in semantic results, None if absent)
          - sparse_rank    (rank in keyword results, None if absent)
    """
    # Run both searches — fetch more than top_k so RRF has enough to merge
    fetch_k = max(top_k * 2, 20)

    dense_results = vector_store.search_restaurants(query, top_k=fetch_k)
    sparse_results = bm25_store.search_restaurants(query, top_k=fetch_k)

    # Map restaurant_id → accumulated RRF score + metadata
    scores: dict[int, dict] = {}

    # Process dense results
    for rank, hit in enumerate(dense_results, start=1):
        rid = hit["restaurant_id"]
        if rid not in scores:
            scores[rid] = {
                "restaurant_id": rid,
                "name": hit["name"],
                "cuisine": hit["cuisine"],
                "city": hit["city"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None
            }
        scores[rid]["rrf_score"] += _rrf_score(rank)
        scores[rid]["dense_rank"] = rank

    # Process sparse results
    for rank, hit in enumerate(sparse_results, start=1):
        rid = hit["restaurant_id"]
        if rid not in scores:
            scores[rid] = {
                "restaurant_id": rid,
                "name": hit["name"],
                "cuisine": hit["cuisine"],
                "city": hit["city"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None
            }
        scores[rid]["rrf_score"] += _rrf_score(rank)
        scores[rid]["sparse_rank"] = rank

    # Sort by RRF score descending, return top_k
    ranked = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    # Round for clean API output
    for r in ranked:
        r["rrf_score"] = round(r["rrf_score"], 6)

    return ranked[:top_k]
