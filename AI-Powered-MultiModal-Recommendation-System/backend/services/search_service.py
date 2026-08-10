"""
backend/services/search_service.py
------------------------------------
All search logic in one place.
Three search modes:

  hybrid_search()       — BM25 + dense ChromaDB, RRF fused
  full_search()         — hybrid + review sentiment, weighted merge
  review_search()       — review sentiment only (ChromaDB review collection)

After search:
  - apply_profile_boost() optionally reranks by user cuisine preferences
  - log_search() writes to search_logs for analytics
  - session memory is updated (queries + results)
"""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.retrieving import retrieval, vector_store
from backend.services.feedback_service import apply_profile_boost
from backend.services.analytics_service import log_search
from backend.services import memory_service

from backend.retrieving.vector_store import search_by_review_sentiment

async def hybrid_search(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    user_id: str | None = None,
    profile: dict | None = None,
) -> dict:
    """
    BM25 + ChromaDB dense search fused with RRF.
    Optionally personalised using the user's feedback profile.
    """
    results = retrieval.hybrid_search(query=query, top_k=top_k)

    personalised = False
    if profile:
        results    = apply_profile_boost(results, profile)
        personalised = True

    if user_id:
        memory_service.add_query(user_id, query)
        memory_service.add_results(user_id, results)

    await log_search(db=db, query=query, result_count=len(results), user_id=user_id)

    return {
        "query":        query,
        "result_count": len(results),
        "personalised": personalised,
        "results":      results,
    }


async def full_search(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    user_id: str | None = None,
    profile: dict | None = None,
    w_identity: float = 0.6,
    w_review: float = 0.4,
) -> dict:
    """
    Three-signal search: BM25 + dense (identity) + review sentiment.
    Signals merged using configurable weights.
    Fake-review-flagged restaurants have their review weight halved.
    """
    from backend.retrieving.vector_store import search_by_review_sentiment, review_collection_count

    identity_results = retrieval.hybrid_search(query=query, top_k=top_k * 2)
    review_results   = (
        search_by_review_sentiment(query=query, top_k=top_k * 2)
        if review_collection_count() > 0 else []
    )

    scores: dict[int, dict] = {}

    for rank, r in enumerate(identity_results, start=1):
        rid = r["restaurant_id"]
        scores[rid] = {
            "restaurant_id":     rid,
            "name":              r["name"],
            "cuisine":           r["cuisine"],
            "city":              r["city"],
            "identity_score":    round(w_identity / (60 + rank), 6),
            "review_score":      0.0,
            "combined_score":    0.0,
            "review_summary":    None,
            "review_disclaimer": None,
            "has_fake_signals":  False,
            "most_recent_review": None,
        }

    for rank, r in enumerate(review_results, start=1):
        rid        = r["restaurant_id"]
        eff_w      = w_review * (0.5 if r.get("has_fake_signals") else 1.0)
        contrib    = round(eff_w / (60 + rank), 6)

        if rid in scores:
            scores[rid]["review_score"]      = contrib
            scores[rid]["review_summary"]    = r.get("review_summary")
            scores[rid]["review_disclaimer"] = r.get("disclaimer")
            scores[rid]["has_fake_signals"]  = r.get("has_fake_signals", False)
            scores[rid]["most_recent_review"] = r.get("most_recent_review")
        else:
            scores[rid] = {
                "restaurant_id":     rid,
                "name":              r["name"],
                "cuisine":           r["cuisine"],
                "city":              r["city"],
                "identity_score":    0.0,
                "review_score":      contrib,
                "combined_score":    0.0,
                "review_summary":    r.get("review_summary"),
                "review_disclaimer": r.get("disclaimer"),
                "has_fake_signals":  r.get("has_fake_signals", False),
                "most_recent_review": r.get("most_recent_review"),
            }

    for rid in scores:
        scores[rid]["combined_score"] = round(
            scores[rid]["identity_score"] + scores[rid]["review_score"], 6
        )

    ranked = sorted(scores.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]

    personalised = False
    if profile:
        ranked       = apply_profile_boost(ranked, profile)
        personalised = True

    if user_id:
        memory_service.add_query(user_id, query)

    await log_search(db=db, query=query, result_count=len(ranked), user_id=user_id)

    return {
        "query":        query,
        "result_count": len(ranked),
        "personalised": personalised,
        "weights":      {"identity": w_identity, "review": w_review},
        "results":      ranked,
    }


async def review_search(
    query: str,
    top_k: int = 5,
    min_confidence: str = "low",
) -> dict:
    """
    Search restaurants by their review summary content only.
    Useful for sentiment-heavy queries like 'clean and fast service'.
    """

    confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    min_rank        = confidence_rank.get(min_confidence, 1)

    results = search_by_review_sentiment(query=query, top_k=top_k * 2)
    results = [
        r for r in results
        if confidence_rank.get(r.get("confidence", "none"), 0) >= min_rank
    ][:top_k]

    return {
        "query":        query,
        "result_count": len(results),
        "source":       "review_summaries",
        "results":      results,
    }