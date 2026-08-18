"""
backend/routers/search.py
--------------------------
Search endpoints — PUBLIC, no login required.

GET /search              — hybrid BM25 + dense + RRF
GET /search/full         — three-signal (identity + review sentiment)
GET /search/by-review    — review sentiment only
GET /search/by-image     — CLIP image search
"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.models.schemas import SearchResponse, SearchResult
from backend.retrieving import retrieval, vector_store
from backend.services import feedback_service as feedback_module
from backend.services import analytics_service as analytics_module
from backend.memory import session as session_memory
from backend.retrieving.vector_store import (
    search_by_image_text,
    search_by_image_text_deduped,
    image_collection_count,
    search_by_review_sentiment,
    review_collection_count
)

router = APIRouter(prefix="/search", tags=["search"])


def _safe_uuid(user_id: str | None) -> UUID | None:
    """user_id is now compared against a UUID column. A malformed or
    made-up string (e.g. a leftover test value) would otherwise raise
    a DB error and 500 the whole search. Since personalisation here is
    optional, invalid input just falls back to anonymous instead of
    breaking the request."""
    if not user_id:
        return None
    try:
        return UUID(user_id)
    except ValueError:
        return None


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    user_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Hybrid search: ChromaDB dense + BM25 sparse → RRF fusion."""
    if vector_store.collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No restaurants indexed. Run /ingestion/restaurant-sync first."
        )

    results = retrieval.hybrid_search(query=q, top_k=top_k)

    personalised = False
    uid = _safe_uuid(user_id)
    if uid:
        profile = await feedback_module.get_profile(db, uid)
        if profile:
            results = feedback_module.apply_profile_boost(results, profile)
            personalised = True
        await session_memory.add_query(user_id, q)
        await session_memory.add_results(user_id, results)

    await analytics_module.log_search(
        db=db, query=q, result_count=len(results), user_id=user_id
    )

    return SearchResponse(
        query=q,
        result_count=len(results),
        personalised=personalised,
        results=[SearchResult(**r) for r in results]
    )


@router.get("/full")
async def search_full(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    user_id: str | None = Query(default=None),
    w_identity: float = Query(default=0.6, ge=0.0, le=1.0),
    w_review: float = Query(default=0.4, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    """Three-signal search: BM25 + dense identity + review sentiment, RRF fused."""
    if vector_store.collection_count() == 0:
        raise HTTPException(status_code=503, detail="No restaurants indexed.")

    identity_results = retrieval.hybrid_search(query=q, top_k=top_k * 2)
    review_results = []
    if review_collection_count() > 0:
        review_results = search_by_review_sentiment(query=q, top_k=top_k * 2)

    scores: dict[int, dict] = {}

    for rank, r in enumerate(identity_results, start=1):
        rid = r["restaurant_id"]
        scores[rid] = {
            "restaurant_id": rid, "name": r["name"],
            "cuisine": r["cuisine"], "city": r["city"],
            "identity_score": round(w_identity * (1.0 / (60 + rank)), 6),
            "review_score": 0.0, "combined_score": 0.0,
            "review_summary": None, "review_disclaimer": None,
            "has_fake_signals": False, "most_recent_review": None,
        }

    for rank, r in enumerate(review_results, start=1):
        rid = r["restaurant_id"]
        effective_w = w_review * (0.5 if r.get("has_fake_signals") else 1.0)
        contribution = effective_w * (1.0 / (60 + rank))
        if rid in scores:
            scores[rid]["review_score"] = round(contribution, 6)
            scores[rid]["review_summary"] = r.get("review_summary")
            scores[rid]["review_disclaimer"] = r.get("disclaimer")
            scores[rid]["has_fake_signals"] = r.get("has_fake_signals", False)
            scores[rid]["most_recent_review"] = r.get("most_recent_review")
        else:
            scores[rid] = {
                "restaurant_id": rid, "name": r["name"],
                "cuisine": r["cuisine"], "city": r["city"],
                "identity_score": 0.0,
                "review_score": round(contribution, 6), "combined_score": 0.0,
                "review_summary": r.get("review_summary"),
                "review_disclaimer": r.get("disclaimer"),
                "has_fake_signals": r.get("has_fake_signals", False),
                "most_recent_review": r.get("most_recent_review"),
            }

    for rid in scores:
        scores[rid]["combined_score"] = round(
            scores[rid]["identity_score"] + scores[rid]["review_score"], 6
        )

    ranked = sorted(scores.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]

    personalised = False
    uid = _safe_uuid(user_id)
    if uid:
        profile = await feedback_module.get_profile(db, uid)
        if profile:
            ranked = feedback_module.apply_profile_boost(ranked, profile)
            personalised = True
        await session_memory.add_query(user_id, q)

    await analytics_module.log_search(
        db=db, query=q, result_count=len(ranked), user_id=user_id
    )

    return {
        "query": q, "result_count": len(ranked),
        "personalised": personalised,
        "weights": {"identity": w_identity, "review": w_review},
        "results": ranked
    }


@router.get("/by-review")
async def search_by_review(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    min_confidence: str = Query(default="low"),
):
    """Search restaurants by review summary content (sentiment-based)."""
    if review_collection_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No review summaries indexed. Run POST /ingestion/summarise-all-reviews first."
        )

    confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    min_rank = confidence_rank.get(min_confidence, 1)

    results = search_by_review_sentiment(query=q, top_k=top_k * 2)
    results = [
        r for r in results
        if confidence_rank.get(r.get("confidence", "none"), 0) >= min_rank
    ][:top_k]

    return {"query": q, "result_count": len(results), "source": "review_summaries", "results": results}


@router.get("/by-image")
async def search_by_image(
    q: str = Query(..., min_length=1,
                    description="e.g. 'rooftop seating with city view', 'plated biryani'"),
    top_k: int = Query(default=10, ge=1, le=30),
    dedupe: bool = Query(default=True,
                          description="One result per restaurant (best matching photo) vs one per image"),
):
    """Search restaurant photos using natural language via CLIP."""
    if image_collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No images indexed yet. Run POST /ingestion/enrich-reviews "
                "with embed_images=true first."
            )
        )

    results = (
        search_by_image_text_deduped(query=q, top_k=top_k)
        if dedupe else
        search_by_image_text(query=q, top_k=top_k)
    )

    return {
        "query": q,
        "result_count": len(results),
        "source": "restaurant_images",
        "deduped": dedupe,
        "results": results,
    }