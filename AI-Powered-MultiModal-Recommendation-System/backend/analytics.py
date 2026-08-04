"""
backend/analytics.py
---------------------
Analytics queries over search_logs and user_feedback tables.

All functions are async and take a db session.
Called by GET /analytics in main.py.

Metrics provided:
  - total_searches      : all-time search count
  - searches_today      : searches in the last 24 hours
  - top_queries         : most frequent search strings
  - top_cuisines_liked  : cuisines most often liked via feedback
  - top_cuisines_avoided: cuisines most often disliked
  - total_feedback      : total thumbs up + thumbs down recorded
  - like_rate           : percentage of positive signals
"""

import json
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.model.models import SearchLog, UserFeedback


async def get_analytics(db: AsyncSession) -> dict:
    """
    Compute and return all analytics metrics in one call.
    """
    total_searches = await _total_searches(db)
    searches_today = await _searches_today(db)
    top_queries = await _top_queries(db, limit=10)
    feedback_summary = await _feedback_summary(db)
    top_cuisines_liked = await _top_cuisines(db, signal=1, limit=10)
    top_cuisines_avoided = await _top_cuisines(db, signal=-1, limit=10)

    return {
        "search_volume": {
            "total": total_searches,
            "last_24h": searches_today
        },
        "top_queries": top_queries,
        "feedback": feedback_summary,
        "cuisine_preferences": {
            "most_liked": top_cuisines_liked,
            "most_avoided": top_cuisines_avoided
        }
    }


async def log_search(
    db: AsyncSession,
    query: str,
    result_count: int,
    user_id: str = None
) -> None:
    """
    Write one search event to search_logs.
    Called by the /search endpoint after every successful query.
    Fire-and-forget — errors are caught and logged, not raised.
    """
    try:
        log = SearchLog(
            user_id=user_id,
            query=query.strip(),
            result_count=result_count
        )
        db.add(log)
        await db.commit()
    except Exception as e:
        print(f"[analytics] log_search error: {e}")


# ── Private helpers ─────────────────────────────────────────────────────────

async def _total_searches(db: AsyncSession) -> int:
    result = await db.execute(
        select(func.count()).select_from(SearchLog)
    )
    return result.scalar() or 0


async def _searches_today(db: AsyncSession) -> int:
    cutoff = datetime.utcnow() - timedelta(hours=24)
    result = await db.execute(
        select(func.count())
        .select_from(SearchLog)
        .where(SearchLog.created_at >= cutoff)
    )
    return result.scalar() or 0


async def _top_queries(db: AsyncSession, limit: int = 10) -> list[dict]:
    result = await db.execute(
        select(SearchLog.query, func.count().label("count"))
        .group_by(SearchLog.query)
        .order_by(func.count().desc())
        .limit(limit)
    )
    rows = result.all()
    return [{"query": row.query, "count": row.count} for row in rows]


async def _feedback_summary(db: AsyncSession) -> dict:
    result = await db.execute(select(UserFeedback))
    all_feedback = result.scalars().all()

    total = len(all_feedback)
    likes = sum(1 for f in all_feedback if f.signal == 1)
    dislikes = total - likes
    like_rate = round(likes / total * 100, 1) if total > 0 else 0.0

    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "like_rate_pct": like_rate
    }


async def _top_cuisines(
    db: AsyncSession,
    signal: int,
    limit: int = 10
) -> list[dict]:
    result = await db.execute(
        select(UserFeedback.cuisine, func.count().label("count"))
        .where(UserFeedback.signal == signal)
        .group_by(UserFeedback.cuisine)
        .order_by(func.count().desc())
        .limit(limit)
    )
    rows = result.all()
    return [{"cuisine": row.cuisine, "count": row.count} for row in rows]
