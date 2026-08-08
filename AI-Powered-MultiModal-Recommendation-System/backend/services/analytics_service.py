"""
backend/services/analytics_service.py
---------------------------------------
All analytics DB queries in one place.
Routers import from here — no raw SQL in routers.
"""

from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import SearchLog, UserFeedback


async def get_analytics(db: AsyncSession) -> dict:
    return {
        "search_volume": {
            "total":    await _total_searches(db),
            "last_24h": await _searches_today(db),
        },
        "top_queries":        await _top_queries(db),
        "feedback":           await _feedback_summary(db),
        "cuisine_preferences": {
            "most_liked":   await _top_cuisines(db, signal=1),
            "most_avoided": await _top_cuisines(db, signal=-1),
        },
    }


async def log_search(
    db: AsyncSession,
    query: str,
    result_count: int,
    user_id: str = None,
) -> None:
    """Write one search event. Fire-and-forget — errors are printed, not raised."""
    try:
        db.add(SearchLog(
            user_id=user_id,
            query=query.strip(),
            result_count=result_count,
        ))
        await db.commit()
    except Exception as e:
        print(f"[analytics] log_search error: {e}")


async def _total_searches(db: AsyncSession) -> int:
    result = await db.execute(select(func.count()).select_from(SearchLog))
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
    return [{"query": row.query, "count": row.count} for row in result.all()]


async def _feedback_summary(db: AsyncSession) -> dict:
    result = await db.execute(select(UserFeedback))
    all_fb = result.scalars().all()
    total  = len(all_fb)
    likes  = sum(1 for f in all_fb if f.signal == 1)
    return {
        "total":        total,
        "likes":        likes,
        "dislikes":     total - likes,
        "like_rate_pct": round(likes / total * 100, 1) if total else 0.0,
    }


async def _top_cuisines(db: AsyncSession, signal: int, limit: int = 10) -> list[dict]:
    result = await db.execute(
        select(UserFeedback.cuisine, func.count().label("count"))
        .where(UserFeedback.signal == signal)
        .group_by(UserFeedback.cuisine)
        .order_by(func.count().desc())
        .limit(limit)
    )
    return [{"cuisine": row.cuisine, "count": row.count} for row in result.all()]