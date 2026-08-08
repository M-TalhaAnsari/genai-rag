"""
backend/routers/memory.py
--------------------------
GET /memory/{user_id} — short-term + long-term memory context
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.memory.long_term import get_full_memory_context
from backend.memory import session as session_memory

router = APIRouter(tags=["memory"])


@router.get("/memory/{user_id}")
async def get_user_memory(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Full memory context for a user.

    Returns:
      long_term.memory_summary   — LLM-written 100-word summary (recomputed every 10 turns)
      long_term.recent_history   — last 10 conversation turns from PostgreSQL
      long_term.recent_searches  — deduplicated past search queries
      session.recent_queries     — queries this session (RAM only)
      session.shown_restaurants  — restaurants shown this session
      session.turn_count         — turns in current session
    """
    long_term = await get_full_memory_context(db, user_id)
    session   = session_memory.get_session_summary(user_id)

    return {
        "user_id":   user_id,
        "long_term": long_term,
        "session":   session,
    }