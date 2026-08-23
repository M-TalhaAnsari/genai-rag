"""
backend/routers/analytics.py
------------------------------
GET /analytics — system-wide search and feedback analytics
GET /vector-stats — ChromaDB + BM25 debug info
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.services import analytics_service as analytics_module
from backend.retrieving import vector_store, bm25_store
from backend.memory import session as session_memory

from fastapi import APIRouter, Depends
from backend.core.security import get_current_user, require_admin
from backend.models.db_models import User

router = APIRouter(tags=["analytics"])


@router.get("/analytics", dependencies=[Depends(require_admin)])
async def get_analytics(db: AsyncSession = Depends(get_db)):
    """
    Aggregated analytics across all users and searches.

    Returns:
      search_volume       — total + last 24h search counts
      top_queries         — most searched terms
      feedback            — likes, dislikes, like rate %
      cuisine_preferences — most liked and most avoided cuisines
    """
    return await analytics_module.get_analytics(db)


@router.get("/vector-stats", dependencies=[Depends(require_admin)])
async def vector_stats():
    """
    Debug: ChromaDB vector count, BM25 index status, active session count.
    Useful after a sync to confirm embeddings were stored correctly.
    """
    return {
        "chroma_restaurants":   vector_store.collection_count(),
        "chroma_reviews":       vector_store.review_collection_count(),
        "bm25_index_exists":    bm25_store.index_exists(),
        "active_sessions":      await session_memory.active_sessions(),
    }