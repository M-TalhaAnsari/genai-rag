"""
mcp_service/mcp_server.py
--------------------------
FastMCP server that exposes the Connoisseur backend as MCP tools.

Any MCP-compatible client can connect to this server and call:
  - search_restaurants    : hybrid BM25 + ChromaDB + RRF search
  - get_recommendations   : full 6-agent workflow with reasoning
  - submit_feedback       : thumbs up/down → user profile update
  - get_user_profile      : read a user's preference profile
  - get_analytics         : search volume + cuisine stats

Resource:
  - restaurant://stats    : live counts from PostgreSQL + ChromaDB

Run standalone:
    python mcp_service/mcp_server.py

Or via MCP client (stdio transport):
    The client launches this as a subprocess automatically.
"""

import asyncio
import json
import sys
import os

# Add project root to path so backend imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from backend.retrieving import retrieval as retrieval_module
from backend.retrieving import vector_store, bm25_store
from backend.agents.workflow import run_recommendation_workflow
from backend.core.database import AsyncSessionLocal, engine, Base
from backend.services import feedback_service as feedback_module
from backend.services import analytics_service as analytics_module
from sqlalchemy import select
from backend.models.schemas import Restaurant

# ── Server init ────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="Connoisseur",
    instructions=(
        "You are connected to the Connoisseur restaurant discovery system for Pakistan. "
        "Use search_restaurants for quick keyword/semantic search. "
        "Use get_recommendations for personalised results with agent reasoning. "
        "Use submit_feedback to record user likes/dislikes and improve future results. "
        "Cities covered: Lahore, Islamabad, Karachi, Rawalpindi."
    )
)


# ── Helper: run async functions from sync MCP tools ───────────────────────

def _run(coro):
    """Run an async coroutine from a synchronous MCP tool handler."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Tool 1: search_restaurants ─────────────────────────────────────────────

@mcp.tool()
def search_restaurants(
    query: str,
    top_k: int = 5,
    user_id: str = None
) -> str:
    """
    Search for restaurants using hybrid semantic + keyword retrieval.

    Combines ChromaDB dense vector search with BM25 keyword search,
    fused using Reciprocal Rank Fusion (RRF).

    Args:
        query:   Natural language search query.
                 Examples: "biryani in Lahore", "rooftop cafe Islamabad",
                           "cheap fast food Karachi", "family restaurant Rawalpindi"
        top_k:   Number of results to return (1-20, default 5).
        user_id: Optional. Pass to apply personalisation from feedback history.

    Returns:
        JSON string with ranked restaurant results.
        Each result includes: name, cuisine, city, rrf_score,
        dense_rank, sparse_rank, and personalisation_boost if user_id provided.
    """
    if vector_store.collection_count() == 0:
        return json.dumps({
            "error": "No restaurants indexed yet.",
            "hint": "Run POST /restaurant-sync or POST /load-apify first."
        })

    top_k = max(1, min(20, top_k))
    results = retrieval_module.hybrid_search(query=query, top_k=top_k)

    # Apply personalisation if user_id provided
    personalised = False
    if user_id:
        async def _get_profile():
            async with AsyncSessionLocal() as db:
                return await feedback_module.get_profile(db, user_id)

        profile = _run(_get_profile())
        if profile:
            results = feedback_module.apply_profile_boost(results, profile)
            personalised = True

    return json.dumps({
        "query": query,
        "result_count": len(results),
        "personalised": personalised,
        "results": results
    }, indent=2)


# ── Tool 2: get_recommendations ────────────────────────────────────────────

@mcp.tool()
def get_recommendations(
    query: str,
    user_id: str = None,
    top_k: int = 5
) -> str:
    """
    Get AI-powered restaurant recommendations using the full 6-agent workflow.

    Unlike search_restaurants (pure retrieval), this tool runs:
      - Agent 1: analyses user profile and dining personality
      - Agent 2: retrieves and filters candidates
      - Agent 3: trend analysis (parallel)
      - Agent 4: food style matching (parallel)
      - Agent 5: nutrition/dietary check (parallel)
      - Agent 6: final reranking with written reasoning per result

    Each recommendation includes a specific 2-3 sentence explanation
    of why it was chosen for this user and query.

    Args:
        query:   What the user is looking for.
                 Examples: "best karahi in Lahore", "date night restaurant Islamabad"
        user_id: Optional. Enables personalisation from feedback history.
        top_k:   Number of final recommendations (1-10, default 5).

    Returns:
        JSON with ranked recommendations, each with a "reasoning" field.
        Also includes debug info: profile_summary, candidates_after_filter.

    Note: This takes 10-30 seconds due to the multi-agent pipeline.
    Use search_restaurants for faster results.
    """
    if vector_store.collection_count() == 0:
        return json.dumps({
            "error": "No restaurants indexed yet.",
            "hint": "Run POST /restaurant-sync or POST /load-apify first."
        })

    # Load user profile if provided
    profile = None
    if user_id:
        async def _get_profile():
            async with AsyncSessionLocal() as db:
                return await feedback_module.get_profile(db, user_id)
        profile = _run(_get_profile())

    # Run the full agent workflow
    result_state = run_recommendation_workflow(
        query=query,
        user_id=user_id or "anonymous",
        profile=profile
    )

    recommendations = result_state.get("final_recommendations", [])[:top_k]

    # Log to analytics
    async def _log():
        async with AsyncSessionLocal() as db:
            await analytics_module.log_search(
                db=db,
                query=query,
                result_count=len(recommendations),
                user_id=user_id
            )
    _run(_log())

    return json.dumps({
        "query": query,
        "user_id": user_id,
        "personalised": profile is not None,
        "result_count": len(recommendations),
        "recommendations": recommendations,
        "debug": {
            "profile_summary": result_state.get("profile_summary", ""),
            "candidates_after_filter": len(result_state.get("filtered_candidates", []))
        }
    }, indent=2)


# ── Tool 3: submit_feedback ────────────────────────────────────────────────

@mcp.tool()
def submit_feedback(
    user_id: str,
    restaurant_id: int,
    restaurant_name: str,
    cuisine: str,
    city: str,
    signal: int,
    query: str = None
) -> str:
    """
    Record a user's thumbs up or thumbs down for a restaurant.

    Automatically recomputes the user's preference profile after saving.
    Future calls to search_restaurants or get_recommendations with this
    user_id will reflect the updated preferences immediately.

    Args:
        user_id:         Unique identifier for the user.
        restaurant_id:   The restaurant's ID from search results.
        restaurant_name: Name of the restaurant.
        cuisine:         Cuisine type of the restaurant.
        city:            City of the restaurant.
        signal:          1 = thumbs up (liked), -1 = thumbs down (disliked).
        query:           Optional. The search query that produced this result.

    Returns:
        JSON with confirmation and the updated user profile summary.
    """
    if signal not in (1, -1):
        return json.dumps({
            "error": "signal must be 1 (thumbs up) or -1 (thumbs down)"
        })

    async def _save():
        async with AsyncSessionLocal() as db:
            await feedback_module.save_feedback(
                db=db,
                user_id=user_id,
                restaurant_id=restaurant_id,
                restaurant_name=restaurant_name,
                cuisine=cuisine,
                city=city,
                signal=signal,
                query=query
            )
            return await feedback_module.get_profile(db, user_id)

    profile = _run(_save())

    return json.dumps({
        "message": "Feedback saved. Profile updated.",
        "user_id": user_id,
        "signal": "liked" if signal == 1 else "disliked",
        "restaurant": restaurant_name,
        "updated_profile": profile
    }, indent=2)


# ── Tool 4: get_user_profile ───────────────────────────────────────────────

@mcp.tool()
def get_user_profile(user_id: str) -> str:
    """
    Retrieve a user's preference profile built from their feedback history.

    Shows:
      - preferred_cuisines:  cuisines they have liked (most frequent first)
      - avoided_cuisines:    cuisines they have disliked
      - preferred_cities:    cities of liked restaurants
      - feedback_count:      total thumbs up + thumbs down recorded

    Args:
        user_id: The user's unique identifier.

    Returns:
        JSON profile dict, or an error if no feedback exists yet.
    """
    async def _get():
        async with AsyncSessionLocal() as db:
            return await feedback_module.get_profile(db, user_id)

    profile = _run(_get())

    if not profile:
        return json.dumps({
            "error": f"No profile found for '{user_id}'.",
            "hint": "Submit feedback first using submit_feedback."
        })

    return json.dumps(profile, indent=2)


# ── Tool 5: get_analytics ──────────────────────────────────────────────────

@mcp.tool()
def get_analytics() -> str:
    """
    Get aggregated analytics across all users and searches.

    Returns:
      - search_volume:       total searches + last 24h count
      - top_queries:         most frequently searched terms
      - feedback:            total likes/dislikes + like rate percentage
      - cuisine_preferences: most liked and most avoided cuisine types

    Useful for understanding what users are looking for and
    which cuisines are most/least popular.
    """
    async def _get():
        async with AsyncSessionLocal() as db:
            return await analytics_module.get_analytics(db)

    result = _run(_get())
    return json.dumps(result, indent=2)


# ── Resource: restaurant://stats ───────────────────────────────────────────

@mcp.resource("restaurant://stats")
def get_stats() -> str:
    """
    Live system statistics.

    Returns counts of:
      - Total restaurants in PostgreSQL
      - Restaurants with embeddings in ChromaDB
      - BM25 index status
      - Cities covered
    """
    async def _count():
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Restaurant))
            all_recs = result.scalars().all()
            total = len(all_recs)
            embedded = sum(1 for r in all_recs if r.is_embedded)
            cities = list({r.city for r in all_recs})
            return total, embedded, cities

    total, embedded, cities = _run(_count())

    return json.dumps({
        "postgresql": {
            "total_restaurants": total,
            "embedded_restaurants": embedded,
            "not_embedded": total - embedded
        },
        "chromadb": {
            "vector_count": vector_store.collection_count()
        },
        "bm25": {
            "index_exists": bm25_store.index_exists()
        },
        "cities_covered": sorted(cities)
    }, indent=2)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
