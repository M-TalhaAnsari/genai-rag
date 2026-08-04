"""
backend/main.py
----------------
FastAPI application — production-hardened.

All endpoints have:
  - Typed Pydantic request + response models
  - Proper HTTP status codes
  - Error handling with meaningful messages
  - Session + long-term memory integrated
  - Analytics logging

Endpoints
---------
GET  /                          Health check
GET  /fetch-restaurants         Fetch from OSM + Foursquare (n8n calls this)
POST /restaurant-sync           Ingest → PostgreSQL + ChromaDB + BM25
POST /load-apify                One-time Apify bulk load
GET  /restaurants               List all restaurants
GET  /restaurants/{id}          Single restaurant detail with reviews
GET  /restaurants/{id}/reviews  All reviews for a restaurant
GET  /search                    Hybrid search with personalisation
POST /recommend                 Full 6-agent workflow (streaming SSE)
POST /feedback                  Thumbs up/down → profile update
GET  /profile/{user_id}         User preference profile
GET  /memory/{user_id}          User conversation memory
GET  /analytics                 System-wide analytics
GET  /vector-stats              ChromaDB + BM25 debug info
"""

import json
import asyncio
from typing import Optional


from fastapi import FastAPI, Depends, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

from backend.db.database import AsyncSessionLocal, engine, Base
from backend.model.models import Restaurant, Review, UserFeedback, UserProfile, SearchLog
from backend.memory.long_term import (
    ConversationHistory, UserMemorySummary,
    save_conversation, get_full_memory_context
)
from backend.retrieving import vector_store, bm25_store, retrieval
from backend.data_loader.restaurant_fetcher import fetch_all_cities, TARGET_CITIES
from backend import feedback as feedback_module
from backend import analytics as analytics_module
from backend.agents.workflow import run_recommendation_workflow
from backend.memory import session as session_memory

from backend.model.schemas import (
    RestaurantRequest, FeedbackRequest,RestaurantDetail,
    RecommendRequest, HealthResponse, SyncResponse,
    SearchResult, SearchResponse, ReviewOut,
    
)
app = FastAPI(
    title="Connoisseur Restaurant API",
    description="AI-powered restaurant discovery for Pakistan",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Create all tables on startup. Safe to run multiple times."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── DB dependency ──────────────────────────────────────────────────────────

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session




# ── Health check ───────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="ok"
        version="4.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


# ── Fetch restaurants from OSM + Foursquare (n8n calls this) ──────────────

@app.get("/fetch-restaurants")
def fetch_restaurants(
    cities: Optional[str] = Query(
        default=None,
        description="Comma-separated city names. Default: all 4 cities."
    )
):
    city_list = (
        [c.strip() for c in cities.split(",") if c.strip()]
        if cities else TARGET_CITIES
    )
    try:
        restaurants = fetch_all_cities(cities=city_list)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Data fetch failed: {str(e)}"
        )
    return {
        "cities_fetched": city_list,
        "total": len(restaurants),
        "restaurants": restaurants
    }


# ── Restaurant sync ────────────────────────────────────────────────────────

@app.post("/restaurant-sync", response_model=SyncResponse)
async def restaurant_sync(
    request: RestaurantRequest,
    db: AsyncSession = Depends(get_db)
):
    saved, skipped, newly_inserted = [], [], []

    for item in request.restaurants:
        result = await db.execute(
            select(Restaurant).where(
                Restaurant.name == item.name,
                Restaurant.city == item.city
            )
        )
        if result.scalars().first():
            skipped.append(item.name)
            continue

        record = Restaurant(**item.model_dump(), is_embedded=False)
        db.add(record)
        newly_inserted.append(record)
        saved.append(item.name)

    await db.commit()

    embedded_count = 0
    for record in newly_inserted:
        await db.refresh(record)
        try:
            vector_store.upsert_restaurant(
                restaurant_id=record.id,
                name=record.name,
                cuisine=record.cuisine,
                city=record.city
            )
            record.is_embedded = True
            embedded_count += 1
        except Exception as e:
            print(f"[embed error] {record.name}: {e}")

    await db.commit()

    if newly_inserted:
        all_result = await db.execute(select(Restaurant))
        all_recs = all_result.scalars().all()
        bm25_store.build_index([
            {"id": r.id, "name": r.name, "cuisine": r.cuisine, "city": r.city}
            for r in all_recs
        ])

    return SyncResponse(
        message="Sync complete",
        inserted_count=len(saved),
        skipped_count=len(skipped),
        embedded_count=embedded_count,
        inserted=saved,
        skipped=skipped
    )


# ── Apify one-time load ────────────────────────────────────────────────────

@app.post("/load-apify")
async def load_apify():
    """
    One-time bulk load from data/apify_export.json.
    Run this once after downloading your Apify Google Maps export.
    """
    from backend.data_loader.apify_loader import load_apify_export
    result = await load_apify_export()
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    return result


# ── List all restaurants ───────────────────────────────────────────────────

@app.get("/restaurants")
async def get_restaurants(
    city:    Optional[str] = Query(default=None),
    cuisine: Optional[str] = Query(default=None),
    limit:   int           = Query(default=50, ge=1, le=500),
    offset:  int           = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List restaurants with optional city/cuisine filter and pagination."""
    query = select(Restaurant)
    if city:
        query = query.where(Restaurant.city.ilike(f"%{city}%"))
    if cuisine:
        query = query.where(Restaurant.cuisine.ilike(f"%{cuisine}%"))
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    restaurants = result.scalars().all()

    return {
        "total": len(restaurants),
        "offset": offset,
        "limit": limit,
        "restaurants": [
            {
                "id": r.id, "name": r.name, "cuisine": r.cuisine,
                "city": r.city, "rating": r.rating,
                "price_level": r.price_level, "address": r.address,
                "phone": r.phone, "website": r.website,
                "is_embedded": r.is_embedded
            }
            for r in restaurants
        ]
    }


# ── Single restaurant detail ───────────────────────────────────────────────

@app.get("/restaurants/{restaurant_id}", response_model=RestaurantDetail)
async def get_restaurant(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Full detail for one restaurant including all reviews."""
    result = await db.execute(
        select(Restaurant).where(Restaurant.id == restaurant_id)
    )
    restaurant = result.scalars().first()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found."
        )

    reviews_result = await db.execute(
        select(Review).where(Review.restaurant_id == restaurant_id)
    )
    reviews = reviews_result.scalars().all()

    return RestaurantDetail(
        **{c.name: getattr(restaurant, c.name)
           for c in Restaurant.__table__.columns},
        reviews=[
            ReviewOut(
                id=r.id,
                reviewer_name=r.reviewer_name,
                rating=r.rating,
                text=r.text,
                published_date=r.published_date,
                source=r.source
            )
            for r in reviews
        ]
    )


# ── Restaurant reviews ─────────────────────────────────────────────────────

@app.get("/restaurants/{restaurant_id}/reviews")
async def get_reviews(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """All reviews for a specific restaurant."""
    result = await db.execute(
        select(Review).where(Review.restaurant_id == restaurant_id)
    )
    reviews = result.scalars().all()

    if not reviews:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No reviews found for restaurant {restaurant_id}."
        )

    return {
        "restaurant_id": restaurant_id,
        "review_count": len(reviews),
        "reviews": [
            {
                "id": r.id,
                "reviewer_name": r.reviewer_name,
                "rating": r.rating,
                "text": r.text,
                "published_date": r.published_date,
                "source": r.source
            }
            for r in reviews
        ]
    }


# ── Hybrid search ──────────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse)
async def search(
    q:       str            = Query(..., min_length=1),
    top_k:   int            = Query(default=5, ge=1, le=20),
    user_id: Optional[str]  = Query(default=None),
    db: AsyncSession = Depends(get_db)
):
    """
    Hybrid search: ChromaDB dense + BM25 sparse → RRF fusion.
    Pass user_id to activate personalisation from feedback history.
    """
    if vector_store.collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No restaurants indexed. Run /restaurant-sync first."
        )

    results = retrieval.hybrid_search(query=q, top_k=top_k)

    personalised = False
    if user_id:
        profile = await feedback_module.get_profile(db, user_id)
        if profile:
            results = feedback_module.apply_profile_boost(results, profile)
            personalised = True
        session_memory.add_query(user_id, q)
        session_memory.add_results(user_id, results)

    await analytics_module.log_search(db=db, query=q,
                                      result_count=len(results), user_id=user_id)

    return SearchResponse(
        query=q,
        result_count=len(results),
        personalised=personalised,
        results=[SearchResult(**r) for r in results]
    )


# ── Multi-agent recommend with streaming ───────────────────────────────────

@app.post("/recommend")
async def recommend(
    request: RecommendRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Full 6-agent recommendation pipeline with Server-Sent Events streaming.

    Streams progress tokens as the workflow runs so the user sees
    updates in real time rather than waiting 30 seconds for a response.

    Event types streamed:
      data: {"event": "phase", "message": "Running trend analysis..."}
      data: {"event": "result", "recommendations": [...]}
      data: {"event": "done"}
    """
    if vector_store.collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No restaurants indexed. Run /restaurant-sync first."
        )

    profile = None
    memory_context = {}
    if request.user_id:
        profile = await feedback_module.get_profile(db, request.user_id)
        memory_context = await get_full_memory_context(db, request.user_id)
        session_memory.add_query(request.user_id, request.query)
        session_memory.add_message(request.user_id, "user", request.query)

    async def event_stream():
        # Stream phase updates
        phases = [
            "Analysing your profile...",
            "Retrieving candidates...",
            "Running trend, style and nutrition analysis...",
            "Generating personalised recommendations..."
        ]

        for phase_msg in phases:
            yield f"data: {json.dumps({'event': 'phase', 'message': phase_msg})}\n\n"
            await asyncio.sleep(0.05)

        # Run workflow in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        result_state = await loop.run_in_executor(
            None,
            lambda: run_recommendation_workflow(
                query=request.query,
                user_id=request.user_id or "anonymous",
                profile=profile
            )
        )

        recommendations = result_state.get("final_recommendations", [])[:request.top_k]

        # Save conversation to long-term memory
        if request.user_id:
            assistant_response = (
                f"I found {len(recommendations)} recommendations for: {request.query}"
            )
            await save_conversation(
                db=db,
                user_id=request.user_id,
                user_message=request.query,
                assistant_response=assistant_response,
                query=request.query
            )
            session_memory.add_message(
                request.user_id, "assistant", assistant_response
            )

        await analytics_module.log_search(
            db=db, query=request.query,
            result_count=len(recommendations),
            user_id=request.user_id
        )

        # Stream final result
        yield f"data: {json.dumps({'event': 'result', 'query': request.query, 'user_id': request.user_id, 'personalised': profile is not None, 'result_count': len(recommendations), 'recommendations': recommendations, 'debug': {'profile_summary': result_state.get('profile_summary', ''), 'candidates_after_filter': len(result_state.get('filtered_candidates', []))}})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ── Feedback ───────────────────────────────────────────────────────────────

@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    if request.signal not in (1, -1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="signal must be 1 (thumbs up) or -1 (thumbs down)."
        )

    await feedback_module.save_feedback(
        db=db,
        user_id=request.user_id,
        restaurant_id=request.restaurant_id,
        restaurant_name=request.restaurant_name,
        cuisine=request.cuisine,
        city=request.city,
        signal=request.signal,
        query=request.query
    )

    profile = await feedback_module.get_profile(db, request.user_id)

    return {
        "message": "Feedback saved. Profile updated.",
        "user_id": request.user_id,
        "signal": request.signal,
        "updated_profile": profile
    }


# ── User profile ───────────────────────────────────────────────────────────

@app.get("/profile/{user_id}")
async def get_user_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    profile = await feedback_module.get_profile(db, user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile for '{user_id}'. Submit feedback first."
        )
    return profile


# ── Memory ─────────────────────────────────────────────────────────────────

@app.get("/memory/{user_id}")
async def get_user_memory(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Return the full memory context for a user:
      - LLM-generated memory summary
      - Recent conversation history (last 10 turns)
      - Recent search queries
      - Active session summary (current session only)
    """
    long_term  = await get_full_memory_context(db, user_id)
    session    = session_memory.get_session_summary(user_id)

    return {
        "user_id":    user_id,
        "long_term":  long_term,
        "session":    session
    }


# ── Analytics ──────────────────────────────────────────────────────────────

@app.get("/analytics")
async def get_analytics(db: AsyncSession = Depends(get_db)):
    return await analytics_module.get_analytics(db)


# ── Debug ──────────────────────────────────────────────────────────────────

@app.get("/vector-stats")
def vector_stats():
    return {
        "chroma_count":    vector_store.collection_count(),
        "bm25_index_exists": bm25_store.index_exists(),
        "active_sessions": session_memory.active_sessions()
    }
