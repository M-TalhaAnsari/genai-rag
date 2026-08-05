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
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, Depends, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

from backend.db.database import AsyncSessionLocal, engine, Base
from backend.model.models import Restaurant, Review
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

from pydantic import BaseModel, Field

from backend.model.schemas import (
    RestaurantRequest, FeedbackRequest,RestaurantDetail,
    RecommendRequest, HealthResponse, SyncResponse,
    SearchResult, SearchResponse, ReviewOut,
    GenerateMessageRequest, ContactRequest
    
)

app = FastAPI(
    title="Connoisseur Restaurant API",
    description="AI-powered restaurant discovery for Pakistan",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# n8n-triggered fully automated Apify sync (no manual export/import)
from n8n.api.routers import router as apify_automation_router
app.include_router(apify_automation_router)


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
        status="ok",
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




@app.post("/generate-message")
async def generate_message(request: GenerateMessageRequest):
    """
    Generate a draft contact message from the user to a restaurant.

    Called when the user clicks 'Select' on a restaurant card.
    Returns a draft message the user can approve or edit before sending.

    The message tone adapts to the contact channel:
      email    → formal
      whatsapp → casual and brief
      booking  → concise reservation request
    """
    from n8n.backend.contact import generate_contact_message

    # Determine best contact method based on what restaurant has
    result = await _get_restaurant_contact_method(request.restaurant_id)
    contact_method = result.get("method", request.contact_method)

    draft = generate_contact_message(
        restaurant_name=request.restaurant_name,
        cuisine=request.cuisine,
        city=request.city,
        user_name=request.user_name,
        user_query=request.user_query,
        contact_method=contact_method
    )

    return {
        "draft_message":   draft,
        "contact_method":  contact_method,
        "restaurant_name": request.restaurant_name
    }


@app.post("/contact-restaurant")
async def contact_restaurant(
    request: ContactRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Send the approved message to the restaurant via n8n.

    n8n receives the full payload and routes to:
      - Email    if restaurant.email is set
      - WhatsApp if restaurant.phone is set
      - Booking  if restaurant.website is set

    The routing logic lives in n8n — this endpoint just fires the webhook.
    """
    from n8n.backend.contact import trigger_n8n_contact

    result = trigger_n8n_contact(
        restaurant_id=request.restaurant_id,
        restaurant_name=request.restaurant_name,
        cuisine=request.cuisine,
        city=request.city,
        email=request.email,
        phone=request.phone,
        website=request.website,
        message=request.message,
        user_name=request.user_name,
        user_query=request.user_query
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.get("error", "n8n webhook failed.")
        )

    return result


# ── Internal helper ────────────────────────────────────────────────────────

async def _get_restaurant_contact_method(restaurant_id: int) -> dict:
    """
    Look up a restaurant's available contact channels and return
    the best one for the draft message tone.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Restaurant).where(Restaurant.id == restaurant_id)
        )
        r = result.scalars().first()

    if not r:
        return {"method": "email"}

    if r.email:
        return {"method": "email",    "value": r.email}
    if r.phone:
        return {"method": "whatsapp", "value": r.phone}
    if r.website:
        return {"method": "booking",  "value": r.website}

    return {"method": "email"}


# ── Review summarisation endpoint ──────────────────────────────────────────

@app.get("/restaurants/{restaurant_id}/review-summary")
async def get_review_summary_endpoint(restaurant_id: int):
    """
    Fetch the stored review summary for one restaurant.
    Includes quality dimensions, recency info, fake signal warnings.
    Returns 404 if no summary has been generated yet.
    """
    from backend.retrieving.vector_store import get_review_summary
    data = get_review_summary(restaurant_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No review summary for restaurant {restaurant_id}. "
                   f"Call POST /restaurants/{restaurant_id}/summarise-reviews first."
        )
    return data


@app.post("/restaurants/{restaurant_id}/summarise-reviews")
async def summarise_restaurant_reviews(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a cautious review summary for a restaurant and embed it.

    Call this after loading Apify data which includes raw reviews.
    The summary is stored in the ChromaDB "restaurant_reviews" collection.

    The summary:
      - Only states what MULTIPLE reviews agree on
      - Flags mixed/polarised reviews explicitly
      - Never fabricates details
      - Is shown to users WITH a disclaimer

    Can also be called as a batch via /summarise-all-reviews.
    """
    from backend.data_loader.review_summariser import summarise_reviews
    from backend.retrieving.vector_store import upsert_review_summary

    # Load restaurant
    r_result = await db.execute(
        select(Restaurant).where(Restaurant.id == restaurant_id)
    )
    restaurant = r_result.scalars().first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found.")

    # Load reviews
    rv_result = await db.execute(
        select(Review).where(Review.restaurant_id == restaurant_id)
    )
    reviews = rv_result.scalars().all()
    review_dicts = [
        {
            "rating":         r.rating,
            "text":           r.text,
            "published_date": r.published_date,
            "source":         r.source
        }
        for r in reviews
    ]

    # Generate cautious summary
    summary_data = summarise_reviews(
        restaurant_name=restaurant.name,
        cuisine=restaurant.cuisine,
        reviews=review_dicts
    )

    # Embed and store
    upsert_review_summary(
        restaurant_id=restaurant_id,
        restaurant_name=restaurant.name,
        cuisine=restaurant.cuisine,
        city=restaurant.city,
        **summary_data
    )

    return {
        "restaurant_id":   restaurant_id,
        "restaurant_name": restaurant.name,
        "summary_data":    summary_data
    }


@app.post("/summarise-all-reviews")
async def summarise_all_reviews(db: AsyncSession = Depends(get_db)):
    """
    Batch generate review summaries for all restaurants that have reviews
    but no summary yet in ChromaDB.

    Run this once after /load-apify to populate the review collection.
    Takes time proportional to number of restaurants × LLM latency.
    """
    from backend.data_loader.review_summariser import summarise_reviews
    from backend.retrieving.vector_store import upsert_review_summary, get_review_summary

    r_result = await db.execute(select(Restaurant))
    restaurants = r_result.scalars().all()

    done, skipped, errors = 0, 0, []

    for restaurant in restaurants:
        # Skip if already summarised
        if get_review_summary(restaurant.id):
            skipped += 1
            continue

        rv_result = await db.execute(
            select(Review).where(Review.restaurant_id == restaurant.id)
        )
        reviews = rv_result.scalars().all()

        if not reviews:
            skipped += 1
            continue

        try:
            review_dicts = [
                {"rating": r.rating, "text": r.text,
                 "published_date": r.published_date, "source": r.source}
                for r in reviews
            ]
            summary_data = summarise_reviews(
                restaurant_name=restaurant.name,
                cuisine=restaurant.cuisine,
                reviews=review_dicts
            )
            upsert_review_summary(
                restaurant_id=restaurant.id,
                restaurant_name=restaurant.name,
                cuisine=restaurant.cuisine,
                city=restaurant.city,
                **summary_data
            )
            done += 1
        except Exception as e:
            errors.append(f"{restaurant.name}: {e}")

    return {
        "summarised": done,
        "skipped":    skipped,
        "errors":     errors[:10]
    }


# ── Search by review sentiment ─────────────────────────────────────────────

@app.get("/search/by-review")
async def search_by_review(
    q: str = Query(..., min_length=1,
                   description="e.g. 'great service and clean tables'"),
    top_k: int = Query(default=5, ge=1, le=20),
    min_confidence: str = Query(
        default="low",
        description="Filter by confidence level: low | medium | high"
    )
):
    """
    Search restaurants by their review summary content.

    Unlike /search which matches on restaurant identity (name, cuisine, city),
    this endpoint matches on what customers actually said — food quality,
    cleanliness, service, vibe, and menu variety.

    Examples:
      /search/by-review?q=clean tables and fast service
      /search/by-review?q=great biryani fresh ingredients
      /search/by-review?q=quiet atmosphere good for families
      /search/by-review?q=avoid dirty kitchen

    Each result includes:
      - review_summary: the cautious 2-sentence summary
      - confidence: low | medium | high (based on review count)
      - avg_rating and weighted_rating (recency-weighted)
      - disclaimer and fake signal warnings if applicable
      - most_recent_review date so you know how fresh the data is
    """
    from backend.retrieving.vector_store import search_by_review_sentiment, review_collection_count

    if review_collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No review summaries indexed yet. "
                "Run POST /summarise-all-reviews after loading restaurant data."
            )
        )

    confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    min_rank = confidence_rank.get(min_confidence, 1)

    results = search_by_review_sentiment(query=q, top_k=top_k * 2)

    # Filter by minimum confidence
    results = [
        r for r in results
        if confidence_rank.get(r.get("confidence", "none"), 0) >= min_rank
    ][:top_k]

    return {
        "query":        q,
        "result_count": len(results),
        "source":       "review_summaries",
        "results":      results
    }


# ── Hybrid search including review signal ──────────────────────────────────

@app.get("/search/full")
async def search_full(
    q: str          = Query(..., min_length=1),
    top_k: int      = Query(default=5, ge=1, le=20),
    user_id: Optional[str] = Query(default=None),
    w_identity: float = Query(default=0.6, ge=0.0, le=1.0,
                              description="Weight for identity search (name/cuisine/city)"),
    w_review: float   = Query(default=0.4, ge=0.0, le=1.0,
                              description="Weight for review sentiment search"),
    db: AsyncSession = Depends(get_db)
):
    """
    Three-signal hybrid search:
      Signal 1: BM25 keyword search        (identity)
      Signal 2: ChromaDB dense search      (identity)
      Signal 3: ChromaDB review sentiment  (what customers said)

    Signals 1+2 are fused with RRF then combined with Signal 3
    using configurable weights.

    Default weights: 60% identity, 40% review sentiment.
    Adjust w_identity and w_review to change the balance.

    Restaurants with fake review warnings have their review
    signal weight automatically reduced by 50%.

    Examples:
      /search/full?q=best biryani Lahore
      /search/full?q=clean family restaurant Islamabad&w_review=0.6
      /search/full?q=rooftop cafe Karachi&user_id=ahmed_123
    """
    from backend.retrieving.vector_store import (
        search_by_review_sentiment, review_collection_count
    )

    if vector_store.collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No restaurants indexed. Run /restaurant-sync first."
        )

    # Signal 1+2: existing hybrid retrieval (BM25 + dense, RRF fused)
    identity_results = retrieval.hybrid_search(query=q, top_k=top_k * 2)

    # Signal 3: review sentiment (only if collection exists)
    review_results = []
    if review_collection_count() > 0:
        review_results = search_by_review_sentiment(query=q, top_k=top_k * 2)

    # Merge by restaurant_id
    scores: dict[int, dict] = {}

    for rank, r in enumerate(identity_results, start=1):
        rid = r["restaurant_id"]
        scores[rid] = {
            "restaurant_id":   rid,
            "name":            r["name"],
            "cuisine":         r["cuisine"],
            "city":            r["city"],
            "identity_score":  round(w_identity * (1.0 / (60 + rank)), 6),
            "review_score":    0.0,
            "combined_score":  0.0,
            "review_summary":  None,
            "review_disclaimer": None,
            "has_fake_signals": False,
            "most_recent_review": None,
        }

    for rank, r in enumerate(review_results, start=1):
        rid = r["restaurant_id"]

        # Reduce review weight for restaurants with fake signals
        effective_w = w_review * (0.5 if r.get("has_fake_signals") else 1.0)
        review_contribution = effective_w * (1.0 / (60 + rank))

        if rid in scores:
            scores[rid]["review_score"]     = round(review_contribution, 6)
            scores[rid]["review_summary"]   = r.get("review_summary")
            scores[rid]["review_disclaimer"] = r.get("disclaimer")
            scores[rid]["has_fake_signals"] = r.get("has_fake_signals", False)
            scores[rid]["most_recent_review"] = r.get("most_recent_review")
        else:
            scores[rid] = {
                "restaurant_id":    rid,
                "name":             r["name"],
                "cuisine":          r["cuisine"],
                "city":             r["city"],
                "identity_score":   0.0,
                "review_score":     round(review_contribution, 6),
                "combined_score":   0.0,
                "review_summary":   r.get("review_summary"),
                "review_disclaimer": r.get("disclaimer"),
                "has_fake_signals": r.get("has_fake_signals", False),
                "most_recent_review": r.get("most_recent_review"),
            }

    # Compute combined score and sort
    for rid in scores:
        scores[rid]["combined_score"] = round(
            scores[rid]["identity_score"] + scores[rid]["review_score"], 6
        )

    ranked = sorted(
        scores.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )[:top_k]

    # Personalisation
    personalised = False
    if user_id:
        profile = await feedback_module.get_profile(db, user_id)
        if profile:
            ranked = feedback_module.apply_profile_boost(ranked, profile)
            personalised = True
        session_memory.add_query(user_id, q)

    await analytics_module.log_search(
        db=db, query=q,
        result_count=len(ranked),
        user_id=user_id
    )

    return {
        "query":        q,
        "result_count": len(ranked),
        "personalised": personalised,
        "weights":      {"identity": w_identity, "review": w_review},
        "results":      ranked
    }
