"""
backend/routers/restaurants.py
--------------------------------
Restaurant read endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.core.security import get_current_user, require_admin
from backend.models.db_models import Restaurant, Review, User
from backend.models.schemas import RestaurantDetail, ReviewOut
from backend.retrieving.vector_store import get_restaurant_images as _get_images
from backend.retrieving.vector_store import get_review_summary as _get_review_summary
from backend.retrieving.vector_store import upsert_review_summary
from backend.data_loader.review_summariser import summarise_reviews as _summarise

router = APIRouter(prefix="/restaurants", tags=["restaurants"])


@router.get("")
async def list_restaurants(
    city: str | None = Query(default=None),
    cuisine: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List restaurants with optional city/cuisine filter and pagination."""
    q = select(Restaurant)
    if city:
        q = q.where(Restaurant.city.ilike(f"%{city}%"))
    if cuisine:
        q = q.where(Restaurant.cuisine.ilike(f"%{cuisine}%"))
    q = q.offset(offset).limit(limit)

    result = await db.execute(q)
    restaurants = result.scalars().all()

    return {
        "total": len(restaurants),
        "offset": offset,
        "limit": limit,
        "restaurants": [
            {
                "id": r.id,
                "name": r.name,
                "cuisine": r.cuisine,
                "city": r.city,
                "area": r.area,
                "rating": r.rating,
                "price_level": r.price_level,
                "address": r.address,
                "phone": r.phone,
                "website": r.website,
                "has_email": bool(r.email),
                "has_menu": bool(r.menu_url),
                "is_embedded": r.is_embedded,
            }
            for r in restaurants
        ]
    }


@router.get("/{restaurant_id}", response_model=RestaurantDetail)
async def get_restaurant(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
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


@router.get("/{restaurant_id}/reviews")
async def get_reviews(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
                "source": r.source,
            }
            for r in reviews
        ]
    }


@router.get("/{restaurant_id}/review-summary")
async def get_review_summary_route(restaurant_id: int):
    """Stored review quality summary from ChromaDB."""
    data = _get_review_summary(restaurant_id)   # was self-calling before the fix
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No review summary for restaurant {restaurant_id}. "
                   f"Call POST /restaurants/{restaurant_id}/summarise-reviews first."
        )
    return data


@router.post("/{restaurant_id}/summarise-reviews", dependencies=[Depends(require_admin)])
async def summarise_reviews(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a cautious review summary for one restaurant and embed it.
    Admin-only: this triggers an LLM call and writes to ChromaDB.
    """
    r_result = await db.execute(
        select(Restaurant).where(Restaurant.id == restaurant_id)
    )
    restaurant = r_result.scalars().first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found.")

    rv_result = await db.execute(
        select(Review).where(Review.restaurant_id == restaurant_id)
    )
    reviews = [
        {"rating": r.rating, "text": r.text,
         "published_date": r.published_date, "source": r.source}
        for r in rv_result.scalars().all()
    ]

    summary_data = _summarise(
        restaurant_name=restaurant.name,
        cuisine=restaurant.cuisine,
        reviews=reviews
    )
    upsert_review_summary(
        restaurant_id=restaurant_id,
        restaurant_name=restaurant.name,
        cuisine=restaurant.cuisine,
        city=restaurant.city,
        **summary_data
    )
    return {"restaurant_id": restaurant_id, "summary_data": summary_data}


@router.get("/{restaurant_id}/images")
async def get_restaurant_images(restaurant_id: int):
    """All embedded images for one restaurant."""
    images = _get_images(restaurant_id)
    return {"restaurant_id": restaurant_id, "image_count": len(images), "images": images}