"""
backend/routers/feedback.py
-----------------------------
POST /feedback             — thumbs up/down → profile update
GET  /profile/{user_id}    — user preference profile
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.models.schemas import FeedbackRequest
from backend.services import feedback_service as feedback_module

from fastapi import APIRouter, Depends
from backend.core.security import get_current_user
from backend.models.db_models import User

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db), current_user: User=Depends(get_current_user)
):
    """
    Save a thumbs up (signal=1) or thumbs down (signal=-1).
    Automatically recomputes the user's preference profile.
    Future /search?user_id=... and /recommend calls reflect this immediately.
    """
    if request.signal not in (1, -1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="signal must be 1 (thumbs up) or -1 (thumbs down)."
        )

    await feedback_module.save_feedback(
        db=db,
        user_id=current_user.id,
        restaurant_id=request.restaurant_id,
        restaurant_name=request.restaurant_name,
        cuisine=request.cuisine,
        city=request.city,
        signal=request.signal,
        query=request.query
    )

    profile = await feedback_module.get_profile(db, request.user_id)

    return {
        "message":        "Feedback saved. Profile updated.",
        "user_id":        request.user_id,
        "signal":         request.signal,
        "updated_profile": profile
    }


@router.get("/profile/{user_id}")
async def get_user_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db), current_user: User= Depends(get_current_user)
):
    """
    Read the preference profile for a user derived from their feedback history.

    Returns:
      preferred_cuisines   — most liked cuisines (most frequent first)
      avoided_cuisines     — most disliked cuisines
      preferred_cities     — cities of liked restaurants
      feedback_count       — total signals recorded

    404 if the user has no feedback history yet.
    """
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view another user's profile."
        )
    
    profile = await feedback_module.get_profile(db, user_id)

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile for '{user_id}'. Submit feedback first."
        )

    return profile