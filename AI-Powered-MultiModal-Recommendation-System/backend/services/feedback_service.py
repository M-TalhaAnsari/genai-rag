"""
backend/services/feedback_service.py
--------------------------------------
Save feedback signals and recompute the user preference profile.

Flow:
  save_feedback()
      → insert UserFeedback row
      → call recompute_profile()

  recompute_profile()
      → load all signals for user
      → build liked/avoided cuisine lists (by frequency)
      → compute mean embedding of liked restaurants (preference vector)
      → upsert UserProfile row

  get_profile()
      → return UserProfile as clean dict (or None)

  apply_profile_boost()
      → adjust rrf_score of search results using liked/avoided cuisines
      → re-sort results list
"""

import json
from collections import Counter

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import UserFeedback, UserProfile


# ── Write ───────────────────────────────────────────────────────────────────

async def save_feedback(
    db: AsyncSession,
    user_id: str,
    restaurant_id: int,
    restaurant_name: str,
    cuisine: str,
    city: str,
    signal: int,
    query: str = None,
) -> None:
    """Persist one signal and recompute the profile."""
    if signal not in (1, -1):
        raise ValueError("signal must be 1 (like) or -1 (dislike)")

    db.add(UserFeedback(
        user_id=user_id,
        restaurant_id=restaurant_id,
        restaurant_name=restaurant_name,
        cuisine=cuisine,
        city=city,
        signal=signal,
        query=query,
    ))
    await db.commit()
    await recompute_profile(db, user_id)


async def recompute_profile(db: AsyncSession, user_id: str) -> UserProfile | None:
    """Rebuild preference profile from full feedback history."""
    result = await db.execute(
        select(UserFeedback).where(UserFeedback.user_id == user_id)
    )
    all_fb = result.scalars().all()
    if not all_fb:
        return None

    liked    = [f for f in all_fb if f.signal == 1]
    disliked = [f for f in all_fb if f.signal == -1]

    liked_cuisines   = _ranked_list([f.cuisine for f in liked])
    avoided_cuisines = _ranked_list([f.cuisine for f in disliked])

    # A later like overrides an earlier dislike for the same cuisine
    net_liked   = [c for c in liked_cuisines if c not in avoided_cuisines]
    net_avoided = [c for c in avoided_cuisines if c not in liked_cuisines]
    preferred_cities = _ranked_list([f.city for f in liked])

    # Mean embedding of liked restaurants
    preference_vector: list[float] = []
    if liked:
        from backend.embedder import embed_restaurant
        vecs = []
        for f in liked:
            vec, _ = embed_restaurant(f.restaurant_name, f.cuisine, f.city)
            vecs.append(vec)
        mean = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        preference_vector = mean.tolist()

    # Upsert UserProfile
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalars().first()

    if profile:
        profile.preferred_cuisines = json.dumps(net_liked)
        profile.avoided_cuisines   = json.dumps(net_avoided)
        profile.preferred_cities   = json.dumps(preferred_cities)
        profile.preference_vector  = json.dumps(preference_vector)
        profile.feedback_count     = len(all_fb)
    else:
        profile = UserProfile(
            user_id=user_id,
            preferred_cuisines=json.dumps(net_liked),
            avoided_cuisines=json.dumps(net_avoided),
            preferred_cities=json.dumps(preferred_cities),
            preference_vector=json.dumps(preference_vector),
            feedback_count=len(all_fb),
        )
        db.add(profile)

    await db.commit()
    await db.refresh(profile)
    return profile


# ── Read ────────────────────────────────────────────────────────────────────

async def get_profile(db: AsyncSession, user_id: str) -> dict | None:
    """Return UserProfile as a clean dict, or None if no feedback exists."""
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalars().first()
    if not profile:
        return None

    return {
        "user_id":                user_id,
        "preferred_cuisines":     json.loads(profile.preferred_cuisines),
        "avoided_cuisines":       json.loads(profile.avoided_cuisines),
        "preferred_cities":       json.loads(profile.preferred_cities),
        "preference_vector_length": len(json.loads(profile.preference_vector)),
        "feedback_count":         profile.feedback_count,
        "updated_at":             profile.updated_at.isoformat() if profile.updated_at else None,
    }


# ── Personalisation ─────────────────────────────────────────────────────────

def apply_profile_boost(results: list[dict], profile: dict | None) -> list[dict]:
    """
    Boost/penalise search results based on the user's cuisine preferences.

    Adds a personalisation_boost field and adjusts rrf_score (+0.05 / -0.05).
    Does not remove results — only reorders them.
    Anonymous users (profile=None) get boost=0 on every result.
    """
    if not profile:
        for r in results:
            r["personalisation_boost"] = 0.0
        return results

    preferred = set(profile.get("preferred_cuisines", []))
    avoided   = set(profile.get("avoided_cuisines", []))

    for r in results:
        boost   = 0.0
        cuisine = r.get("cuisine", "")
        if cuisine in preferred:
            boost += 0.05
        if cuisine in avoided:
            boost -= 0.05
        r["personalisation_boost"] = round(boost, 4)
        r["rrf_score"] = round(r.get("rrf_score", 0) + boost, 6)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ranked_list(items: list[str]) -> list[str]:
    """Unique items sorted by frequency, most common first."""
    return [item for item, _ in Counter(items).most_common()]