"""
backend/feedback.py
--------------------
User feedback logic: save signals, recompute preference profile.

Flow:
  User gives 👍 or 👎 on a search result
       │
       ▼
  save_feedback()  →  writes to user_feedback table
       │
       ▼
  recompute_profile()  →  reads all feedback for that user
                        →  builds preferred/avoided cuisine lists
                        →  averages embeddings of liked restaurants
                        →  writes to user_profiles table

The preference_vector stored in user_profiles is the mean of all
liked restaurant embeddings. During search, this vector is used
to re-rank results toward the user's demonstrated taste.
"""

import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.model.models import UserFeedback, UserProfile
from backend.retrieving.embedder import embed_restaurant
import numpy as np


async def save_feedback(
    db: AsyncSession,
    user_id: str,
    restaurant_id: int,
    restaurant_name: str,
    cuisine: str,
    city: str,
    signal: int,            # 1 = like, -1 = dislike
    query: str = None
) -> UserFeedback:
    """
    Save one feedback signal and trigger a profile recompute.

    Args:
        signal: 1 for thumbs up, -1 for thumbs down
        query:  The search query that produced this result (optional)
    """
    if signal not in (1, -1):
        raise ValueError("signal must be 1 (like) or -1 (dislike)")

    feedback = UserFeedback(
        user_id=user_id,
        restaurant_id=restaurant_id,
        restaurant_name=restaurant_name,
        cuisine=cuisine,
        city=city,
        signal=signal,
        query=query
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)

    # Recompute the user profile after every new signal
    await recompute_profile(db, user_id)

    return feedback


async def recompute_profile(db: AsyncSession, user_id: str) -> UserProfile:
    """
    Rebuild the user's preference profile from their full feedback history.

    Called automatically after every feedback save.
    Safe to call multiple times — always overwrites with latest state.
    """
    # Load all feedback for this user
    result = await db.execute(
        select(UserFeedback).where(UserFeedback.user_id == user_id)
    )
    all_feedback = result.scalars().all()

    if not all_feedback:
        return None

    # Separate liked and disliked
    liked = [f for f in all_feedback if f.signal == 1]
    disliked = [f for f in all_feedback if f.signal == -1]

    # Build cuisine preference lists (deduplicated, ordered by frequency)
    liked_cuisines = _ranked_list([f.cuisine for f in liked])
    avoided_cuisines = _ranked_list([f.cuisine for f in disliked])

    # Remove cuisines from avoided if user also liked them later
    # (a more recent like overrides an older dislike)
    liked_ids = {f.restaurant_id for f in liked}
    disliked_ids = {f.restaurant_id for f in disliked}
    net_liked_cuisines = [c for c in liked_cuisines if c not in avoided_cuisines]
    net_avoided_cuisines = [c for c in avoided_cuisines if c not in liked_cuisines]

    # Preferred cities from liked restaurants
    preferred_cities = _ranked_list([f.city for f in liked])

    # Preference vector: mean embedding of all liked restaurants
    preference_vector = []
    if liked:
        embeddings = []
        for f in liked:
            vec = embed_restaurant(f.restaurant_name, f.cuisine, f.city)
            embeddings.append(vec)
        mean_vec = np.mean(embeddings, axis=0)
        # Normalise so cosine similarity comparisons are valid
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        preference_vector = mean_vec.tolist()

    # Upsert into user_profiles
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalars().first()

    if profile:
        profile.preferred_cuisines = json.dumps(net_liked_cuisines)
        profile.avoided_cuisines = json.dumps(net_avoided_cuisines)
        profile.preferred_cities = json.dumps(preferred_cities)
        profile.preference_vector = json.dumps(preference_vector)
        profile.feedback_count = len(all_feedback)
    else:
        profile = UserProfile(
            user_id=user_id,
            preferred_cuisines=json.dumps(net_liked_cuisines),
            avoided_cuisines=json.dumps(net_avoided_cuisines),
            preferred_cities=json.dumps(preferred_cities),
            preference_vector=json.dumps(preference_vector),
            feedback_count=len(all_feedback)
        )
        db.add(profile)

    await db.commit()
    await db.refresh(profile)
    return profile


async def get_profile(db: AsyncSession, user_id: str) -> dict | None:
    """
    Load a user profile and return it as a clean dict.
    Returns None if the user has no feedback history yet.
    """
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalars().first()

    if not profile:
        return None

    return {
        "user_id": profile.user_id,
        "preferred_cuisines": json.loads(profile.preferred_cuisines),
        "avoided_cuisines": json.loads(profile.avoided_cuisines),
        "preferred_cities": json.loads(profile.preferred_cities),
        "preference_vector_length": len(json.loads(profile.preference_vector)),
        "feedback_count": profile.feedback_count,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None
    }


def apply_profile_boost(
    results: list[dict],
    profile: dict | None
) -> list[dict]:
    """
    Re-rank hybrid search results using the user's preference profile.

    Boosts restaurants whose cuisine is in preferred_cuisines.
    Penalises restaurants whose cuisine is in avoided_cuisines.
    Does not remove results — just adjusts the order.

    The boost is additive on top of the RRF score so the original
    ranking is still respected when there is no strong preference signal.

    Args:
        results: Output of retrieval.hybrid_search()
        profile: Output of get_profile(), or None for anonymous users

    Returns:
        Re-sorted results list with a personalisation_boost field added.
    """
    if not profile:
        for r in results:
            r["personalisation_boost"] = 0.0
        return results

    preferred = set(profile.get("preferred_cuisines", []))
    avoided = set(profile.get("avoided_cuisines", []))

    for r in results:
        boost = 0.0
        cuisine = r.get("cuisine", "")

        if cuisine in preferred:
            boost += 0.05      # meaningful lift without overriding relevance
        if cuisine in avoided:
            boost -= 0.05

        r["personalisation_boost"] = round(boost, 4)
        r["rrf_score"] = round(r["rrf_score"] + boost, 6)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# ── Internal helper ────────────────────────────────────────────────────────

def _ranked_list(items: list[str]) -> list[str]:
    """
    Return unique items sorted by frequency (most common first).
    Used to build cuisine and city preference lists.
    """
    from collections import Counter
    counts = Counter(items)
    return [item for item, _ in counts.most_common()]
