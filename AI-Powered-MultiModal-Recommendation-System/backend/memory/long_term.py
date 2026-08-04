"""
backend/memory/long_term.py
-----------------------------
Long-term memory — persists across server restarts in PostgreSQL.

Stores:
  - Full conversation history per user (all sessions)
  - Summarised memory: key facts the system has learned about the user
  - Search history (every query ever made, deduplicated)

Used by:
  - /recommend  → agents get long-term context ("last time you searched X")
  - /profile    → shows conversation history alongside preference profile
  - Agents      → profile_analyser reads memory summary for richer context

The memory_summary is a short paragraph written by an LLM that
captures the most important things known about the user.
It is recomputed every N conversations to stay current.
"""

import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, select, func
from sqlalchemy.sql import func as sqlfunc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import Base
from backend.agents.configs import call_agent


# ── Model ──────────────────────────────────────────────────────────────────

class ConversationHistory(Base):
    """Stores every conversation turn for every user, permanently."""

    __tablename__ = "conversation_history"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(String, nullable=False, index=True)
    role       = Column(String, nullable=False)      # "user" | "assistant"
    content    = Column(Text, nullable=False)
    query      = Column(String, nullable=True)        # search query if applicable
    created_at = Column(DateTime, server_default=sqlfunc.now())


class UserMemorySummary(Base):
    """
    LLM-generated summary of what the system knows about the user.
    Recomputed every 10 new conversation turns.
    """

    __tablename__ = "user_memory_summaries"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(String, unique=True, nullable=False, index=True)
    summary       = Column(Text, nullable=False)
    turn_count    = Column(Integer, default=0)   # turns when last summarised
    updated_at    = Column(DateTime, server_default=sqlfunc.now(),
                           onupdate=sqlfunc.now())


# ── Write operations ───────────────────────────────────────────────────────

async def save_turn(
    db: AsyncSession,
    user_id: str,
    role: str,
    content: str,
    query: str = None
) -> None:
    """
    Persist one conversation turn to PostgreSQL.

    Args:
        role:    "user" or "assistant"
        content: The message text
        query:   If this turn was triggered by a search, the query string
    """
    turn = ConversationHistory(
        user_id=user_id,
        role=role,
        content=content[:2000],   # cap length to avoid huge rows
        query=query
    )
    db.add(turn)
    await db.commit()

    # Recompute memory summary every 10 turns
    count = await _turn_count(db, user_id)
    if count % 10 == 0:
        await recompute_summary(db, user_id)


async def save_conversation(
    db: AsyncSession,
    user_id: str,
    user_message: str,
    assistant_response: str,
    query: str = None
) -> None:
    """Save both sides of one conversation exchange in one call."""
    await save_turn(db, user_id, "user",      user_message,       query)
    await save_turn(db, user_id, "assistant", assistant_response, query)


# ── Read operations ────────────────────────────────────────────────────────

async def get_recent_history(
    db: AsyncSession,
    user_id: str,
    limit: int = 20
) -> list[dict]:
    """
    Return the most recent conversation turns for a user.
    Used by agents to understand recent context.
    """
    result = await db.execute(
        select(ConversationHistory)
        .where(ConversationHistory.user_id == user_id)
        .order_by(ConversationHistory.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()

    return [
        {
            "role":       r.role,
            "content":    r.content,
            "query":      r.query,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in reversed(rows)   # oldest first for readability
    ]


async def get_memory_summary(
    db: AsyncSession,
    user_id: str
) -> str | None:
    """
    Return the latest LLM-generated memory summary for this user.
    Returns None if no summary has been computed yet.
    """
    result = await db.execute(
        select(UserMemorySummary)
        .where(UserMemorySummary.user_id == user_id)
    )
    record = result.scalars().first()
    return record.summary if record else None


async def get_search_history(
    db: AsyncSession,
    user_id: str,
    limit: int = 50
) -> list[str]:
    """Return deduplicated list of past search queries for this user."""
    result = await db.execute(
        select(ConversationHistory.query)
        .where(
            ConversationHistory.user_id == user_id,
            ConversationHistory.query.isnot(None)
        )
        .order_by(ConversationHistory.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    # Deduplicate while preserving order
    seen, unique = set(), []
    for q in rows:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# ── Memory summary generation ──────────────────────────────────────────────

async def recompute_summary(db: AsyncSession, user_id: str) -> str:
    """
    Ask an LLM to write a short memory summary from recent history.
    Stored in user_memory_summaries and used by agents as context.

    Returns the generated summary string.
    """
    history = await get_recent_history(db, user_id, limit=30)

    if len(history) < 3:
        return ""   # not enough data to summarise

    history_text = "\n".join(
        f"{turn['role'].upper()}: {turn['content'][:200]}"
        for turn in history
    )

    message = f"""Based on this conversation history between a user and a restaurant
discovery assistant, write a concise memory summary (max 100 words) in third person.

Capture:
- What cuisines or restaurants the user has shown interest in
- Any locations they prefer
- Any dietary preferences or restrictions mentioned
- Their general dining style or preferences

Conversation history:
{history_text}

Write ONLY the summary paragraph. No labels, no preamble.
"""
    summary = call_agent("profile_analyser", message)

    # Upsert summary
    result = await db.execute(
        select(UserMemorySummary)
        .where(UserMemorySummary.user_id == user_id)
    )
    record = result.scalars().first()

    turn_count = await _turn_count(db, user_id)

    if record:
        record.summary    = summary
        record.turn_count = turn_count
    else:
        record = UserMemorySummary(
            user_id    = user_id,
            summary    = summary,
            turn_count = turn_count
        )
        db.add(record)

    await db.commit()
    return summary


async def get_full_memory_context(
    db: AsyncSession,
    user_id: str
) -> dict:
    """
    Return everything the system knows about the user for agent prompts.

    Combines:
      - LLM memory summary (what we know about them overall)
      - Recent conversation history (last 10 turns)
      - Recent search queries
    """
    summary        = await get_memory_summary(db, user_id)
    recent_history = await get_recent_history(db, user_id, limit=10)
    search_history = await get_search_history(db, user_id, limit=10)

    return {
        "memory_summary":   summary or "No memory summary yet.",
        "recent_history":   recent_history,
        "recent_searches":  search_history
    }


# ── Internal helpers ───────────────────────────────────────────────────────

async def _turn_count(db: AsyncSession, user_id: str) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(ConversationHistory)
        .where(ConversationHistory.user_id == user_id)
    )
    return result.scalar() or 0
