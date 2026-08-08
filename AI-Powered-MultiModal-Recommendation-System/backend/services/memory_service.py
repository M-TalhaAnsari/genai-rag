"""
backend/services/memory_service.py
------------------------------------
Two-layer memory system:

SHORT-TERM (RAM)
  Per-user session dict using Python deques.
  Cleared on server restart.
  Swap backing store to Redis for multi-server deployments.

LONG-TERM (PostgreSQL)
  Every conversation turn saved permanently.
  LLM-generated 100-word summary recomputed every 10 turns.
  Agents receive the summary, not the full history —
  context window stays constant regardless of session length.
"""

import json
from collections import deque
from datetime import datetime
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import ConversationHistory, UserMemorySummary

# ── Short-term (RAM) ────────────────────────────────────────────────────────

_MAX_QUERIES  = 20
_MAX_RESULTS  = 10
_MAX_MESSAGES = 50

_sessions: dict[str, dict] = {}


def _session(user_id: str) -> dict:
    if user_id not in _sessions:
        _sessions[user_id] = {
            "started_at":   datetime.utcnow().isoformat(),
            "queries":      deque(maxlen=_MAX_QUERIES),
            "last_results": deque(maxlen=_MAX_RESULTS),
            "messages":     deque(maxlen=_MAX_MESSAGES),
            "context":      {},
        }
    return _sessions[user_id]


def add_query(user_id: str, query: str) -> None:
    _session(user_id)["queries"].append(
        {"query": query, "ts": datetime.utcnow().isoformat()}
    )


def add_results(user_id: str, results: list[dict]) -> None:
    _session(user_id)["last_results"].extend(results)


def add_message(user_id: str, role: str, content: str) -> None:
    _session(user_id)["messages"].append(
        {"role": role, "content": content, "ts": datetime.utcnow().isoformat()}
    )


def get_recent_queries(user_id: str, n: int = 5) -> list[str]:
    return [q["query"] for q in list(_session(user_id)["queries"])[-n:]]


def get_last_results(user_id: str) -> list[dict]:
    return list(_session(user_id)["last_results"])


def get_session_summary(user_id: str) -> dict:
    s = _session(user_id)
    shown = list({r.get("name", "") for r in s["last_results"] if r.get("name")})
    return {
        "user_id":           user_id,
        "session_start":     s["started_at"],
        "recent_queries":    get_recent_queries(user_id),
        "shown_restaurants": shown[:10],
        "turn_count":        len(s["messages"]),
    }


def set_context(user_id: str, key: str, value: Any) -> None:
    _session(user_id)["context"][key] = value


def get_context(user_id: str, key: str, default: Any = None) -> Any:
    return _session(user_id)["context"].get(key, default)


def clear_session(user_id: str) -> None:
    _sessions.pop(user_id, None)


def active_sessions() -> int:
    return len(_sessions)


# ── Long-term (PostgreSQL) ──────────────────────────────────────────────────

async def save_turn(
    db: AsyncSession,
    user_id: str,
    role: str,
    content: str,
    query: str = None,
) -> None:
    db.add(ConversationHistory(
        user_id=user_id,
        role=role,
        content=content[:2000],
        query=query,
    ))
    await db.commit()

    count = await _turn_count(db, user_id)
    if count % 10 == 0:
        await recompute_summary(db, user_id)


async def save_conversation(
    db: AsyncSession,
    user_id: str,
    user_message: str,
    assistant_response: str,
    query: str = None,
) -> None:
    await save_turn(db, user_id, "user",      user_message,       query)
    await save_turn(db, user_id, "assistant", assistant_response, query)


async def get_recent_history(
    db: AsyncSession,
    user_id: str,
    limit: int = 20,
) -> list[dict]:
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
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reversed(rows)
    ]


async def get_memory_summary(db: AsyncSession, user_id: str) -> str | None:
    result = await db.execute(
        select(UserMemorySummary).where(UserMemorySummary.user_id == user_id)
    )
    record = result.scalars().first()
    return record.summary if record else None


async def get_search_history(
    db: AsyncSession,
    user_id: str,
    limit: int = 50,
) -> list[str]:
    result = await db.execute(
        select(ConversationHistory.query)
        .where(
            ConversationHistory.user_id == user_id,
            ConversationHistory.query.isnot(None),
        )
        .order_by(ConversationHistory.created_at.desc())
        .limit(limit)
    )
    seen, unique = set(), []
    for q in result.scalars().all():
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


async def recompute_summary(db: AsyncSession, user_id: str) -> str:
    """Ask an LLM to write a 100-word memory summary from recent history."""
    history = await get_recent_history(db, user_id, limit=30)
    if len(history) < 3:
        return ""

    history_text = "\n".join(
        f"{t['role'].upper()}: {t['content'][:200]}" for t in history
    )

    from backend.agents.llm import call_agent
    summary = call_agent(
        "profile_analyser",
        f"Write a concise 100-word memory summary (third person) about this user "
        f"based on their restaurant discovery conversation. Capture cuisine interests, "
        f"preferred cities, dining style.\n\nConversation:\n{history_text}\n\n"
        f"Write ONLY the summary paragraph.",
    )

    count = await _turn_count(db, user_id)
    result = await db.execute(
        select(UserMemorySummary).where(UserMemorySummary.user_id == user_id)
    )
    record = result.scalars().first()

    if record:
        record.summary    = summary
        record.turn_count = count
    else:
        db.add(UserMemorySummary(user_id=user_id, summary=summary, turn_count=count))

    await db.commit()
    return summary


async def get_full_memory_context(db: AsyncSession, user_id: str) -> dict:
    """Combined long-term context for agent prompts."""
    return {
        "memory_summary":  await get_memory_summary(db, user_id) or "No summary yet.",
        "recent_history":  await get_recent_history(db, user_id, limit=10),
        "recent_searches": await get_search_history(db, user_id, limit=10),
    }


async def _turn_count(db: AsyncSession, user_id: str) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(ConversationHistory)
        .where(ConversationHistory.user_id == user_id)
    )
    return result.scalar() or 0