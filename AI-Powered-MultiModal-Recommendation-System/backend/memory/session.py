"""
backend/memory/session.py
--------------------------
Short-term session memory — now Redis-backed instead of a RAM dict.

Survives server restarts. Naturally expires (TTL) instead of growing
forever. Same public function names as before, so the *logic* using
this module doesn't change — every call site just needs `await` added,
since Redis I/O is inherently async.

One JSON blob per user, under key "session:{user_id}", refreshed
(TTL reset) on every write. This is the "combine data updated together
into one key" pattern — cheaper than a separate Redis key per field.
"""

import json
from datetime import datetime, timezone
from typing import Any

from backend.core.redis_client import redis_client
from backend.core.config import settings

MAX_QUERIES = 20
MAX_RESULTS = 10
MAX_MESSAGES = 50

TTL_SECONDS = settings.SESSION_TTL_SECONDS  # default 7200 (2 hours) — see config.py


def _key(user_id: str) -> str:
    return f"session:{user_id}"


def _default_session(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "queries": [],
        "last_results": [],
        "messages": [],
        "context": {},
    }


async def _load(user_id: str) -> dict:
    raw = await redis_client.get(_key(user_id))
    if raw is None:
        return _default_session(user_id)
    return json.loads(raw)


async def _save(user_id: str, session: dict) -> None:
    await redis_client.set(_key(user_id), json.dumps(session), ex=TTL_SECONDS)


# ── Public API — all async now ──────────────────────────────────────────────

async def add_query(user_id: str, query: str) -> None:
    session = await _load(user_id)
    session["queries"].append({
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    session["queries"] = session["queries"][-MAX_QUERIES:]
    await _save(user_id, session)


async def add_results(user_id: str, results: list[dict]) -> None:
    session = await _load(user_id)
    session["last_results"].extend(results)
    session["last_results"] = session["last_results"][-MAX_RESULTS:]
    await _save(user_id, session)


async def add_message(user_id: str, role: str, content: str) -> None:
    session = await _load(user_id)
    session["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    session["messages"] = session["messages"][-MAX_MESSAGES:]
    await _save(user_id, session)


async def get_recent_queries(user_id: str, n: int = 5) -> list[str]:
    session = await _load(user_id)
    return [q["query"] for q in session["queries"][-n:]]


async def get_last_results(user_id: str) -> list[dict]:
    session = await _load(user_id)
    return session["last_results"]


async def get_messages(user_id: str) -> list[dict]:
    session = await _load(user_id)
    return session["messages"]


async def get_session_summary(user_id: str) -> dict:
    session = await _load(user_id)
    recent_queries = [q["query"] for q in session["queries"][-5:]]
    last_results = session["last_results"]

    shown_names = list({r.get("name", "") for r in last_results if r.get("name")})

    return {
        "user_id": user_id,
        "session_start": session["started_at"],
        "recent_queries": recent_queries,
        "shown_restaurants": shown_names[:10],
        "turn_count": len(session["messages"]),
    }


async def set_context(user_id: str, key: str, value: Any) -> None:
    session = await _load(user_id)
    session["context"][key] = value
    await _save(user_id, session)


async def get_context(user_id: str, key: str, default: Any = None) -> Any:
    session = await _load(user_id)
    return session["context"].get(key, default)


async def clear_session(user_id: str) -> None:
    await redis_client.delete(_key(user_id))


async def active_sessions() -> int:
    """Count of active sessions — approximate, scans the session:* keyspace."""
    count = 0
    async for _ in redis_client.scan_iter(match="session:*"):
        count += 1
    return count