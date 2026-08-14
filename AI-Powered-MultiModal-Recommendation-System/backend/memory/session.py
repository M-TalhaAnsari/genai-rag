"""
backend/memory/session.py
--------------------------
Short-term session memory — lives in RAM, cleared when server restarts.

Stores per-user conversation context for the current session:
  - Last N search queries
  - Last N results shown
  - Current conversation messages (for the ReAct loop)
  - Session start time

This is used by:
  - /recommend  → passes recent queries to agents as context
  - /search     → avoids repeating identical results
  - MCP client  → maintains conversation history across tool calls

Intentionally simple — a dict in RAM.
For multi-server deployments, swap the backing store to Redis.
"""

from datetime import datetime
from collections import deque
from typing import Any

MAX_QUERIES   = 20
MAX_RESULTS   = 10
MAX_MESSAGES  = 50   

_sessions: dict[str, dict] = {}


def _get_or_create(user_id: str) -> dict:
    """Return existing session or create a fresh one."""
    if user_id not in _sessions:
        _sessions[user_id] = {
            "user_id":      user_id,
            "started_at":   datetime.utcnow().isoformat(),
            "queries":      deque(maxlen=MAX_QUERIES),
            "last_results": deque(maxlen=MAX_RESULTS),
            "messages":     deque(maxlen=MAX_MESSAGES),
            "context":      {}   # free-form key-value store
        }
    return _sessions[user_id]


# ── Public API ─────────────────────────────────────────────────────────────

def add_query(user_id: str, query: str) -> None:
    """Record a search query for this session."""
    session = _get_or_create(user_id)
    session["queries"].append({
        "query": query,
        "timestamp": datetime.utcnow().isoformat()
    })


def add_results(user_id: str, results: list[dict]) -> None:
    """Record the results shown to the user."""
    session = _get_or_create(user_id)
    session["last_results"].extend(results)


def add_message(user_id: str, role: str, content: str) -> None:
    """
    Append one conversation turn.
    role: "user" | "assistant" | "tool"
    """
    session = _get_or_create(user_id)
    session["messages"].append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.utcnow().isoformat()
    })


def get_recent_queries(user_id: str, n: int = 5) -> list[str]:
    """Return the last n queries for this user in this session."""
    session = _get_or_create(user_id)
    queries = list(session["queries"])
    return [q["query"] for q in queries[-n:]]


def get_last_results(user_id: str) -> list[dict]:
    """Return the most recently shown results."""
    session = _get_or_create(user_id)
    return list(session["last_results"])


def get_messages(user_id: str) -> list[dict]:
    """Return full conversation history for this session."""
    session = _get_or_create(user_id)
    return list(session["messages"])


def get_session_summary(user_id: str) -> dict:
    """
    Return a compact summary of the session for use in agent prompts.
    Agents use this to avoid repeating results or misunderstanding context.
    """
    session = _get_or_create(user_id)
    recent_queries = get_recent_queries(user_id)
    last_results   = get_last_results(user_id)

    shown_names = list({
        r.get("name", "") for r in last_results if r.get("name")
    })

    return {
        "user_id":        user_id,
        "session_start":  session["started_at"],
        "recent_queries": recent_queries,
        "shown_restaurants": shown_names[:10],
        "turn_count":     len(session["messages"])
    }


def set_context(user_id: str, key: str, value: Any) -> None:
    """Store arbitrary key-value context for this session."""
    session = _get_or_create(user_id)
    session["context"][key] = value


def get_context(user_id: str, key: str, default: Any = None) -> Any:
    """Retrieve a context value set earlier in this session."""
    session = _get_or_create(user_id)
    return session["context"].get(key, default)


def clear_session(user_id: str) -> None:
    """Clear all session data for this user."""
    _sessions.pop(user_id, None)


def active_sessions() -> int:
    """Return count of active sessions (for monitoring)."""
    return len(_sessions)
