# MCP service — `mcp_service/`

Optional. **Not mounted into `backend/main.py`** — the main FastAPI app
and Streamlit frontend both run fully without this. Only run it if you
want the backend's search/recommend/feedback functionality exposed as
MCP tools to something like Claude Desktop, or this project's own
`mcp_client.py`.

```
mcp_server.py   FastMCP server — 5 tools + 1 resource over stdio transport
mcp_client.py    A client + ReAct loop (ChatGroq/Gemini) for testing the
                  server, or as a reference for how to build another MCP
                  client against it
```

Run it:

```bash
python mcp_service/mcp_server.py
```

## Tools exposed

```
search_restaurants     → backend.retrieving.retrieval.hybrid_search()
get_recommendations    → the full 6-agent workflow (10-30s — warn callers
                           it's slow, don't let an MCP client time out on it)
submit_feedback         → backend.services.feedback_service.save_feedback()
get_user_profile         → backend.services.feedback_service.get_profile()
get_analytics             → backend.services.analytics_service.get_analytics()
resource: restaurant://stats → live PostgreSQL + ChromaDB counts
```

## Sync tool handlers, async services

FastMCP tool functions are synchronous by convention, but everything
underneath (`hybrid_search`, `save_feedback`, etc.) is async SQLAlchemy.
Each tool handler wraps its async call through a `_run()` helper that
bridges the two via a thread pool executor. If you add a new tool that
calls an async service function, use the same `_run()` pattern rather
than trying to `await` directly in a sync function or reaching for
`asyncio.run()` inline (which breaks if an event loop is already running
in the process).

## A real bug that lived here for a while

`mcp_server.py` had `from backend.models.schemas import Restaurant` —
but `Restaurant` is a SQLAlchemy ORM class in `backend.models.db_models`,
not a Pydantic model, and `schemas.py` never defined one by that name at
all. The line that actually used it, `select(Restaurant)`, needs the ORM
class specifically. This meant **the file could not be imported**, so
`python mcp_service/mcp_server.py` would fail immediately on startup —
not a subtle runtime issue, but apparently never actually run/tested
since whenever this was introduced. Fixed to import from `db_models`
instead. If you're working from an older clone or a stale branch and
this server won't even start, check this import first before assuming
something else broke.

Given this had gone unnoticed, treat this file as less battle-tested
than anything under `backend/routers/` or `backend/services/` — it's
worth an actual run (not just `python -m py_compile`) after any change,
since a compile check would not have caught the bug above (the import
statement is syntactically valid Python either way; it only fails at
runtime when the module is actually loaded).

## This file is independent of the auth system

Nothing in `mcp_service/` goes through `services/auth/` or checks for a
JWT — it talks to the same service-layer functions the FastAPI routers
call, but as its own process with its own DB session, no HTTP layer,
and no auth boundary of its own. If you expose this server beyond your
own machine, it currently has no access control — that's a gap worth
knowing about, not something already handled elsewhere in the codebase.