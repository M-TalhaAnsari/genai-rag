"""
backend/services/auth/errors.py

AuthError lives in its own module (not core.py) specifically so that
core.py can import from rate_limit.py (for login lockout) while
rate_limit.py also needs AuthError as its base class, without the two
modules importing each other.
"""


class AuthError(Exception):
    """Raised for expected auth failures (bad creds, token reuse,
    etc). Routers catch this and translate to the right HTTP status —
    keeps service files free of HTTPException / FastAPI imports."""
    pass