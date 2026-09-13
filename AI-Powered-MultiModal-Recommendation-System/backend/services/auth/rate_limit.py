"""
backend/services/auth/rate_limit.py

Generic Redis-backed fixed-window rate limiting, plus email-keyed
login lockout for brute-force protection. No FastAPI imports — same
convention as every services/ file; routers translate RateLimitError
to a 429 with a Retry-After header.
"""

from typing import Optional

from backend.core.redis_client import redis_client
from backend.services.auth.errors import AuthError


class RateLimitError(AuthError):
    """Subclasses AuthError so existing generic error handling still
    works if a caller forgets to special-case it, but routers should
    catch this FIRST (before AuthError) to return 429 instead of the
    fallback 400/401/409."""

    def __init__(self, message: str, retry_after_seconds: Optional[int] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


async def check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> None:
    """
    Fixed-window counter: the first request in a window sets the TTL,
    every subsequent request just increments. 
    """
    current = await redis_client.incr(key)
    if current == 1:
        await redis_client.expire(key, window_seconds)
    if current > max_attempts:
        ttl = await redis_client.ttl(key)
        raise RateLimitError(
            "Too many requests. Please try again later.",
            retry_after_seconds=ttl if ttl and ttl > 0 else window_seconds,
        )

LOGIN_MAX_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60  # 15 min


async def check_login_lockout(email: str) -> None:
    key = f"login_lockout:{email}"
    locked = await redis_client.get(key)
    if locked:
        ttl = await redis_client.ttl(key)
        raise RateLimitError(
            "Too many failed login attempts. Please try again later.",
            retry_after_seconds=ttl if ttl and ttl > 0 else LOGIN_LOCKOUT_SECONDS,
        )


async def record_login_failure(email: str) -> None:
    fail_key = f"login_fail:{email}"
    failures = await redis_client.incr(fail_key)
    if failures == 1:
        await redis_client.expire(fail_key, LOGIN_LOCKOUT_SECONDS)
    if failures >= LOGIN_MAX_FAILURES:
        await redis_client.set(f"login_lockout:{email}", "1", ex=LOGIN_LOCKOUT_SECONDS)


async def reset_login_failures(email: str) -> None:
    await redis_client.delete(f"login_fail:{email}", f"login_lockout:{email}")