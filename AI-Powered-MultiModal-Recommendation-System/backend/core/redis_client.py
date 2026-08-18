"""
backend/core/redis_client.py

One shared async Redis connection pool. Import `redis_client` anywhere
you need it — same pattern as `engine`/`AsyncSessionLocal` in database.py.
"""

import redis.asyncio as redis
from backend.core.config import settings

redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,   # get back str, not bytes
)