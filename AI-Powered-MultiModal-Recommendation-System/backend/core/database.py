"""
backend/core/database.py
-------------------------
PostgreSQL async connection via SQLAlchemy 2.0 + asyncpg.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.core.config import settings

from backend.core.config import settings
engine = create_async_engine(
    settings.NEON_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=280,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session



