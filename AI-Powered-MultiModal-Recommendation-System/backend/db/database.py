from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv
load_dotenv()

# Load from environment variable — never hardcode credentials
# Set NEON_DATABASE_URL in your .env file
DATABASE_URL = os.environ.get(
    "NEON_DATABASE_URL",
    "postgresql+asyncpg://neondb_owner:npg_4faxovqpmTE3@ep-lively-surf-ay49ddn1-pooler.c-5.us-east-2.aws.neon.tech/neondb?ssl=require"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=True
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()
