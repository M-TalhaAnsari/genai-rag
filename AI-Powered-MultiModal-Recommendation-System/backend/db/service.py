from sqlalchemy import select
from backend.db.database import AsyncSessionLocal
from backend.model.models import Restaurant

async def get_all_restaurants():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Restaurant))
        restaurants = result.scalars().all()
        return restaurants