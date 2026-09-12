import asyncio, json
from backend.core.database import AsyncSessionLocal
from backend.models.db_models import ApifyRawSnapshot
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ApifyRawSnapshot).where(ApifyRawSnapshot.id == 1))
        snap = result.scalars().first()
        if snap and snap.raw_json:
            places = json.loads(snap.raw_json)
            for p in places:
                print(p.get('title'), '| reviews:', len(p.get('reviews', [])), '| images:', len(p.get('imageUrls', [])))
        else:
            print('No snapshot found with id 1')

asyncio.run(main())
