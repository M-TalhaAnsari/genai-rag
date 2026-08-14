"""
backend/main.py
----------------
Thin application entry point.
All logic lives in routers/ and services/.
This file only creates the app and mounts routers.
"""

from datetime import datetime
from fastapi import FastAPI
from backend.core.config import settings
from backend.models.schemas import HealthResponse

from backend.routers.restaurants   import router as restaurants_router
from backend.routers.contact_links import router as contact_links_router
from backend.routers.search        import router as search_router
from backend.routers.recommend     import router as recommend_router
from backend.routers.feedback      import router as feedback_router
from backend.routers.memory        import router as memory_router
from backend.routers.analytics     import router as analytics_router
from backend.routers.ingestion     import router as ingestion_router

from backend.routers import auth as auth_router


app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", response_model=HealthResponse, tags=["health"])
def root():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )


# URL map:
#   /restaurants/*                    read endpoints + contact-links + reviews
#   /search, /search/full, /search/by-review
#   /recommend                        6-agent SSE stream
#   /feedback, /profile/{user_id}
#   /memory/{user_id}
#   /analytics, /vector-stats
#   /ingestion/*                      all write / sync operations

app.include_router(restaurants_router)
app.include_router(contact_links_router)
app.include_router(search_router)
app.include_router(recommend_router)
app.include_router(feedback_router)
app.include_router(memory_router)
app.include_router(analytics_router)
app.include_router(ingestion_router)
app.include_router(auth_router.router)
