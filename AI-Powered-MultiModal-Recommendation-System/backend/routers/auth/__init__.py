"""
backend/routers/auth/__init__.py

"""

from fastapi import APIRouter

from .local import router as local_router
from .google import router as google_router

router = APIRouter(prefix="/auth", tags=["auth"])
router.include_router(local_router)
router.include_router(google_router)