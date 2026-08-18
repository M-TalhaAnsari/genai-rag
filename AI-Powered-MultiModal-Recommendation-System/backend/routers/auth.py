"""
backend/routers/auth.py
"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.db_models import User
from backend.models.schemas import UserCreate, UserLogin, UserOut, TokenPair, RefreshRequest
from backend.services import auth_service, google_oauth_service

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Email + password (unchanged from before) ────────────────────────────────

@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    if len(payload.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        user = await auth_service.create_user(db, payload.email, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", response_model=TokenPair)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_service.authenticate_user(db, payload.email, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        access_token, refresh_token = await auth_service.rotate_refresh_token(
            db, payload.refresh_token
        )
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    await auth_service.revoke_refresh_token(db, payload.refresh_token)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


# ── Google OAuth ─────────────────────────────────────────────────────────────

@router.get("/google/login")
async def google_login():
    """Returns Google's consent-screen URL. Frontend renders a link/button to it."""
    url = await google_oauth_service.build_authorization_url()
    return {"auth_url": url}


@router.get("/google/callback")
async def google_callback(code: str, state: str, db: AsyncSession = Depends(get_db)):
    """
    Google redirects the browser here after the user consents.
    Verifies everything server-side, then bounces the browser back to
    the frontend with a one-time login_code — never the real tokens.
    """
    try:
        await google_oauth_service.verify_state(state)
        id_token_str = await google_oauth_service.exchange_code_for_id_token(code)
        payload = google_oauth_service.verify_google_id_token(id_token_str)
        user = await google_oauth_service.get_or_create_google_user(db, payload)
        login_code = await google_oauth_service.create_login_code(user.id)
    except auth_service.AuthError as e:
        return RedirectResponse(f"{settings.FRONTEND_URL}?auth_error={e}")

    return RedirectResponse(f"{settings.FRONTEND_URL}?login_code={login_code}")


@router.post("/google/exchange", response_model=TokenPair)
async def google_exchange(login_code: str, db: AsyncSession = Depends(get_db)):
    """Frontend calls this right after landing back with ?login_code=...
    in the URL — trades the one-time code for the real token pair."""
    try:
        user_id_str = await google_oauth_service.redeem_login_code(login_code)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    result = await db.execute(select(User).where(User.id == UUID(user_id_str)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found.")

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)