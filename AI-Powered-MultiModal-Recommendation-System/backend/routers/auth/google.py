"""
backend/routers/auth/google.py

Google OAuth (Authorization Code flow) routes. Split from local.py —
see routers/auth/__init__.py for how the two are mounted together
under one /auth prefix. Routes here end up at /auth/google/*.
"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.database import get_db
from backend.models.db_models import User
from backend.models.schemas import TokenPair
from backend.services.auth import core as auth_service
from backend.services.auth import google_oauth as google_oauth_service

router = APIRouter(prefix="/google")


class GoogleExchangeRequest(BaseModel):
    login_code: str

class GoogleLinkConfirm(BaseModel):
    link_token: str
    password: str


@router.get("/login")
async def google_login():
    """Returns Google's consent-screen URL. Frontend renders a link/button to it."""
    url = await google_oauth_service.build_authorization_url()
    return {"auth_url": url}


@router.get("/callback")
async def google_callback(code: str, state: str, db: AsyncSession = Depends(get_db)):
    try:
        await google_oauth_service.verify_state(state)
        id_token_str = await google_oauth_service.exchange_code_for_id_token(code)
        payload = google_oauth_service.verify_google_id_token(id_token_str)
        user, pending_link = await google_oauth_service.resolve_google_identity(db, payload)
    except auth_service.AuthError as e:
        return RedirectResponse(f"{settings.FRONTEND_URL}?auth_error={e}")

    if user:
        login_code = await google_oauth_service.create_login_code(user.id)
        return RedirectResponse(f"{settings.FRONTEND_URL}?login_code={login_code}")

    # Existing local account, not yet linked — frontend must prompt for
    # the local password before we merge identities.
    link_token = await google_oauth_service.create_link_token(
        pending_link["google_id"], pending_link["email"]
    )
    return RedirectResponse(
        f"{settings.FRONTEND_URL}?link_required={link_token}&email={pending_link['email']}"
    )


@router.post("/exchange", response_model=TokenPair)
async def google_exchange(payload: GoogleExchangeRequest, db: AsyncSession = Depends(get_db)):
    """Frontend calls this right after landing back with ?login_code=...
    in the URL — trades the one-time code for the real token pair via
    a POST body (not a query param — keeps the code out of logs)."""
    try:
        user_id_str = await google_oauth_service.redeem_login_code(payload.login_code)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    result = await db.execute(select(User).where(User.id == UUID(user_id_str)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found.")

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/link-confirm", response_model=TokenPair)
async def google_link_confirm(payload: GoogleLinkConfirm, db: AsyncSession = Depends(get_db)):
    """Completes account linking: proves the caller controls the
    existing LOCAL account (password) before attaching a Google
    identity to it."""
    try:
        google_id, email = await google_oauth_service.redeem_link_token(payload.link_token)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    try:
        user = await auth_service.authenticate_user(db, email, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    user.google_id = google_id
    user.auth_provider = "google_and_local"
    await db.commit()
    await db.refresh(user)

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)