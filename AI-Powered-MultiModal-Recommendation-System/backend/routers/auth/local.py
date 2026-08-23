"""
backend/routers/auth/local.py

Email + password auth: register, login, email verification, refresh,
logout, /me. Google OAuth routes live in google.py — split once
auth's route count crossed ~10 endpoints. See routers/auth/__init__.py
for how the two are mounted together under one /auth prefix.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.db_models import User
from backend.models.schemas import UserCreate, UserLogin, UserOut, TokenPair, RefreshRequest
from backend.services.auth import core as auth_service

router = APIRouter()


class VerifyEmailRequest(BaseModel):
    token: str

class ResendVerificationRequest(BaseModel):
    email: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    # No response_model=TokenPair — register no longer logs the user
    # in. It creates an unverified row and sends a verification email.
    # Real tokens only come from /verify-email or /login, once
    # email_verified is True.
    if len(payload.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        await auth_service.create_user(db, payload.email, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))

    return {"message": "Registered. Check your email to verify your account."}


@router.post("/login", response_model=TokenPair)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_service.authenticate_user(db, payload.email, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/verify-email", response_model=TokenPair)
async def verify_email(payload: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_service.verify_email_token(db, payload.token)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/resend-verification", status_code=status.HTTP_204_NO_CONTENT)
async def resend_verification(payload: ResendVerificationRequest, db: AsyncSession = Depends(get_db)):
    await auth_service.resend_verification(db, payload.email)
    # Always 204 — never reveals whether the email exists or is
    # already verified.


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