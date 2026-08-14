"""
backend/routers/auth.py

Thin — matches the project's "routers never contain business logic"
rule. Every function here just: validate input (Pydantic already did
most of this), call auth_service, translate AuthError -> HTTPException,
return response.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.db_models import User
from backend.models.schemas import UserCreate, UserLogin, UserOut, TokenPair, RefreshRequest
from backend.services import auth_services

from fastapi import APIRouter, Depends
from backend.core.security import get_current_user, require_admin
from backend.models.db_models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    if len(payload.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        user = await auth_services.create_user(db, payload.email, payload.password)
    except auth_services.AuthError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))

    access_token, refresh_token = await auth_services.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", response_model=TokenPair)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_services.authenticate_user(db, payload.email, payload.password)
    except auth_services.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    access_token, refresh_token = await auth_services.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        access_token, refresh_token = await auth_services.rotate_refresh_token(
            db, payload.refresh_token
        )
    except auth_services.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    await auth_services.revoke_refresh_token(db, payload.refresh_token)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user