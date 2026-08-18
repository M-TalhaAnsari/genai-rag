"""
backend/services/auth_service.py

Business logic only — no `from fastapi import ...` here, same rule as
every other file in services/. This is what makes it callable from a
script, a test, or a future admin CLI without dragging in FastAPI.

Routers call these functions and translate results/exceptions into
HTTP responses. Token *encoding* lives in core/security.py; this file
is about *what happens* (create a user, check credentials, manage
refresh-token rows) — not the cryptography itself.
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from backend.models.db_models import User, RefreshToken, UserRole


class AuthError(Exception):
    """Raised for expected auth failures (bad creds, token reuse, etc).
    Routers catch this and translate to the right HTTP status —
    keeps this file free of HTTPException / FastAPI imports."""
    pass


async def create_user(db: AsyncSession, email: str, password: str, role: UserRole = UserRole.user) -> User:
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none() is not None:
        raise AuthError("Email already registered")

    user = User(email=email, hashed_password=hash_password(password), role=role)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:


    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
 
    if user is None or user.hashed_password is None:
        # Either no such user, or they signed up via Google and have
        # no password to check — same generic error either way, so
        # we don't leak which case it is.
        raise AuthError("Incorrect email or password")

    if not verify_password(password, user.hashed_password):
        raise AuthError("Incorrect email or password")

    if not user.is_active:
        raise AuthError("Account is disabled")

    return user


async def issue_token_pair(db: AsyncSession, user: User) -> tuple[str, str]:
    """Creates a fresh access token (stateless) + refresh token (DB row)."""
    access_token = create_access_token(user.id, user.role.value)
    refresh_token_str, jti, expires_at = create_refresh_token(user.id)

    db.add(RefreshToken(user_id=user.id, jti=jti, expires_at=expires_at))
    await db.commit()

    return access_token, refresh_token_str


async def rotate_refresh_token(db: AsyncSession, refresh_token_str: str) -> tuple[str, str]:
    """
    Validates a refresh token, revokes it, and issues a brand new pair.
    Rotation (not reuse) on every refresh means a stolen-and-replayed
    old refresh token is detectably invalid the moment the legitimate
    client refreshes — the revoked row is the tell.
    """
    payload = decode_token(refresh_token_str)
    if payload.get("type") != "refresh":
        raise AuthError("Invalid token type")

    jti = payload.get("jti")
    user_id = payload.get("sub")

    result = await db.execute(select(RefreshToken).where(RefreshToken.jti == UUID(jti)))
    stored = result.scalar_one_or_none()

    if stored is None or stored.revoked:
        # Either unknown or already-used-once token — treat as
        # compromised. In a hardened version you'd also revoke ALL
        # of this user's refresh tokens here (reuse-detection response).
        raise AuthError("Refresh token invalid or already used")

    if stored.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise AuthError("Refresh token expired")

    # Revoke the used token
    stored.revoked = True

    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthError("User not found or inactive")

    await db.commit()

    return await issue_token_pair(db, user)


async def revoke_refresh_token(db: AsyncSession, refresh_token_str: str) -> None:
    """Logout: revoke a single refresh token."""
    try:
        payload = decode_token(refresh_token_str)
    except Exception:
        return  # already invalid, nothing to revoke
    jti = payload.get("jti")
    if jti:
        await db.execute(
            update(RefreshToken).where(RefreshToken.jti == UUID(jti)).values(revoked=True)
        )
        await db.commit()


async def revoke_all_user_tokens(db: AsyncSession, user_id: UUID) -> None:
    """Logout-everywhere / suspected compromise response."""
    await db.execute(
        update(RefreshToken).where(RefreshToken.user_id == user_id).values(revoked=True)
    )
    await db.commit()