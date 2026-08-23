"""
backend/services/auth/core.py  (formerly services/auth_service.py)

Business logic only — no `from fastapi import ...` here, same rule as
every other file in services/. This is what makes it callable from a
script, a test, or a future admin CLI without dragging in FastAPI.

Routers call these functions and translate results/exceptions into
HTTP responses. Token *encoding* lives in core/security.py; this file
is about *what happens* (create a user, check credentials, manage
refresh-token rows) — not the cryptography itself.
"""
import secrets
from datetime import datetime, timezone, timedelta
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
from backend.core.redis_client import redis_client

from backend.models.db_models import User, RefreshToken, UserRole
from backend.services.auth import email as email_service

VERIFICATION_TOKEN_TTL_SECONDS = 24 * 60 * 60  # 24h
STALE_UNVERIFIED_THRESHOLD = timedelta(hours=24)


class AuthError(Exception):
    """Raised for expected auth failures (bad creds, token reuse, etc).
    Routers catch this and translate to the right HTTP status —
    keeps this file free of HTTPException / FastAPI imports."""
    pass


async def create_user(db: AsyncSession, email: str, password: str, role: UserRole = UserRole.user) -> User:
    existing_result = await db.execute(select(User).where(User.email == email))
    existing = existing_result.scalar_one_or_none()

    if existing is not None:
        if existing.email_verified:
            raise AuthError("Email already registered")

        # Unverified squat: only reclaim if it's stale. A fresh
        # unverified row (user mid-signup, hasn't checked inbox yet)
        # should NOT be silently deleted out from under them.
        age = datetime.now(timezone.utc) - existing.created_at.replace(tzinfo=timezone.utc)
        if age < STALE_UNVERIFIED_THRESHOLD:
            raise AuthError(
                "Email already registered — check your inbox for a verification link, "
                "or use 'resend verification' if it expired."
            )

        await db.delete(existing)
        await db.flush()

    user = User(
        email=email,
        hashed_password=hash_password(password),
        role=role,
        email_verified=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    await send_verification_email(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or user.hashed_password is None:
        raise AuthError("Incorrect email or password")

    if not verify_password(password, user.hashed_password):
        raise AuthError("Incorrect email or password")

    if not user.email_verified:
        raise AuthError("Please verify your email before logging in.")

    if not user.is_active:
        raise AuthError("Account is disabled")

    return user


async def send_verification_email(user: User) -> None:
    token = secrets.token_urlsafe(32)
    await redis_client.set(
        f"email_verify:{token}", str(user.id), ex=VERIFICATION_TOKEN_TTL_SECONDS
    )
    email_service.send_verification_email(user.email, token)


async def verify_email_token(db: AsyncSession, token: str) -> User:
    key = f"email_verify:{token}"
    user_id_str = await redis_client.get(key)
    if not user_id_str:
        raise AuthError("Invalid or expired verification link.")
    await redis_client.delete(key)

    result = await db.execute(select(User).where(User.id == UUID(user_id_str)))
    user = result.scalar_one_or_none()
    if not user:
        raise AuthError("User not found.")

    user.email_verified = True
    await db.commit()
    await db.refresh(user)
    return user


async def resend_verification(db: AsyncSession, email: str) -> None:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    # Same response whether or not the account exists / is already
    # verified — don't let this endpoint be used to probe which
    # emails are registered.
    if user and not user.email_verified:
        await send_verification_email(user)


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