"""
backend/services/auth/core.py  (formerly services/auth_service.py)

"""
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
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
from backend.services.auth.errors import AuthError
from backend.services.auth.rate_limit import (
    check_login_lockout,
    record_login_failure,
    reset_login_failures,
)

VERIFICATION_TOKEN_TTL_SECONDS = 24 * 60 * 60  # 24h
STALE_UNVERIFIED_THRESHOLD = timedelta(hours=24)
LOGIN_CODE_TTL_SECONDS = 60  # frontend should exchange this immediately
PASSWORD_RESET_TTL_SECONDS = 30 * 60  # 30 min
MFA_TOKEN_TTL_SECONDS = 5 * 60  # 5 min to enter a TOTP code


async def create_user(db: AsyncSession, email: str, password: str, role: UserRole = UserRole.user) -> User:
    email = email.strip().lower()

    existing_result = await db.execute(select(User).where(User.email == email))
    existing = existing_result.scalar_one_or_none()

    if existing is not None:
        if existing.email_verified:
            raise AuthError("Email already registered")

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
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise AuthError(
            "Email already registered — check your inbox for a verification link, "
            "or use 'resend verification' if it expired."
        )
    await db.refresh(user)

    await send_verification_email(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    email = email.strip().lower()
    await check_login_lockout(email)

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or user.hashed_password is None or not verify_password(password, user.hashed_password):
        await record_login_failure(email)
        raise AuthError("Incorrect email or password")

    if not user.email_verified:
        raise AuthError("Please verify your email before logging in.")

    if not user.is_active:
        raise AuthError("Account is disabled")

    await reset_login_failures(email)
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
    email = email.strip().lower()
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user and not user.email_verified:
        await send_verification_email(user)


async def issue_token_pair(
    db: AsyncSession,
    user: User,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> tuple[str, str]:
    """Creates a fresh access token (stateless) + refresh token (DB row).
    """
    access_token = create_access_token(user.id, user.role.value)
    refresh_token_str, jti, expires_at = create_refresh_token(user.id)

    db.add(RefreshToken(
        user_id=user.id,
        jti=jti,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent,
    ))
    await db.commit()

    return access_token, refresh_token_str


async def rotate_refresh_token(
    db: AsyncSession,
    refresh_token_str: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> tuple[str, str]:
    """
    Validates a refresh token, revokes it, and issues a brand new pair.
    """
    payload = decode_token(refresh_token_str)
    if payload.get("type") != "refresh":
        raise AuthError("Invalid token type")

    jti = payload.get("jti")
    user_id = payload.get("sub")

    result = await db.execute(select(RefreshToken).where(RefreshToken.jti == UUID(jti)))
    stored = result.scalar_one_or_none()

    if stored is None or stored.revoked:
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

    return await issue_token_pair(db, user, ip_address=ip_address, user_agent=user_agent)


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

async def create_login_code(user_id) -> str:
    code = secrets.token_urlsafe(32)
    await redis_client.set(f"login_code:{code}", str(user_id), ex=LOGIN_CODE_TTL_SECONDS)
    return code


async def redeem_login_code(code: str) -> str:
    key = f"login_code:{code}"
    user_id_str = await redis_client.get(key)
    if not user_id_str:
        raise AuthError("Invalid or expired login code.")
    await redis_client.delete(key)  # single use
    return user_id_str


# ── Password reset ───────────────────────────────────────────────────────

async def request_password_reset(db: AsyncSession, email: str) -> None:
    email = email.strip().lower()
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or user.hashed_password is None:
        return

    token = secrets.token_urlsafe(32)
    await redis_client.set(f"password_reset:{token}", str(user.id), ex=PASSWORD_RESET_TTL_SECONDS)
    email_service.send_password_reset_email(user.email, token)


async def reset_password(db: AsyncSession, token: str, new_password: str) -> User:
    key = f"password_reset:{token}"
    user_id_str = await redis_client.get(key)
    if not user_id_str:
        raise AuthError("Invalid or expired password reset link.")
    await redis_client.delete(key)  # single use

    result = await db.execute(select(User).where(User.id == UUID(user_id_str)))
    user = result.scalar_one_or_none()
    if not user:
        raise AuthError("User not found.")

    user.hashed_password = hash_password(new_password)
    await db.commit()

    await revoke_all_user_tokens(db, user.id)
    await db.refresh(user)
    return user


# ── Account deactivation ─────────────────────────────────────────────────

async def deactivate_account(db: AsyncSession, user: User, password: Optional[str]) -> None:
    """Soft delete: is_active=False, every refresh token revoked.
    Local accounts must confirm their password; google-only accounts
    (hashed_password is None) have nothing to check beyond the caller
    already holding a valid access token for this user."""
    if user.hashed_password is not None:
        if not password or not verify_password(password, user.hashed_password):
            raise AuthError("Incorrect password.")

    user.is_active = False
    await db.commit()
    await revoke_all_user_tokens(db, user.id)


# ── Session management ───────────────────────────────────────────────────

async def list_sessions(db: AsyncSession, user_id: UUID) -> list[RefreshToken]:
    """Active (non-revoked, non-expired) sessions for a user, newest
    first. Each row corresponds to one issued refresh token — i.e.
    roughly one logged-in device/browser."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(RefreshToken)
        .where(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked == False,  # noqa: E712
            RefreshToken.expires_at > now,
        )
        .order_by(RefreshToken.created_at.desc())
    )
    return list(result.scalars().all())


async def revoke_session(db: AsyncSession, user_id: UUID, jti: UUID) -> None:
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.jti == jti, RefreshToken.user_id == user_id)
    )
    stored = result.scalar_one_or_none()
    if stored is None:
        raise AuthError("Session not found.")
    stored.revoked = True
    await db.commit()


# ── TOTP-based 2FA ────────────────────────────────────────────────────────
# Enrollment is two steps on purpose: setup_totp() generates and stores
# a secret WITHOUT enabling 2FA yet, so a user who never completes
# setup (closes the tab, scans the QR into the wrong app) can't
# accidentally lock themselves out — 2FA only actually turns on once
# confirm_totp() proves they can produce a valid code from it.

async def setup_totp(db: AsyncSession, user: User) -> tuple[str, str]:
    from backend.services.auth import totp as totp_service

    secret = totp_service.generate_secret()
    user.totp_secret = secret
    await db.commit()
    return secret, totp_service.provisioning_uri(secret, user.email)


async def confirm_totp(db: AsyncSession, user: User, code: str) -> None:
    from backend.services.auth import totp as totp_service

    if not user.totp_secret:
        raise AuthError("No 2FA setup in progress. Call /2fa/setup first.")
    if not totp_service.verify_totp(user.totp_secret, code):
        raise AuthError("Invalid code.")
    user.totp_enabled = True
    await db.commit()


async def disable_totp(db: AsyncSession, user: User, password: Optional[str], code: str) -> None:
    from backend.services.auth import totp as totp_service

    if user.hashed_password is not None:
        if not password or not verify_password(password, user.hashed_password):
            raise AuthError("Incorrect password.")
    if not user.totp_enabled or not user.totp_secret or not totp_service.verify_totp(user.totp_secret, code):
        raise AuthError("Invalid code.")

    user.totp_enabled = False
    user.totp_secret = None
    await db.commit()


async def create_mfa_token(user_id) -> str:

    token = secrets.token_urlsafe(32)
    await redis_client.set(f"mfa_token:{token}", str(user_id), ex=MFA_TOKEN_TTL_SECONDS)
    return token


async def redeem_mfa_token(token: str) -> str:
    key = f"mfa_token:{token}"
    user_id_str = await redis_client.get(key)
    if not user_id_str:
        raise AuthError("Invalid or expired 2FA session. Please log in again.")
    return user_id_str  # NOT single-use here — a wrong TOTP code shouldn't burn it; see verify_mfa_code


async def verify_mfa_code(db: AsyncSession, mfa_token: str, code: str) -> User:
    from backend.services.auth import totp as totp_service

    user_id_str = await redeem_mfa_token(mfa_token)
    result = await db.execute(select(User).where(User.id == UUID(user_id_str)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active or not user.totp_enabled or not user.totp_secret:
        raise AuthError("Invalid or expired 2FA session. Please log in again.")

    if not totp_service.verify_totp(user.totp_secret, code):
        raise AuthError("Invalid code.")

    await redis_client.delete(f"mfa_token:{mfa_token}")  # now single-use, on success
    return user