"""
backend/routers/auth/local.py

Email + password auth: register, login, email verification, refresh,
logout, /me, plus password reset, account deactivation, session
management, and TOTP 2FA. 
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.db_models import User
from backend.models.schemas import (
    UserCreate, UserLogin, UserOut, TokenPair, RefreshRequest,
    LoginResult, MFALoginRequest, ForgotPasswordRequest, ResetPasswordRequest,
    DeactivateAccountRequest, SessionOut, TOTPSetupResponse, TOTPCodeRequest,
    TOTPDisableRequest,
)
from backend.services.auth import core as auth_service
from backend.services.auth import audit as audit_service
from backend.services.auth.rate_limit import check_rate_limit, RateLimitError

router = APIRouter()


class VerifyEmailRequest(BaseModel):
    token: str

class ResendVerificationRequest(BaseModel):
    email: str


def _client_ip(request: Request) -> str:

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _user_agent(request: Request) -> str:
    return request.headers.get("user-agent", "unknown")


def _rate_limit_http(e: RateLimitError) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        str(e),
        headers={"Retry-After": str(e.retry_after_seconds or 60)},
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, request: Request, db: AsyncSession = Depends(get_db)):
    # No response_model=TokenPair — register no longer logs the user
    # in. It creates an unverified row and sends a verification email.
    # Real tokens only come from /verify-email or /login, once
    # email_verified is True.
    ip = _client_ip(request)
    try:
        await check_rate_limit(f"ratelimit:register:{ip}", max_attempts=5, window_seconds=3600)
    except RateLimitError as e:
        raise _rate_limit_http(e)

    if len(payload.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        await auth_service.create_user(db, payload.email, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))

    await audit_service.log_event(
        db, "register", email=payload.email.strip().lower(), ip_address=ip, user_agent=_user_agent(request)
    )
    return {"message": "Registered. Check your email to verify your account."}


@router.post("/login", response_model=LoginResult)
async def login(payload: UserLogin, request: Request, db: AsyncSession = Depends(get_db)):
    ip = _client_ip(request)
    ua = _user_agent(request)
    email = payload.email.strip().lower()

    try:
        user = await auth_service.authenticate_user(db, payload.email, payload.password)
    except RateLimitError as e:
        raise _rate_limit_http(e)
    except auth_service.AuthError as e:
        await audit_service.log_event(db, "login_failed", email=email, ip_address=ip, user_agent=ua, detail=str(e))
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    if user.totp_enabled:
        mfa_token = await auth_service.create_mfa_token(user.id)
        await audit_service.log_event(db, "login_mfa_required", email=email, user_id=user.id, ip_address=ip, user_agent=ua)
        return LoginResult(mfa_required=True, mfa_token=mfa_token)

    access_token, refresh_token = await auth_service.issue_token_pair(db, user, ip_address=ip, user_agent=ua)
    await audit_service.log_event(db, "login_success", email=email, user_id=user.id, ip_address=ip, user_agent=ua)
    return LoginResult(access_token=access_token, refresh_token=refresh_token)


@router.post("/login/verify-2fa", response_model=LoginResult)
async def login_verify_2fa(payload: MFALoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    ip = _client_ip(request)
    ua = _user_agent(request)
    try:
        user = await auth_service.verify_mfa_code(db, payload.mfa_token, payload.code)
    except auth_service.AuthError as e:
        await audit_service.log_event(db, "login_mfa_failed", ip_address=ip, user_agent=ua, detail=str(e))
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    access_token, refresh_token = await auth_service.issue_token_pair(db, user, ip_address=ip, user_agent=ua)
    await audit_service.log_event(db, "login_mfa_success", email=user.email, user_id=user.id, ip_address=ip, user_agent=ua)
    return LoginResult(access_token=access_token, refresh_token=refresh_token)


@router.post("/verify-email", response_model=TokenPair)
async def verify_email(payload: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
 
    try:
        user = await auth_service.verify_email_token(db, payload.token)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    access_token, refresh_token = await auth_service.issue_token_pair(db, user)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.get("/verify-email")
async def verify_email_get(token: str, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_service.verify_email_token(db, token)
    except auth_service.AuthError as e:
        return RedirectResponse(f"{settings.FRONTEND_URL}?auth_error={e}")

    login_code = await auth_service.create_login_code(user.id)
    return RedirectResponse(f"{settings.FRONTEND_URL}?login_code={login_code}")


@router.post("/resend-verification", status_code=status.HTTP_204_NO_CONTENT)
async def resend_verification(payload: ResendVerificationRequest, db: AsyncSession = Depends(get_db)):
    email = payload.email.strip().lower()
    try:
        await check_rate_limit(f"ratelimit:resend:{email}", max_attempts=3, window_seconds=3600)
    except RateLimitError as e:
        raise _rate_limit_http(e)
    await auth_service.resend_verification(db, payload.email)

@router.post("/forgot-password", status_code=status.HTTP_204_NO_CONTENT)
async def forgot_password(payload: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    email = payload.email.strip().lower()
    try:
        await check_rate_limit(f"ratelimit:forgot-password:{email}", max_attempts=3, window_seconds=3600)
    except RateLimitError as e:
        raise _rate_limit_http(e)
    await auth_service.request_password_reset(db, payload.email)

@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(payload: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    if len(payload.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        await auth_service.reset_password(db, payload.token, payload.new_password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, request: Request, db: AsyncSession = Depends(get_db)):
    try:
        access_token, refresh_token = await auth_service.rotate_refresh_token(
            db, payload.refresh_token, ip_address=_client_ip(request), user_agent=_user_agent(request)
        )
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    payload: RefreshRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await auth_service.revoke_refresh_token(db, payload.refresh_token)
    await audit_service.log_event(db, "logout", email=current_user.email, user_id=current_user.id)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/deactivate", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate(
    payload: DeactivateAccountRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await auth_service.deactivate_account(db, current_user, payload.password)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))
    await audit_service.log_event(db, "account_deactivated", email=current_user.email, user_id=current_user.id)


# ── Sessions ──────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions_route(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sessions = await auth_service.list_sessions(db, current_user.id)
    current_ua = _user_agent(request)
    current_ip = _client_ip(request)
    return [
        SessionOut(
            jti=s.jti,
            created_at=s.created_at,
            expires_at=s.expires_at,
            user_agent=s.user_agent,
            ip_address=s.ip_address,
            current=(s.user_agent == current_ua and s.ip_address == current_ip),
        )
        for s in sessions
    ]


@router.delete("/sessions/{jti}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_session_route(
    jti: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        jti_uuid = UUID(jti)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid session id.")
    try:
        await auth_service.revoke_session(db, current_user.id, jti_uuid)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    await audit_service.log_event(db, "session_revoked", email=current_user.email, user_id=current_user.id, detail=jti)


# ── TOTP 2FA ──────────────────────────────────────────────────────────────

@router.post("/2fa/setup", response_model=TOTPSetupResponse)
async def setup_2fa(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if current_user.totp_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "2FA is already enabled. Disable it first to re-enroll.")
    secret, uri = await auth_service.setup_totp(db, current_user)
    return TOTPSetupResponse(secret=secret, otpauth_uri=uri)


@router.post("/2fa/verify", status_code=status.HTTP_204_NO_CONTENT)
async def verify_2fa(
    payload: TOTPCodeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await auth_service.confirm_totp(db, current_user, payload.code)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    await audit_service.log_event(db, "2fa_enabled", email=current_user.email, user_id=current_user.id)


@router.post("/2fa/disable", status_code=status.HTTP_204_NO_CONTENT)
async def disable_2fa(
    payload: TOTPDisableRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await auth_service.disable_totp(db, current_user, payload.password, payload.code)
    except auth_service.AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))
    await audit_service.log_event(db, "2fa_disabled", email=current_user.email, user_id=current_user.id)