"""
backend/services/google_oauth_service.py

Google OAuth2 Authorization Code flow. No FastAPI imports — same
convention as every other services/ file.

Flow:
  1. build_authorization_url() — random `state` (CSRF protection)
     stored in Redis with a short TTL. Points the browser at Google's
     consent screen.
  2. Google redirects back to our /auth/google/callback with a `code`
     and the `state` we sent.
  3. verify_state() — confirm the state matches what we issued and
     hasn't been used before (deleted on read = single use).
  4. exchange_code_for_id_token() — trade the code for Google's
     id_token via a direct POST to Google's token endpoint.
  5. verify_google_id_token() — validate the id_token's signature
     against Google's public keys via the `google-auth` library. This
     is the one part that is NOT hand-rolled — JWKS fetching/caching
     and RS256 verification is real cryptographic surface area, and
     Google maintains this library specifically so nobody has to.
  6. get_or_create_google_user() — match by google_id (stable across
     email changes) first, fall back to matching a verified email to
     link an existing local account, otherwise create a new
     Google-only user (no password).
  7. create_login_code() / redeem_login_code() — instead of putting
     real tokens in the redirect URL back to the frontend (browser
     history + referrer header exposure), we hand back a one-time,
     60-second code. The frontend immediately exchanges it for the
     real access+refresh pair via a POST body.
"""

import secrets
from urllib.parse import urlencode

import httpx
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.redis_client import redis_client
from backend.models.db_models import User, UserRole
from backend.services.auth_service import AuthError

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

STATE_TTL_SECONDS = 600      # 10 min to complete the consent screen
LOGIN_CODE_TTL_SECONDS = 60  # frontend should exchange this immediately


async def build_authorization_url() -> str:
    state = secrets.token_urlsafe(32)
    await redis_client.set(f"oauth_state:{state}", "1", ex=STATE_TTL_SECONDS)

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",  # we mint our own refresh token — don't need Google's
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"


async def verify_state(state: str) -> None:
    key = f"oauth_state:{state}"
    exists = await redis_client.get(key)
    if not exists:
        raise AuthError("Invalid or expired OAuth state — possible CSRF attempt.")
    await redis_client.delete(key)  # single use


async def exchange_code_for_id_token(code: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(GOOGLE_TOKEN_ENDPOINT, data={
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        })
    if resp.status_code != 200:
        raise AuthError(f"Google token exchange failed: {resp.text}")

    data = resp.json()
    id_token_str = data.get("id_token")
    if not id_token_str:
        raise AuthError("Google did not return an id_token.")
    return id_token_str


def verify_google_id_token(id_token_str: str) -> dict:
    """Signature + audience + issuer check against Google's rotating
    public keys. Do not replace this with manual JWT decoding."""
    try:
        payload = google_id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        raise AuthError(f"Invalid Google id_token: {e}")

    if not payload.get("email_verified", False):
        raise AuthError("Google account email is not verified.")

    return payload  # sub (google_id), email, name, picture, ...


async def get_or_create_google_user(db: AsyncSession, payload: dict) -> User:
    google_id = payload["sub"]
    email = payload["email"]

    result = await db.execute(select(User).where(User.google_id == google_id))
    user = result.scalar_one_or_none()
    if user:
        return user

    # Link to an existing local account with the same verified email,
    # rather than creating a duplicate.
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        user.google_id = google_id
        if user.auth_provider == "local":
            user.auth_provider = "google_and_local"
        await db.commit()
        await db.refresh(user)
        return user

    user = User(
        email=email,
        hashed_password=None,
        google_id=google_id,
        auth_provider="google",
        role=UserRole.user,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


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