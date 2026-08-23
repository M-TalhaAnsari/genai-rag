"""
backend/services/auth/google_oauth.py  (formerly services/google_oauth_service.py)

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
  6. resolve_google_identity() — match by google_id (stable across
     email changes) first. New Google-only user if no email
     collision. If the email belongs to an existing LOCAL account,
     return a pending-link payload instead of touching the DB — the
     caller must collect that account's password before merging
     (see create_link_token / redeem_link_token).
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
from backend.services.auth.core import AuthError

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

STATE_TTL_SECONDS = 600       # 10 min to complete the consent screen
LOGIN_CODE_TTL_SECONDS = 60   # frontend should exchange this immediately
LINK_TOKEN_TTL_SECONDS = 300  # 5 min to enter password


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


async def resolve_google_identity(db: AsyncSession, payload: dict) -> tuple[User | None, dict | None]:
    """
    Returns (user, None) if login can proceed immediately — either an
    exact google_id match, or a brand-new Google-only user with no
    email collision.

    Returns (None, pending) if this email already belongs to a LOCAL
    account that has never been linked to this Google identity. The
    caller must NOT touch the DB in this case — the pending dict is
    handed to create_link_token() so the frontend can collect the
    account's password before we merge anything.
    """
    google_id = payload["sub"]
    email = payload["email"]

    result = await db.execute(select(User).where(User.google_id == google_id))
    user = result.scalar_one_or_none()
    if user:
        return user, None

    result = await db.execute(select(User).where(User.email == email))
    existing = result.scalar_one_or_none()
    if existing:
        # Do NOT set existing.google_id here. We don't yet know that
        # whoever is sitting at this browser controls the existing
        # account's password — only that Google says they control
        # this email. Those are different claims.
        return None, {"google_id": google_id, "email": email}

    user = User(
        email=email,
        hashed_password=None,
        google_id=google_id,
        auth_provider="google",
        role=UserRole.user,
        email_verified=True,   # Google already proved this
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user, None


async def create_link_token(google_id: str, email: str) -> str:
    token = secrets.token_urlsafe(32)
    await redis_client.set(
        f"link_token:{token}",
        f"{google_id}:{email}",
        ex=LINK_TOKEN_TTL_SECONDS,
    )
    return token


async def redeem_link_token(token: str) -> tuple[str, str]:
    key = f"link_token:{token}"
    raw = await redis_client.get(key)
    if not raw:
        raise AuthError("Invalid or expired link request. Please sign in with Google again.")
    await redis_client.delete(key)  # single use
    google_id, email = raw.split(":", 1)
    return google_id, email