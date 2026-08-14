"""
backend/core/security.py

SINGLE SOURCE OF TRUTH for authentication + authorization.
Every router that needs to protect a route imports a dependency from
HERE and nowhere else. Do not reimplement token checks in individual
routers — that's exactly the scattering this file exists to prevent.

Two things live in this file, deliberately together:
  1. AUTHENTICATION — "who is making this request?" (JWT decode, password
     hashing)
  2. AUTHORIZATION   — "is this identity allowed to do this?" (role checks)

Keeping both in one file means there's exactly one place to audit when
you ask "how does access control work in this project?"
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_db
from backend.models.db_models import User, RefreshToken

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# JWT — access tokens (stateless, short-lived)
# ---------------------------------------------------------------------------

# Points the OpenAPI docs "Authorize" button at /auth/login. This does NOT
# create a route itself — routers/auth.py defines the real /auth/login.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def create_access_token(user_id: UUID, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_EXPIRE_MINUTES
    )
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# JWT — refresh tokens (DB-backed, revocable)
# ---------------------------------------------------------------------------
# Refresh tokens carry a `jti` (JWT ID). The jti is stored in Postgres so
# logout / revocation / reuse-detection is possible. The access token never
# touches the DB — that's what keeps normal request-path auth cheap.

def create_refresh_token(user_id: UUID) -> tuple[str, UUID, datetime]:
    jti = uuid4()
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_EXPIRE_DAYS
    )
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": str(jti),
        "exp": expire,
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, jti, expire


# ---------------------------------------------------------------------------
# Dependencies — use these in routers via Depends(...)
# ---------------------------------------------------------------------------

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    THE central authentication dependency. Decodes the access token,
    loads the User row, confirms the account is active.

    Every protected route — admin or regular user — depends on this
    (directly, or transitively via require_role). Routers never read a
    user_id from a query param / request body again; identity comes only
    from this dependency.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


def require_role(*allowed_roles: str):
    """
    Authorization layer, built ON TOP of get_current_user.

    Usage:
        @router.post("/ingestion/...", dependencies=[Depends(require_role("admin"))])

    This is the ONLY place role-checking logic exists. If you ever add a
    third role, this is the only function that changes.
    """

    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {' or '.join(allowed_roles)}",
            )
        return current_user

    return _check


# Convenience aliases — read intent at the call site in routers.
require_user = require_role("user", "admin")  # any authenticated identity
require_admin = require_role("admin")          # admin console only