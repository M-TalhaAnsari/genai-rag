"""
backend/services/auth/audit.py

Append-only audit trail for auth events: login/logout, failed
logins, password resets, account linking/deactivation, 2FA changes.

Best-effort by design: a failure to write an audit row must never
break the auth action it's describing, so failures are swallowed here
rather than left for every call site to remember to guard against.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import AuditLog


async def log_event(
    db: AsyncSession,
    event_type: str,
    email: Optional[str] = None,
    user_id: Optional[UUID] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    try:
        db.add(AuditLog(
            event_type=event_type,
            email=email,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            detail=detail,
        ))
        await db.commit()
    except Exception:
        await db.rollback()