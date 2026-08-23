"""
backend/services/auth/email.py

Outbound email for auth flows (currently just verification links).
No FastAPI imports — same convention as every services/ file.

DEV MODE: if SMTP_HOST isn't set in .env, send_verification_email()
prints the link to the console instead of failing, so the full
register -> verify -> login loop works locally with zero email setup.
Set real SMTP_* values in .env when you're ready to actually deliver
mail.
"""

import smtplib
from email.mime.text import MIMEText

from backend.core.config import settings


def _send_raw(to_email: str, subject: str, body: str) -> None:
    if not settings.SMTP_HOST:
        print(f"\n[DEV EMAIL] To: {to_email}\nSubject: {subject}\n{body}\n")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM
    msg["To"] = to_email

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.starttls()
        if settings.SMTP_USER:
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_FROM, [to_email], msg.as_string())


def send_verification_email(to_email: str, token: str) -> None:
    link = f"{settings.FRONTEND_URL}?verify_token={token}"
    _send_raw(
        to_email,
        "Verify your Connoisseur account",
        f"Click to verify your email (expires in 24 hours):\n\n{link}",
    )