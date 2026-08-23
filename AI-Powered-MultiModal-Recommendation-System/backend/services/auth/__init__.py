"""
backend/services/auth/__init__.py

Auth business logic, split by concern:
  - core.py         password/JWT/refresh/email-verification logic
  - google_oauth.py Google OAuth (Authorization Code flow)
  - email.py        outbound email sending (verification links)

Deliberately no imports here. core.py depends on email.py, and
google_oauth.py depends on core.py — re-exporting everything through
this __init__ would create a circular import the moment any of them
loads. Import submodules directly instead:

    from backend.services.auth import core as auth_service
    from backend.services.auth import google_oauth as google_oauth_service
    from backend.services.auth import email as email_service
"""