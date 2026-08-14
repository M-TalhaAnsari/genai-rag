# Setup — env vars, dependencies, wiring

## 1. New env vars — add to `core/config.py`'s `settings` object

```python
JWT_SECRET_KEY: str          # required — generate with: openssl rand -hex 32
JWT_ALGORITHM: str = "HS256"
JWT_ACCESS_EXPIRE_MINUTES: int = 30
JWT_REFRESH_EXPIRE_DAYS: int = 14
```

Add to `.env`:
```
JWT_SECRET_KEY=<output of: openssl rand -hex 32>
JWT_ALGORITHM=HS256
JWT_ACCESS_EXPIRE_MINUTES=30
JWT_REFRESH_EXPIRE_DAYS=14
```

Never commit the real secret. Rotating `JWT_SECRET_KEY` invalidates
every issued access token instantly (and every refresh token, since
they're signed with the same key) — useful as a manual "kill switch"
if a key ever leaks.

## 2. New dependencies

```bash
pip install "python-jose[cryptography]" "passlib[bcrypt]" python-multipart email-validator
```
- `python-jose` — JWT encode/decode
- `passlib[bcrypt]` — password hashing
- `python-multipart` — required by FastAPI's `OAuth2PasswordBearer` form handling (even though your login route takes JSON, not form data, FastAPI's security utilities import this)
- `email-validator` — backs Pydantic's `EmailStr`

## 3. Mount the auth router in `main.py`

```python
from backend.routers import auth as auth_router
app.include_router(auth_router.router)
```

Stays consistent with "main.py only creates app + mounts routers" —
no logic added there.

## 4. Database migration

You'll need an Alembic migration (or manual SQL if you're not using
Alembic yet) for:
- new `users` table
- new `refresh_tokens` table
- `user_id` column type change on `UserFeedback`, `UserProfile`,
  `ConversationHistory`, `UserMemorySummary` — String → UUID FK

Since your current `user_id` values are almost certainly placeholder
test strings (not real UUIDs), the cleanest path is dropping and
recreating those four tables rather than trying to migrate the data —
worth confirming that's fine before running it, since it's a real data
loss step even for test data.

## 5. What did NOT change

- `contact_service.py` — deliberately public, no auth needed (it just
  builds `mailto:`/`wa.me` links, no user data involved)
- `/search`, `/search/full` — worth deciding: do these stay public
  (read-only, no personalization) or require auth too? Right now
  nothing in the design forces this either way. If you want anonymous
  browsing before signup, leave `/search` public and only gate
  `/recommend` (which needs profile personalization anyway) and
  anything writing data.
