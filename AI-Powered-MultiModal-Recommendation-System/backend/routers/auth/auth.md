# Auth system — `backend/services/auth/` + `backend/routers/auth/`

Self-hosted email/password + Google OAuth, JWT access/refresh tokens,
TOTP 2FA, session management, password reset, account deactivation,
rate limiting, and an audit log. No third-party auth provider — admin
is a `role` column on the same `User` table, not a separate system.

Read this before touching anything under `services/auth/` or
`routers/auth/` — several of the design choices here exist because an
earlier, simpler version had a real bug, and the fix is easy to
accidentally undo by "simplifying" it back.

---

## File map

```
services/auth/
├── errors.py       AuthError — its own module specifically so core.py
│                    and rate_limit.py can both depend on it without
│                    importing each other (rate_limit.RateLimitError
│                    subclasses AuthError)
├── core.py          password hashing, JWT issue/rotate/revoke, email
│                     verification, password reset, account deactivation,
│                     session list/revoke, TOTP enrollment/verification
├── google_oauth.py  Authorization Code flow + account linking (Case 2)
├── email.py         outbound email — verification + password reset links
├── rate_limit.py     Redis fixed-window rate limiting + login lockout
├── audit.py           append-only writes to the audit_logs table
└── totp.py             RFC 6238 TOTP, stdlib only (hmac/hashlib/base64/struct)

routers/auth/
├── local.py    everything except Google — 20 routes total across both files
├── google.py    OAuth login/callback/exchange/link-confirm
└── auth.md      session-by-session build notes — HISTORY, not current-state
                  truth. Useful for "why does this exist" archaeology; this
                  CLAUDE.md file is the one to trust for current behavior.
```

---

## Why `AuthError` lives in its own file

`core.py`'s `authenticate_user()` needs to check login lockout, which
lives in `rate_limit.py`. `rate_limit.py`'s `RateLimitError` needs to
subclass `AuthError` so routers can catch either uniformly. If
`AuthError` lived in `core.py`, that would make `rate_limit.py` import
`core.py` AND `core.py` import `rate_limit.py` — a circular import that
fails at module load time, not at some obscure runtime edge case. Verified
concretely: `issubclass(rate_limit.RateLimitError, auth_service.AuthError)`
and `auth_service.AuthError is google_oauth_service.AuthError` are both
`True` at runtime with this structure — don't reintroduce the cycle by
moving `AuthError` back into `core.py` "for simplicity."

## Exception ordering — the one thing every router handler must get right

`RateLimitError` **subclasses** `AuthError`. Every `except` block that
handles both must catch `RateLimitError` FIRST:

```python
try:
    user = await auth_service.authenticate_user(db, email, password)
except RateLimitError as e:          # MUST come first
    raise _rate_limit_http(e)
except auth_service.AuthError as e:  # catches everything else
    raise HTTPException(401, str(e))
```

Reversed, the generic `AuthError` branch would swallow lockouts and
return a misleading 401 instead of 429 — the person would think their
password is wrong, not that they're locked out. This bit `login()` in
`routers/auth/local.py` during development; check any new endpoint that
touches `authenticate_user()` or the rate-limited endpoints for this
same ordering.

## Redis key namespaces

Every short-lived token uses its own prefix — deliberately, so a leak
of one Redis key never enables anything beyond its specific flow:

| Prefix | Set by | TTL | Notes |
|---|---|---|---|
| `email_verify:{token}` | `send_verification_email` | 24h | single-use |
| `login_code:{code}` | `create_login_code` | 60s | **shared** between the Google-exchange flow and the email-verify-link flow — `/auth/google/exchange` doesn't care which flow produced the code, it just redeems whatever's under this key. Deliberate reuse, not an accident. |
| `link_token:{token}` | google_oauth.py (Case 2) | short | account-linking confirmation |
| `password_reset:{token}` | `request_password_reset` | 30 min | single-use, revokes ALL sessions on success |
| `mfa_token:{token}` | `create_mfa_token` | 5 min | issued after a correct password when 2FA is on; **not single-use on a wrong code** — see below |
| `login_fail:{email}` | `record_login_failure` | 15 min | failure counter |
| `login_lockout:{email}` | `record_login_failure` at 5 failures | 15 min | the actual lockout flag |
| `ratelimit:register:{ip}` | `check_rate_limit` | 1h | 5/hr |
| `ratelimit:resend:{email}` / `ratelimit:forgot-password:{email}` | `check_rate_limit` | 1h | 3/hr each |

**Why `mfa_token` isn't burned on a wrong code:** `redeem_mfa_token()`
only reads the key; `verify_mfa_code()` deletes it only after a
*correct* TOTP code. If a wrong code burned the token immediately, a
user who fat-fingers their 6 digits would have to re-enter their
password to get a new `mfa_token` — annoying and unnecessary, since the
token only proves "this caller already knows the password," not
anything about the TOTP code itself. Don't "fix" this into single-use;
it's already correct.

## Login lockout is keyed by email, not IP

An attacker rotating IPs still gets locked out this way. The tradeoff —
someone else could theoretically get locked out if an attacker targets
their exact email from many IPs — is the standard cost every
account-lockout scheme accepts. This is checked in `authenticate_user()`
**before the database is even touched**, so a locked-out email doesn't
leak whether the account exists via response timing either.

## Password reset revokes every session, not just the current one

`reset_password()` calls `revoke_all_user_tokens()` on success. A
password reset is the highest-confidence "this session may be
compromised" signal short of an explicit report — a stolen session
shouldn't outlive the password that (maybe) leaked it. Don't scope this
down to "just the token that requested the reset."

## Account deactivation and 2FA-disable both skip the password check for Google-only accounts

`user.hashed_password is None` means the account was created purely via
Google OAuth and never set a local password. `deactivate_account()` and
`disable_totp()` both check `if user.hashed_password is not None:
require password` — for a Google-only account, holding a valid access
token (i.e., already being logged in) is the only re-auth available,
since there's no password to check. Don't add a hard password
requirement here; it would make these actions impossible for Google-only
accounts.

## 2FA enrollment is two steps on purpose

`setup_totp()` generates and stores a secret WITHOUT setting
`totp_enabled = True`. Only `confirm_totp()` — which requires a valid
code from that secret — flips the flag. This means a user who scans the
QR into the wrong app, or just closes the tab mid-setup, can't
accidentally lock themselves out of their own account. Don't collapse
this into one step "for simplicity" — the two-step design IS the safety
mechanism.

## TOTP is hand-rolled, not `pyotp` — verified against the actual RFC

`totp.py` is ~50 lines against RFC 4226 (HOTP) / RFC 6238 (TOTP), stdlib
only. This was a deliberate choice over adding a dependency for
something this security-sensitive and this small — but "small and
hand-rolled" only stays safe if it's actually correct, so it was
verified against all 10 official RFC 4226 Appendix D test vectors
before being trusted:

```python
secret_ascii = b'12345678901234567890'  # the RFC's own test secret
# counters 0-9 → 755224, 287082, 359152, 969429, 338314,
#                254676, 287922, 162583, 399871, 520489
# all 10 matched exactly
```

If you ever touch `_hotp()` or `verify_totp()`, re-run this test before
trusting the change — a subtly wrong TOTP implementation fails silently
(codes just don't verify) rather than throwing an obvious error.

## The `hashed_password` migration trap — read this before running `alembic revision --autogenerate`

`User.hashed_password` in `db_models.py` must stay `nullable=True`
(Google-only accounts have no password). It was `nullable=False` in the
original schema, then made nullable in `0002_google_oauth.py` — but the
**model file itself was never updated to match**, so any later
`alembic revision --autogenerate` will see model (`nullable=False`) vs.
actual DB (`nullable=True`) as drift and generate a migration that
silently flips it back to `NOT NULL`. This actually happened once
(migration `65b9f0789708`, originally about an unrelated refresh-token
timezone fix) and was caught and stripped out before causing damage.
The model now correctly says `nullable=True` — if you ever see
autogenerate propose an `alter_column('users', 'hashed_password',
nullable=False)`, that's this same class of bug resurfacing, not a
legitimate change to make.

## Account linking (Case 2) reuses the `login_code` Redis namespace

When someone with an existing local/password account clicks "Continue
with Google" using the same email, `google_oauth.py` doesn't silently
merge accounts — it issues a `link_token` and redirects the frontend to
a "confirm with your password" screen (`_render_link_confirm()` in
`login_page.py`). Only after that password check succeeds does it issue
real tokens. Don't skip the password confirmation step even though it
feels redundant with "they already proved they own the Google account"
— the whole point is proving they *also* own the local account, since
Google's own email verification and this app's are two different trust
chains.

---

## What's NOT implemented here (see root CLAUDE.md's cleanup table / README for status)

- No rate limiting on `/login` beyond the lockout (lockout IS the login
  rate limit — there's no separate per-IP throttle on it, only per-email)
- Session list (`GET /auth/sessions`) marks "current" via IP+user-agent
  match, not an exact identifier — the access token itself carries no
  `jti` (only the refresh token does), so this is a best-effort signal,
  not a guarantee, on shared IPs (e.g. two tabs behind the same NAT/proxy)
- No admin-facing view of the `audit_logs` table yet — it's written to,
  but nothing reads it back out through the API. Query it directly in
  Postgres for now.