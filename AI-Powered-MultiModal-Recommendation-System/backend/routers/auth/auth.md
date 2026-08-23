# Connoisseur — Auth System Status

Last updated: 2026-08-23

This tracks what's been built, what's actually been tested end-to-end, and what's
still missing before this auth system is production-ready. Use the checkboxes as
your working list.

---

## 1. Architecture summary

Self-hosted auth (no third-party auth provider like Auth0/Clerk). Two entry
points that converge on the same JWT session system:

- **Email + password**, gated by email verification
- **Google OAuth (Authorization Code flow)**, using OpenID Connect's
  `id_token` for identity

```
Local:
  register → row created, email_verified=False → email sent → NOT logged in
  login    → blocked until email_verified=True
  verify-email(token) → flips email_verified=True → issues tokens

Google:
  callback → google_id already known? → login_code → exchange → tokens
           → new email, no local match? → create verified user → login_code → tokens
           → email matches an existing LOCAL account? → link_token
                → frontend prompts password → link-confirm → tokens
```

### Why `google_id` (the `sub` claim), not email, is the identity key

Email can change or be reassigned; Google's `sub` claim is permanent for a
given Google account. `google_id` is the column matched first on every
Google login — email is only a fallback used to detect a linking scenario.

### Why account linking requires a password, not just an email match

Silently linking a Google identity to any local account with a matching
email is an account-takeover vector: an attacker could register locally
using someone else's email (nothing stops that at register time — see
open gap below), then have the real owner's later "Continue with Google"
click silently attached to the attacker's local account. Requiring the
local account's password before linking closes this.

### Why register no longer issues a token pair

Previously, `/register` returned real tokens immediately — meaning an
unverified account got a live session before anyone proved the email was
real. That's the same class of trust mistake as auto-linking. Register's
job is now narrower: create an inert, unverified row and send proof of
ownership to the one channel that matters (the inbox). Only
`/verify-email` (proof arrived) or `/login` (proof already happened
earlier) issue real sessions.

---

## 2. What's implemented

| Component | File |
|---|---|
| Password hashing, JWT encode/decode | `core/security.py` |
| Register / login / refresh / logout business logic | `services/auth_service.py` |
| Email verification (token issue/redeem, stale-account reclaim) | `services/auth_service.py` |
| Dev-mode email sending (console fallback, real SMTP when configured) | `services/email_service.py` |
| Google OAuth flow (state, token exchange, id_token verification, get-or-create) | `services/google_oauth_service.py` |
| All auth HTTP routes | `routers/auth.py` |
| `email_verified` column + backfill for pre-existing Google users | `alembic/versions/0003_email_verification.py` |

### Security properties in place

- CSRF protection on OAuth via single-use, Redis-backed `state` (10 min TTL)
- `id_token` signature verified via Google's official `google-auth` library
  (JWKS fetch/cache/RS256 handled by Google, not hand-rolled)
- `email_verified` claim from Google checked before trusting a Google email
- Real tokens never appear in a redirect URL — one-time `login_code`
  (60s TTL) is exchanged for the real pair via a POST body
- Refresh tokens rotate on every use; reuse of an already-used refresh
  token is detectable (`revoked` flag + `jti` tracking)
- Account linking requires proof of the local account's password
  (`link_token`, 5 min TTL, single use)
- Generic "incorrect email or password" / no-op `resend-verification`
  responses — neither leaks whether an email is registered

---

## 3. Test status

### ✅ Confirmed working (manually tested this session)

- [x] Local register → row created with `email_verified=False`, no session issued
- [x] Login blocked with "please verify your email" before verification
      *(note: this only fires after email/password match — verify your
      curl bodies are byte-identical between register and login if you
      see "incorrect email or password" instead)*
- [x] Google OAuth — brand new user (no prior local account)
  - `email_verified=True`, `auth_provider='google'`, `hashed_password` null
  - Confirmed via direct DB query
- [x] CSRF `state` protection — reusing an old callback URL correctly
      fails with "Invalid or expired OAuth state"
- [x] Clock-skew sensitivity in `id_token` verification — hit and fixed
      a real "Token used too early" error (Windows Time service issue)
- [x] One-time `login_code` → `/auth/google/exchange` → real token pair

### ⏳ Implemented but not yet manually verified

- [ ] `/auth/verify-email` — flips `email_verified`, issues tokens
      *(logic shared with tested paths, but never run directly with a
      real printed token)*
- [ ] `/auth/resend-verification` — sends a fresh token; returns 204
      even for unknown/already-verified emails
- [ ] **Account linking (Case 2)** — existing local account + Google
      sign-in with the same email → `link_required` redirect →
      `/auth/google/link-confirm` with local password → `google_id`
      attached, `auth_provider` becomes `'google_and_local'`
      **This is the actual security fix from this session — test it
      before considering auth "done."**
- [ ] Google OAuth — already-linked user logs in again → should go
      straight to `login_code`, no `link_required` prompt
- [ ] Refresh token rotation — old token fails after a new one is issued
- [ ] Logout — refresh token fails after logout

### ⚠️ Known gaps / open bugs

- [ ] **Duplicate-register on the same unverified email was never
      cleanly re-confirmed to return 409.** Was tested once with an
      ambiguous result (see conversation history) — re-run this
      specifically:
      ```bash
      curl -X POST http://localhost:8000/auth/register \
        -d '{"email":"same@example.com","password":"x"}'
      # then immediately again with the same email
      # second call should be 409, not 201
      ```
- [ ] **Email case-sensitivity** — `Test@x.com` and `test@x.com` are
      currently treated as different DB rows. Should lowercase email on
      both write and lookup, everywhere (register, login,
      resend-verification, Google linking).
- [ ] **No rate limiting** on `/register` or `/resend-verification` —
      an attacker can spam verification emails to any address.
- [ ] Minor race: two near-simultaneous registrations on the same
      stale-unverified email could both attempt delete+recreate.
      Low priority.
- [ ] `/auth/verify-email` is currently a **POST** requiring the token
      in a JSON body — meaning a real user can't just click the emailed
      link, they'd need a frontend page that reads the token and POSTs
      it. Consider converting to a GET route (token in query/path) that
      redirects to the frontend with a success/failure flag, so
      clicking the email link works directly.

---

## 4. Frontend (Streamlit) — not yet wired up

The backend redirects to `FRONTEND_URL` with query params the Streamlit
app needs to read via `st.query_params`:

| Redirect param | Meaning | Frontend action needed |
|---|---|---|
| `?login_code=...` | Google login succeeded | POST to `/auth/google/exchange`, store tokens in `st.session_state`, clear query param |
| `?link_required=...&email=...` | Existing local account, needs password to link | Show a password field, POST to `/auth/google/link-confirm` |
| `?auth_error=...` | Something failed (bad state, invalid token, etc.) | Display as an error message |

Additional frontend pieces needed:
- Register form → POST `/auth/register`, then show "check your email"
- Login form → POST `/auth/login`
- A way to handle the verification link (see the GET-route conversion
  note above — this is the cleanest fix)
- Token storage strategy in `st.session_state`, and wiring every
  authenticated API call to attach the access token + retry once
  through `/auth/refresh` on a 401 (this was mentioned as designed but
  not yet fully wired in an earlier session — confirm current state)

---

## 5. Beyond MVP — not started, not blocking

Standard production auth features not yet touched. None block shipping
the current scope, but list them here so they don't get forgotten:

- [ ] Password reset flow (forgot-password → email link → set new password)
- [ ] Account deletion / deactivation endpoint
- [ ] Session/device management (list + revoke individual sessions)
- [ ] Login attempt throttling / brute-force protection on `/login`
- [ ] Audit log of auth events (login, failed login, password change,
      account linked)
- [ ] Two-factor authentication (optional)

---

## 6. Environment variables required

```env
# Google OAuth — from console.cloud.google.com/apis/credentials
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback   # must match Console EXACTLY

# Frontend
FRONTEND_URL=http://localhost:8501

# Email (optional for dev — falls back to console-printed links if unset)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=noreply@connoisseur.app
```

**Common first-run failures and their fixes (all hit and resolved this session):**
- `AttributeError: 'Settings' object has no attribute 'GOOGLE_CLIENT_ID'`
  → field not declared on the `Settings` class in `core/config.py`, or
  `.env` not loaded
- `redirect_uri_mismatch` from Google → the URI in `.env` doesn't
  character-match what's registered in Google Cloud Console (trailing
  slash, `localhost` vs `127.0.0.1`)
- `Token used too early` → system clock drift; sync Windows Time
  service (`net start w32time` → `w32tm /resync /force`) or set the
  clock manually if NTP is blocked
- `Invalid or expired OAuth state` → reused an old `/callback` URL
  (browser back button, cached tab) instead of starting a fresh
  `/auth/google/login` call each time; `state` is single-use by design

---

## 7. Suggested next-session order

1. Re-confirm duplicate-register returns 409 (open question, never cleanly closed)
2. Fix email lowercasing (small, prevents a real class of bugs)
3. Test account linking (Case 2) end-to-end with a real second Gmail address
4. Convert `/verify-email` to a clickable GET route
5. Wire up Streamlit frontend (query param handling, forms, token storage, refresh-on-401)
6. Test refresh rotation + logout
7. Decide which "beyond MVP" items (section 5) are actually needed before real users touch this