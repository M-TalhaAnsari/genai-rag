# Frontend — `frontend/`

Streamlit app, gated end-to-end by the auth system in
`backend/services/auth/CLAUDE.md`. Three files, one direction of
dependency: `app.py` depends on `login_page.py` and `api_client.py`;
neither of those depends on `app.py`.

```
login_page.py   Login/register/Google/2FA/password-reset/account-linking
                 screens. Exports render_login() — call it before anything
                 else renders.
api_client.py    authed_request() — every call to the backend from app.py
                  goes through this. Attaches the Bearer token, transparently
                  refreshes it on a 401, retries once.
app.py            The actual product — 5 tabs (Discover, My Profile, My
                   Memory, Analytics, Account), gated by render_login().
```

---

## The login gate

```python
# app.py, right after st.set_page_config() — must be this early
if not render_login():
    st.stop()
```

`render_login()` handles its own internal routing (login/register/Google
tabs, 2FA code entry, password reset, account linking) and returns
`False` (having already rendered *something*) until the user actually
has a valid session. Everything below the gate in `app.py` can assume a
real, logged-in user.

## `authed_request()` — the only way `app.py` should talk to the backend

Every single backend call in `app.py` — `_get`, `_post`, the SSE stream
in `api_recommend`, `api_review_summary`, `api_profile`, `api_memory`,
the whole Account tab — goes through `authed_request()`, never raw
`requests.get`/`requests.post`. This is not a style preference: it's
what attaches `Authorization: Bearer <token>` and silently retries once
with a refreshed access token on a 401. Calling `requests` directly
anywhere in `app.py` means that call silently has no auth on it.

If you add a new API call to `app.py`, it should look like:

```python
resp = authed_request("GET", "/some/path", params={...})
```

not:

```python
resp = requests.get(f"{BASE_URL}/some/path", params={...})  # no auth header — wrong
```

## Every network call in this codebase must be wrapped, and here's why it actually matters

This isn't a generic "always handle exceptions" note — it's specific to
how Streamlit executes a script. **All tabs render in one script pass**;
Streamlit just hides the ones you're not looking at, it doesn't skip
executing their code. A concrete bug that existed here: the "Continue
with Google" tab in `login_page.py` made an *unguarded*
`requests.get(f"{BASE_URL}/auth/google/login")` — unconditionally, on
every page load, not behind a button. If the backend was ever briefly
unreachable, that one call threw an uncaught `ConnectionError` that
crashed the **entire login screen**, including the Log In and Sign Up
tabs the person actually wanted to use. This was found by running the
app under Streamlit's own `AppTest` harness (`streamlit.testing.v1`),
not by reading the code — the bug was invisible to static review because
the code looked completely ordinary.

Every network call in `login_page.py` now goes through a local
`_safe_request()` helper that catches `requests.RequestException` and
returns `None`; every call site checks for `None` before touching
`.status_code`. `app.py`'s Account tab has the equivalent `_safe_request()`
wrapping `authed_request()` for the same reason — sessions/2FA/deactivate
all make calls unconditionally when that tab renders, not just on a
button click.

**When adding any new network call anywhere in this frontend:** if it's
unconditional (runs just because a tab or screen renders), it MUST be
wrapped. If it's inside an `if st.button(...):` block, wrapping it is
still correct (a click-triggered crash still kills the whole app, just
less often) but is more forgivable to miss under time pressure than an
unconditional one.

## Testing this frontend without a running backend

`streamlit.testing.v1.AppTest` runs the actual script — including every
top-level call — and reports exceptions without needing a browser or a
live backend:

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file('app.py', default_timeout=15)
at.run()
print(list(at.exception))   # should be [] even with no backend running

# Simulate a logged-in session to exercise the gated tabs:
at2 = AppTest.from_file('app.py', default_timeout=15)
at2.session_state['access_token'] = 'fake'
at2.session_state['refresh_token'] = 'fake'
at2.run()
print(list(at2.exception))  # should also be [] — every failure should
                              # degrade to st.error(), not an exception
```

Run this after touching either file — it catches real bugs (the Google
tab crash above, an `AttributeError` from a wrong dict key, a missing
import) that `python -m py_compile` cannot, since compiling only checks
syntax, not whether the code actually runs correctly.

## `user_id` / `user_name` in session state predate real auth

`/profile/{user_id}`, `/memory/{user_id}`, and `/feedback` are all keyed
on a plain string `user_id` that has nothing to do with the `User` table
or JWT `sub` claim — it's a separate, older personalization concept.
Since real auth now exists, `app.py` defaults `st.session_state.user_id`
to the authenticated account's email on first load (`current_email`,
fetched once from `/auth/me`) instead of leaving it for the person to
type in manually — but it's still just a string those endpoints accept,
still editable in the sidebar, and still not connected to `User.id` at
the database level. Don't assume `user_id` here is a UUID or maps to
any row in the `users` table.

## The Account tab (`app.py`, bottom section)

Sessions list, TOTP 2FA enrollment/disable, and account deactivation —
all reading/writing the endpoints documented in
`backend/services/auth/CLAUDE.md`. The 2FA QR code uses `qrcode[pil]`,
imported in a `try/except ImportError` — if it's not installed, setup
falls back to showing the manual-entry secret as text instead of
failing. Don't make this import a hard requirement at the top of the
file; the manual-entry fallback is the intended degraded path, not a
placeholder to be removed once the dependency is "properly" installed.