"""
frontend/api_client.py

Every backend call in the Streamlit app should go through here — not
`requests` directly. Two jobs:
  1. Attach the access token to every request.
  2. On a 401, transparently call /auth/refresh and retry once before
     forcing the user back to the login screen.

"""

import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:8000"  # swap for your deployed backend URL


def _headers() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _try_refresh() -> bool:
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False

    resp = requests.post(f"{BASE_URL}/auth/refresh", json={"refresh_token": refresh_token})
    if resp.status_code != 200:
        return False

    data = resp.json()
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]  # rotated — old one is now dead
    return True


def authed_request(method: str, path: str, **kwargs) -> requests.Response:
    """
    method: "GET" / "POST" / etc.
    path:   e.g. "/recommend" — BASE_URL is prepended.
    kwargs: passed straight through to requests (json=, params=, ...).
    """
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)

    if resp.status_code == 401 and _try_refresh():
        resp = requests.request(method, url, headers=_headers(), **kwargs)

    if resp.status_code == 401:
        # Refresh token is also dead — session is genuinely over.
        for key in ("access_token", "refresh_token"):
            st.session_state.pop(key, None)
        st.warning("Your session expired. Please log in again.")
        st.rerun()

    return resp


def is_logged_in() -> bool:
    return "access_token" in st.session_state


def logout():
    refresh_token = st.session_state.get("refresh_token")
    if refresh_token:
        try:
            requests.post(f"{BASE_URL}/auth/logout", json={"refresh_token": refresh_token})
        except requests.RequestException:
            pass  # best-effort — clear local state regardless
    for key in ("access_token", "refresh_token"):
        st.session_state.pop(key, None)