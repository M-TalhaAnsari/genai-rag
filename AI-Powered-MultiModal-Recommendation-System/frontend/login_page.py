"""
frontend/login_page.py

Call render_login() at the very top of your Streamlit app, before
anything else. It returns True if the user is logged in and the rest
of the app should render; False if it just rendered the login screen
and the caller should stop.

Usage in app.py:

    from login_page import render_login

    if not render_login():
        st.stop()

    # ... rest of the app, using api_client.authed_request() for calls
"""

import requests
import streamlit as st
from api_client import BASE_URL, is_logged_in


def _safe_request(method: str, url: str, **kwargs):
    kwargs.setdefault("timeout", 10)
    try:
        return requests.request(method, url, **kwargs)
    except requests.RequestException:
        return None


_NETWORK_ERROR = "Can't reach the server right now. Please try again."


def _handle_google_redirect():

    params = st.query_params
    login_code = params.get("login_code")
    link_required = params.get("link_required")
    link_email = params.get("email")
    auth_error = params.get("auth_error")
    reset_token = params.get("reset_token")

    if auth_error:
        st.error(f"Google sign-in failed: {auth_error}")
        st.query_params.clear()
        return

    if login_code and not is_logged_in():

        resp = _safe_request("POST", f"{BASE_URL}/auth/google/exchange", json={"login_code": login_code})
        st.query_params.clear()
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
            st.rerun()
        else:
            st.error("Google sign-in failed — the login link may have expired. Try again.")
        return

    if link_required and not is_logged_in():
     
        st.session_state["_link_token"] = link_required
        st.session_state["_link_email"] = link_email
        return

    if reset_token and not is_logged_in():
    
        st.session_state["_reset_token"] = reset_token
        st.query_params.clear()


def _render_reset_password():
    st.title("Choose a new password")
    new_password = st.text_input("New password (min 8 characters)", type="password", key="reset_new_password")
    confirm_password = st.text_input("Confirm new password", type="password", key="reset_confirm_password")

    if st.button("Reset password", key="reset_submit_btn"):
        if new_password != confirm_password:
            st.error("Passwords don't match.")
        elif len(new_password) < 8:
            st.error("Password must be at least 8 characters.")
        else:
            resp = _safe_request(
                "POST", f"{BASE_URL}/auth/reset-password",
                json={"token": st.session_state["_reset_token"], "new_password": new_password},
            )
            if resp is not None and resp.status_code == 204:
                st.session_state.pop("_reset_token", None)
                st.success("Password updated. Every existing session was signed out — log in with your new password.")
                st.rerun()
            elif resp is not None:
                st.error(resp.json().get("detail", "That reset link is invalid or has expired."))
            else:
                st.error(_NETWORK_ERROR)

    if st.button("Cancel", key="reset_cancel_btn"):
        st.session_state.pop("_reset_token", None)
        st.rerun()


def _render_mfa_verify():
    st.title("Enter your 2FA code")
    st.write("Open your authenticator app and enter the current 6-digit code.")
    code = st.text_input("Code", key="mfa_code", max_chars=6)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Verify", key="mfa_verify_btn"):
            resp = _safe_request(
                "POST", f"{BASE_URL}/auth/login/verify-2fa",
                json={"mfa_token": st.session_state["_mfa_token"], "code": code},
            )
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                st.session_state["access_token"] = data["access_token"]
                st.session_state["refresh_token"] = data["refresh_token"]
                st.session_state.pop("_mfa_token", None)
                st.rerun()
            elif resp is not None:
                st.error(resp.json().get("detail", "Invalid code."))
            else:
                st.error(_NETWORK_ERROR)
    with col2:
        if st.button("Cancel", key="mfa_cancel_btn"):
            st.session_state.pop("_mfa_token", None)
            st.rerun()


def _render_link_confirm():
    """Shown instead of the normal tabs when a link_required redirect
    landed. Collects the local account's password and completes the
    merge via /auth/google/link-confirm."""
    email = st.session_state.get("_link_email")
    st.title("Link your Google account")
    st.write(
        f"An account already exists for **{email}**. "
        "Enter that account's password to link Google sign-in to it."
    )
    password = st.text_input("Password", type="password", key="link_password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Link accounts", key="link_confirm_btn"):
            resp = _safe_request(
                "POST", f"{BASE_URL}/auth/google/link-confirm",
                json={"link_token": st.session_state["_link_token"], "password": password},
            )
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                st.session_state["access_token"] = data["access_token"]
                st.session_state["refresh_token"] = data["refresh_token"]
                st.session_state.pop("_link_token", None)
                st.session_state.pop("_link_email", None)
                st.query_params.clear()
                st.rerun()
            elif resp is not None:
                st.error(resp.json().get("detail", "Linking failed."))
            else:
                st.error(_NETWORK_ERROR)
    with col2:
        if st.button("Cancel", key="link_cancel_btn"):
            st.session_state.pop("_link_token", None)
            st.session_state.pop("_link_email", None)
            st.query_params.clear()
            st.rerun()


def render_login() -> bool:
    _handle_google_redirect()

    if is_logged_in():
        return True

    if st.session_state.get("_mfa_token"):
        _render_mfa_verify()
        return False

    if st.session_state.get("_link_token"):
        _render_link_confirm()
        return False

    if st.session_state.get("_reset_token"):
        _render_reset_password()
        return False

    st.title("Find your next meal")
    st.caption("Search across Lahore, Islamabad, Karachi and Rawalpindi.")

    tab_login, tab_register, tab_google = st.tabs(["Log in", "Sign up", "Continue with Google"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in", key="login_btn"):
            resp = _safe_request("POST", f"{BASE_URL}/auth/login", json={"email": email, "password": password})
            if resp is None:
                st.error(_NETWORK_ERROR)
            elif resp.status_code == 200:
                data = resp.json()
                if data.get("mfa_required"):
                    st.session_state["_mfa_token"] = data["mfa_token"]
                    st.rerun()
                else:
                    st.session_state["access_token"] = data["access_token"]
                    st.session_state["refresh_token"] = data["refresh_token"]
                    st.rerun()
            elif resp.status_code == 429:
                st.error(resp.json().get("detail", "Too many attempts. Please wait and try again."))
            else:
                st.error(resp.json().get("detail", "Login failed."))

        with st.expander("Forgot password?"):
            forgot_email = st.text_input("Email", key="forgot_email")
            if st.button("Send reset link", key="forgot_btn"):
                resp = _safe_request("POST", f"{BASE_URL}/auth/forgot-password", json={"email": forgot_email})
                if resp is not None:
                    st.info("If that email is registered, a password reset link has been sent.")
                else:
                    st.error(_NETWORK_ERROR)

    with tab_register:
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password (min 8 characters)", type="password", key="register_password")
        if st.button("Create account", key="register_btn"):
            resp = _safe_request("POST", f"{BASE_URL}/auth/register", json={"email": email, "password": password})
            if resp is None:
                st.error(_NETWORK_ERROR)
            elif resp.status_code == 201:

                st.success("Account created. Check your email for a verification link, then log in.")
            else:
                st.error(resp.json().get("detail", "Registration failed."))

    with tab_google:
        st.write("Sign in with your Google account — no password needed.")
        auth_url_resp = _safe_request("GET", f"{BASE_URL}/auth/google/login")
        if auth_url_resp is not None and auth_url_resp.status_code == 200:
            auth_url = auth_url_resp.json()["auth_url"]
            st.link_button("Continue with Google", auth_url)
        else:
            st.error("Google sign-in is currently unavailable.")

    return False