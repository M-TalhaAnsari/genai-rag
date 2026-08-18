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


def _handle_google_redirect():
    """Runs on every page load. Checks whether we just landed back from
    Google's consent screen with a login_code (or an error) in the URL."""
    params = st.query_params
    login_code = params.get("login_code")
    auth_error = params.get("auth_error")

    if auth_error:
        st.error(f"Google sign-in failed: {auth_error}")
        st.query_params.clear()
        return

    if login_code and not is_logged_in():
        resp = requests.post(f"{BASE_URL}/auth/google/exchange", params={"login_code": login_code})
        st.query_params.clear()
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
            st.rerun()
        else:
            st.error("Google sign-in failed — the login link may have expired. Try again.")


def render_login() -> bool:
    _handle_google_redirect()

    if is_logged_in():
        return True

    st.title("Find your next meal")
    st.caption("Search across Lahore, Islamabad, Karachi and Rawalpindi.")

    tab_login, tab_register, tab_google = st.tabs(["Log in", "Sign up", "Continue with Google"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in", key="login_btn"):
            resp = requests.post(f"{BASE_URL}/auth/login", json={"email": email, "password": password})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["access_token"] = data["access_token"]
                st.session_state["refresh_token"] = data["refresh_token"]
                st.rerun()
            else:
                st.error(resp.json().get("detail", "Login failed."))

    with tab_register:
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password (min 8 characters)", type="password", key="register_password")
        if st.button("Create account", key="register_btn"):
            resp = requests.post(f"{BASE_URL}/auth/register", json={"email": email, "password": password})
            if resp.status_code == 201:
                data = resp.json()
                st.session_state["access_token"] = data["access_token"]
                st.session_state["refresh_token"] = data["refresh_token"]
                st.rerun()
            else:
                st.error(resp.json().get("detail", "Registration failed."))

    with tab_google:
        st.write("Sign in with your Google account — no password needed.")
        auth_url_resp = requests.get(f"{BASE_URL}/auth/google/login")
        if auth_url_resp.status_code == 200:
            auth_url = auth_url_resp.json()["auth_url"]
            st.link_button("Continue with Google", auth_url)
        else:
            st.error("Google sign-in is currently unavailable.")

    return False