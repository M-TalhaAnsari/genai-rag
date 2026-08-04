"""
frontend/app.py
----------------
Streamlit frontend for the Connoisseur restaurant discovery system.

Features:
  - Search bar → calls GET /search or POST /recommend
  - Restaurant cards with rating, cuisine, city, contact info
  - "Select" button → generates draft message → shows approval modal
  - Edit + Send → triggers n8n via POST /contact-restaurant
  - Thumbs up / Thumbs down → POST /feedback
  - My Profile tab → GET /profile/{user_id}
  - Memory tab → GET /memory/{user_id}

Run:
    streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json


# ── Config ──────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Connoisseur",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0F0F;
    color: #F0EDE8;
}

/* Headings */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #F0EDE8;
}

/* Search input */
.stTextInput > div > div > input {
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 12px;
    color: #F0EDE8;
    font-size: 1rem;
    padding: 0.75rem 1rem;
}
.stTextInput > div > div > input:focus {
    border-color: #C8A96E;
    box-shadow: 0 0 0 2px rgba(200,169,110,0.15);
}

/* Restaurant card */
.restaurant-card {
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.restaurant-card:hover {
    border-color: #C8A96E;
}
.card-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #F0EDE8;
    margin: 0 0 0.25rem 0;
}
.card-meta {
    font-size: 0.85rem;
    color: #888;
    margin: 0 0 0.75rem 0;
}
.card-badge {
    display: inline-block;
    background: #252525;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    color: #C8A96E;
    margin-right: 6px;
}
.rating-star {
    color: #C8A96E;
}
.contact-tag {
    display: inline-block;
    font-size: 0.75rem;
    color: #666;
    margin-right: 8px;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.4rem 1rem;
    transition: all 0.15s;
}

/* Modal overlay */
.modal-box {
    background: #1A1A1A;
    border: 1px solid #C8A96E;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.modal-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #C8A96E;
    margin-bottom: 0.75rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2E2E2E;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 8px;
    color: #888;
    padding: 0.4rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #252525;
    border-color: #C8A96E;
    color: #F0EDE8;
}

/* Score pill */
.score-pill {
    font-size: 0.72rem;
    color: #555;
}

/* Success / error banners */
.success-banner {
    background: #0D2B1A;
    border: 1px solid #1A5C36;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #4CAF82;
    margin: 0.5rem 0;
}
.error-banner {
    background: #2B0D0D;
    border: 1px solid #5C1A1A;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #CF6679;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "user_id":       "",
        "user_name":     "",
        "search_results": [],
        "last_query":    "",
        "contact_modal": None,   # holds restaurant data when modal is open
        "draft_message": "",
        "contact_sent":  False,
        "feedback_sent": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── API helpers ─────────────────────────────────────────────────────────────

def api_search(query: str, user_id: str) -> list:
    try:
        params = {"q": query, "top_k": 10}
        if user_id:
            params["user_id"] = user_id
        r = requests.get(f"{API_BASE}/search", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []


def api_recommend(query: str, user_id: str) -> list:
    """Call /recommend and consume the SSE stream, return final recommendations."""
    try:
        payload = {"query": query, "user_id": user_id or None, "top_k": 5}
        with requests.post(
            f"{API_BASE}/recommend",
            json=payload,
            stream=True,
            timeout=60
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("event") == "result":
                        return data.get("recommendations", [])
        return []
    except Exception as e:
        st.error(f"Recommendation failed: {e}")
        return []


def api_get_restaurant(restaurant_id: int) -> dict:
    try:
        r = requests.get(f"{API_BASE}/restaurants/{restaurant_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_generate_message(restaurant: dict, user_name: str, query: str) -> str:
    try:
        payload = {
            "restaurant_id":   restaurant.get("restaurant_id", 0),
            "restaurant_name": restaurant.get("name", ""),
            "cuisine":         restaurant.get("cuisine", ""),
            "city":            restaurant.get("city", ""),
            "user_name":       user_name or "Guest",
            "user_query":      query,
        }
        r = requests.post(
            f"{API_BASE}/generate-message",
            json=payload,
            timeout=15
        )
        r.raise_for_status()
        return r.json().get("draft_message", "")
    except Exception as e:
        return f"Hi, I found your restaurant and would like to get in touch. Best regards, {user_name}"


def api_contact_restaurant(restaurant: dict, full_restaurant: dict,
                            message: str, user_name: str, query: str) -> dict:
    try:
        payload = {
            "restaurant_id":   restaurant.get("restaurant_id", 0),
            "restaurant_name": restaurant.get("name", ""),
            "cuisine":         restaurant.get("cuisine", ""),
            "city":            restaurant.get("city", ""),
            "email":           full_restaurant.get("email"),
            "phone":           full_restaurant.get("phone"),
            "website":         full_restaurant.get("website"),
            "message":         message,
            "user_name":       user_name or "Guest",
            "user_query":      query,
        }
        r = requests.post(
            f"{API_BASE}/contact-restaurant",
            json=payload,
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_feedback(restaurant: dict, signal: int, query: str, user_id: str):
    if not user_id:
        return
    try:
        payload = {
            "user_id":         user_id,
            "restaurant_id":   restaurant.get("restaurant_id", 0),
            "restaurant_name": restaurant.get("name", ""),
            "cuisine":         restaurant.get("cuisine", ""),
            "city":            restaurant.get("city", ""),
            "signal":          signal,
            "query":           query,
        }
        requests.post(f"{API_BASE}/feedback", json=payload, timeout=5)
    except Exception:
        pass


def api_get_profile(user_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/profile/{user_id}", timeout=5)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_get_memory(user_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/memory/{user_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_analytics() -> dict:
    try:
        r = requests.get(f"{API_BASE}/analytics", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍽️ Connoisseur")
    st.markdown("<p style='color:#888;font-size:0.85rem;'>Restaurant discovery for Pakistan</p>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**Your details**")
    name_input = st.text_input(
        "Your name",
        value=st.session_state.user_name,
        placeholder="e.g. Ahmed",
        label_visibility="collapsed"
    )
    uid_input = st.text_input(
        "User ID (for personalisation)",
        value=st.session_state.user_id,
        placeholder="e.g. ahmed_123",
        label_visibility="collapsed"
    )
    st.caption("Name · User ID for personalised results")

    if name_input != st.session_state.user_name:
        st.session_state.user_name = name_input
    if uid_input != st.session_state.user_id:
        st.session_state.user_id = uid_input

    st.divider()
    st.markdown("<p style='color:#555;font-size:0.78rem;'>Cities: Lahore · Islamabad<br>Karachi · Rawalpindi</p>",
                unsafe_allow_html=True)


# ── Main tabs ───────────────────────────────────────────────────────────────

tab_search, tab_profile, tab_memory, tab_analytics = st.tabs([
    "🔍 Discover", "👤 My Profile", "🧠 My Memory", "📊 Analytics"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════

with tab_search:

    st.markdown("## Find your next meal")
    st.markdown("<p style='color:#888;margin-top:-0.5rem;'>Search across Lahore, Islamabad, Karachi and Rawalpindi.</p>",
                unsafe_allow_html=True)

    # Search bar row
    col_input, col_mode, col_btn = st.columns([5, 2, 1])

    with col_input:
        query = st.text_input(
            "query",
            placeholder="biryani in Lahore · rooftop cafe Islamabad · family dinner Karachi",
            label_visibility="collapsed",
            key="query_input"
        )

    with col_mode:
        mode = st.selectbox(
            "mode",
            ["Quick search", "AI recommendations"],
            label_visibility="collapsed"
        )

    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Run search
    if search_clicked and query.strip():
        st.session_state.last_query   = query.strip()
        st.session_state.contact_sent = False

        with st.spinner("Searching..." if mode == "Quick search" else "Running AI recommendation pipeline..."):
            if mode == "Quick search":
                results = api_search(query.strip(), st.session_state.user_id)
            else:
                results = api_recommend(query.strip(), st.session_state.user_id)

        st.session_state.search_results = results

    # ── Contact modal (shown above results when active) ─────────────────

    if st.session_state.contact_modal:
        restaurant = st.session_state.contact_modal

        st.markdown(f"""
<div class="modal-box">
  <div class="modal-title">📨 Contact {restaurant['name']}</div>
</div>
""", unsafe_allow_html=True)

        edited_message = st.text_area(
            "Message to send",
            value=st.session_state.draft_message,
            height=160,
            key="message_editor",
            help="Edit the message before sending, or send as-is."
        )

        col_send, col_cancel = st.columns([1, 1])

        with col_send:
            if st.button("✅ Send", type="primary", use_container_width=True):
                full_restaurant = api_get_restaurant(
                    restaurant.get("restaurant_id", 0)
                )
                result = api_contact_restaurant(
                    restaurant=restaurant,
                    full_restaurant=full_restaurant,
                    message=edited_message,
                    user_name=st.session_state.user_name,
                    query=st.session_state.last_query
                )

                st.session_state.contact_modal = None
                st.session_state.draft_message  = ""

                if result.get("success"):
                    methods = ", ".join(result.get("contact_methods", ["unknown"]))
                    st.markdown(
                        f'<div class="success-banner">✓ Message sent via {methods}. '
                        f'The restaurant will be in touch.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="error-banner">✗ Could not send: {result.get("error", "unknown error")}</div>',
                        unsafe_allow_html=True
                    )

        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True):
                st.session_state.contact_modal = None
                st.session_state.draft_message  = ""
                st.rerun()

        st.divider()

    # ── Results ─────────────────────────────────────────────────────────

    results = st.session_state.search_results

    if results:
        st.markdown(
            f"<p style='color:#555;font-size:0.85rem;margin-bottom:1rem;'>"
            f"{len(results)} result{'s' if len(results)!=1 else ''} for "
            f"<em style='color:#C8A96E;'>{st.session_state.last_query}</em></p>",
            unsafe_allow_html=True
        )

        for r in results:
            rid      = r.get("restaurant_id", 0)
            name     = r.get("name", "Unknown")
            cuisine  = r.get("cuisine", "")
            city     = r.get("city", "")
            rating   = r.get("rating")
            price    = r.get("price_level", "")
            reasoning = r.get("reasoning", "")    # agent recommendation
            rrf_score = r.get("rrf_score", 0)

            # Rating stars
            stars = ""
            if rating:
                full  = int(rating)
                stars = "★" * full + "☆" * (5 - full)

            # Card HTML
            st.markdown(f"""
<div class="restaurant-card">
  <p class="card-name">{name}</p>
  <p class="card-meta">
    <span class="card-badge">{cuisine}</span>
    <span class="card-badge">{city}</span>
    {f'<span class="card-badge">{price}</span>' if price else ''}
    {f'<span class="rating-star">{stars}</span> <span style="color:#888;font-size:0.82rem;">{rating}/5</span>' if rating else ''}
  </p>
  {f'<p style="color:#aaa;font-size:0.88rem;margin:0 0 0.5rem 0;">{reasoning}</p>' if reasoning else ''}
  <span class="score-pill">relevance {rrf_score:.4f}</span>
</div>
""", unsafe_allow_html=True)

            # Action buttons row
            col_select, col_up, col_down, col_spacer = st.columns([2, 1, 1, 4])

            with col_select:
                if st.button(
                    f"Select →", key=f"select_{rid}",
                    help="Generate a contact message for this restaurant"
                ):
                    with st.spinner("Generating message..."):
                        draft = api_generate_message(
                            restaurant=r,
                            user_name=st.session_state.user_name,
                            query=st.session_state.last_query
                        )
                    st.session_state.contact_modal = r
                    st.session_state.draft_message  = draft
                    st.rerun()

            with col_up:
                if st.button(
                    "👍", key=f"up_{rid}",
                    help="I liked this result",
                    disabled=rid in st.session_state.feedback_sent
                ):
                    api_feedback(r, 1, st.session_state.last_query,
                                 st.session_state.user_id)
                    st.session_state.feedback_sent.add(rid)
                    st.toast(f"👍 Noted! We'll show more like {name}.")

            with col_down:
                if st.button(
                    "👎", key=f"down_{rid}",
                    help="Not relevant",
                    disabled=rid in st.session_state.feedback_sent
                ):
                    api_feedback(r, -1, st.session_state.last_query,
                                 st.session_state.user_id)
                    st.session_state.feedback_sent.add(rid)
                    st.toast(f"👎 Got it. We'll adjust your results.")

    elif st.session_state.last_query:
        st.markdown(
            "<div class='error-banner'>No results found. "
            "Try a different query or city name.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
<div style='color:#444;text-align:center;padding:3rem 0;'>
  <p style='font-size:2rem;'>🍽️</p>
  <p style='font-size:1rem;'>Search for a restaurant above to get started.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — MY PROFILE
# ══════════════════════════════════════════════════════════════════════════

with tab_profile:
    st.markdown("## Your taste profile")

    if not st.session_state.user_id:
        st.info("Enter your User ID in the sidebar to see your profile.")
    else:
        if st.button("Refresh profile", key="refresh_profile"):
            st.rerun()

        profile = api_get_profile(st.session_state.user_id)

        if not profile:
            st.markdown(
                "<div class='error-banner'>No profile yet. "
                "Search for restaurants and use 👍 👎 to build your profile.</div>",
                unsafe_allow_html=True
            )
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Cuisines you love**")
                preferred = profile.get("preferred_cuisines", [])
                if preferred:
                    for c in preferred:
                        st.markdown(f"✅ {c}")
                else:
                    st.caption("None yet — give some 👍")

                st.markdown("**Cuisines you avoid**")
                avoided = profile.get("avoided_cuisines", [])
                if avoided:
                    for c in avoided:
                        st.markdown(f"❌ {c}")
                else:
                    st.caption("None yet")

            with col_b:
                st.markdown("**Preferred cities**")
                cities = profile.get("preferred_cities", [])
                if cities:
                    for c in cities:
                        st.markdown(f"📍 {c}")
                else:
                    st.caption("None yet")

                st.markdown("**Feedback given**")
                st.metric("Total signals", profile.get("feedback_count", 0))


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — MY MEMORY
# ══════════════════════════════════════════════════════════════════════════

with tab_memory:
    st.markdown("## What we remember about you")

    if not st.session_state.user_id:
        st.info("Enter your User ID in the sidebar to see your memory.")
    else:
        memory = api_get_memory(st.session_state.user_id)

        if not memory:
            st.caption("No memory data yet. Start searching to build your history.")
        else:
            long_term = memory.get("long_term", {})
            session   = memory.get("session", {})

            st.markdown("**Summary**")
            summary = long_term.get("memory_summary", "")
            if summary and summary != "No memory summary yet.":
                st.markdown(
                    f"<div class='modal-box'><p style='color:#aaa;font-size:0.9rem;'>"
                    f"{summary}</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("Summary generated after 10 conversations.")

            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.markdown("**This session**")
                session_queries = session.get("recent_queries", [])
                if session_queries:
                    for q in session_queries:
                        st.markdown(f"🔍 {q}")
                else:
                    st.caption("No searches this session.")

            with col_m2:
                st.markdown("**Past searches**")
                past = long_term.get("recent_searches", [])
                if past:
                    for q in past[:8]:
                        st.markdown(f"🕐 {q}")
                else:
                    st.caption("No search history yet.")

            st.markdown("**Recent conversation**")
            history = long_term.get("recent_history", [])
            if history:
                for turn in history[-6:]:
                    role    = turn.get("role", "")
                    content = turn.get("content", "")
                    icon    = "🧑" if role == "user" else "🤖"
                    st.markdown(
                        f"<p style='font-size:0.85rem;color:#888;'>"
                        f"{icon} <strong style='color:#aaa;'>{role.title()}:</strong> {content[:200]}"
                        f"</p>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No conversation history yet.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.markdown("## System analytics")

    if st.button("Refresh", key="refresh_analytics"):
        st.rerun()

    data = api_analytics()

    if not data:
        st.caption("No analytics data available yet.")
    else:
        vol = data.get("search_volume", {})
        fb  = data.get("feedback", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total searches",   vol.get("total", 0))
        col2.metric("Searches (24h)",   vol.get("last_24h", 0))
        col3.metric("Total feedback",   fb.get("total", 0))
        col4.metric("Like rate",        f"{fb.get('like_rate_pct', 0)}%")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Top queries**")
            for item in data.get("top_queries", [])[:8]:
                q     = item.get("query", "")
                count = item.get("count", 0)
                st.markdown(
                    f"<p style='font-size:0.88rem;'>"
                    f"<span style='color:#C8A96E;'>{q}</span> "
                    f"<span style='color:#555;'>× {count}</span></p>",
                    unsafe_allow_html=True
                )

        with col_right:
            st.markdown("**Most liked cuisines**")
            for item in data.get("cuisine_preferences", {}).get("most_liked", [])[:6]:
                st.markdown(
                    f"<p style='font-size:0.88rem;'>✅ "
                    f"<span style='color:#aaa;'>{item['cuisine']}</span> "
                    f"<span style='color:#555;'>× {item['count']}</span></p>",
                    unsafe_allow_html=True
                )

            st.markdown("**Most avoided cuisines**")
            for item in data.get("cuisine_preferences", {}).get("most_avoided", [])[:6]:
                st.markdown(
                    f"<p style='font-size:0.88rem;'>❌ "
                    f"<span style='color:#aaa;'>{item['cuisine']}</span> "
                    f"<span style='color:#555;'>× {item['count']}</span></p>",
                    unsafe_allow_html=True
                )