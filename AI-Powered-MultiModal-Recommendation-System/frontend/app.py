"""
frontend/app.py
----------------
Streamlit frontend — four tabs:
  🔍 Discover   — search + restaurant cards with clickable contact links
  👤 My Profile — preference profile from feedback history
  🧠 My Memory  — session + long-term conversation memory
  📊 Analytics  — search volume, top queries, cuisine stats

Contact links per card (shown only when data exists):
  📧 Email    → opens default email client with pre-filled message
  💬 WhatsApp → opens WhatsApp with pre-filled message
  🌐 Website  → opens restaurant website in new tab
  📋 Menu     → opens menu link in new tab

Run:
    streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Connoisseur",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0F0F;
    color: #F0EDE8;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #F0EDE8; }

.stTextInput > div > div > input {
    background: #1A1A1A; border: 1px solid #2E2E2E;
    border-radius: 12px; color: #F0EDE8; font-size: 1rem; padding: 0.75rem 1rem;
}
.stTextInput > div > div > input:focus {
    border-color: #C8A96E; box-shadow: 0 0 0 2px rgba(200,169,110,0.15);
}

.restaurant-card {
    background: #1A1A1A; border: 1px solid #2E2E2E;
    border-radius: 16px; padding: 1.25rem 1.5rem;
    margin-bottom: 1rem; transition: border-color 0.2s;
}
.restaurant-card:hover { border-color: #C8A96E; }
.card-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem; color: #F0EDE8; margin: 0 0 0.2rem 0;
}
.card-meta { font-size: 0.84rem; color: #888; margin: 0 0 0.6rem 0; }
.card-badge {
    display: inline-block; background: #252525;
    border: 1px solid #333; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem; color: #C8A96E; margin-right: 5px;
}
.rating-star { color: #C8A96E; }
.reasoning-text { color: #aaa; font-size: 0.87rem; margin: 0.3rem 0 0.6rem 0; }
.score-pill { font-size: 0.72rem; color: #444; }

/* Contact link buttons */
.contact-links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.6rem; }
.contact-btn {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 12px; border-radius: 8px; font-size: 0.82rem;
    font-weight: 500; text-decoration: none !important;
    transition: all 0.15s; border: 1px solid;
}
.btn-email    { background:#0D1F2D; border-color:#1A4A6E; color:#5BA3D9 !important; }
.btn-whatsapp { background:#0D2B1A; border-color:#1A5C36; color:#4CAF82 !important; }
.btn-website  { background:#2B1F0D; border-color:#6E3F1A; color:#D9915B !important; }
.btn-menu     { background:#2B0D2B; border-color:#6E1A6E; color:#D95BD9 !important; }
.btn-email:hover    { background:#1A3D57; }
.btn-whatsapp:hover { background:#1A5C36; }
.btn-website:hover  { background:#6E3F1A; }
.btn-menu:hover     { background:#6E1A6E; }

/* Review panel */
.review-panel {
    background: #141414; border: 1px solid #2E2E2E;
    border-radius: 10px; padding: 0.9rem 1rem; margin-top: 0.5rem;
}
.dim-row { font-size: 0.86rem; margin: 0.15rem 0; }
.dim-label { color: #888; font-weight: 500; }
.sig-positive { color: #4CAF82; }
.sig-negative { color: #CF6679; }
.sig-mixed    { color: #E5A34A; }
.recency-recent { color: #4CAF82; font-size: 0.72rem; margin-left:6px; }
.recency-old    { color: #E5A34A; font-size: 0.72rem; margin-left:6px; }
.review-meta { color: #444; font-size: 0.76rem; margin-top: 0.4rem; }
.fake-warning { color:#CF6679; font-size:0.8rem; margin-top:0.3rem; }

section[data-testid="stSidebar"] { background:#141414; border-right:1px solid #2E2E2E; }
.stTabs [data-baseweb="tab"] { background:#1A1A1A; border:1px solid #2E2E2E; border-radius:8px; color:#888; }
.stTabs [aria-selected="true"] { background:#252525; border-color:#C8A96E; color:#F0EDE8; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "user_id":        "",
        "user_name":      "",
        "search_results": [],
        "last_query":     "",
        "feedback_sent":  set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── API helpers ──────────────────────────────────────────────────────────────

def _get(path, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def _post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_search(q, user_id):
    params = {"q": q, "top_k": 10}
    if user_id:
        params["user_id"] = user_id
    return _get("/search", **params).get("results", [])


def api_recommend(q, user_id):
    """Call /recommend SSE stream and return final recommendations."""
    try:
        payload = {"query": q, "user_id": user_id or None, "top_k": 5}
        with requests.post(f"{API_BASE}/recommend", json=payload,
                           stream=True, timeout=90) as r:
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
        st.error(f"Recommend error: {e}")
        return []


def api_contact_links(restaurant_id, user_name, user_query):
    return _get(
        f"/restaurants/{restaurant_id}/contact-links",
        user_name=user_name or "Guest",
        user_query=user_query or ""
    )


def api_review_summary(restaurant_id):
    try:
        r = requests.get(f"{API_BASE}/restaurants/{restaurant_id}/review-summary",
                         timeout=5)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_feedback(restaurant, signal, query, user_id):
    if not user_id:
        return
    _post("/feedback", {
        "user_id":         user_id,
        "restaurant_id":   restaurant.get("restaurant_id", 0),
        "restaurant_name": restaurant.get("name", ""),
        "cuisine":         restaurant.get("cuisine", ""),
        "city":            restaurant.get("city", ""),
        "signal":          signal,
        "query":           query,
    })


def api_profile(user_id):
    try:
        r = requests.get(f"{API_BASE}/profile/{user_id}", timeout=5)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_memory(user_id):
    try:
        r = requests.get(f"{API_BASE}/memory/{user_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_analytics():
    return _get("/analytics")


# ── Contact links HTML ────────────────────────────────────────────────────────

def render_contact_links_html(links: dict) -> str:
    """Build the contact buttons HTML from the API response."""
    if not links or not links.get("available_channels"):
        return "<p style='color:#444;font-size:0.8rem;'>No contact info available.</p>"

    buttons = []

    if links.get("email_href"):
        buttons.append(
            f'<a class="contact-btn btn-email" href="{links["email_href"]}" target="_blank">'
            f'📧 Email</a>'
        )
    if links.get("whatsapp_href"):
        buttons.append(
            f'<a class="contact-btn btn-whatsapp" href="{links["whatsapp_href"]}" target="_blank">'
            f'💬 WhatsApp</a>'
        )
    if links.get("website"):
        buttons.append(
            f'<a class="contact-btn btn-website" href="{links["website"]}" target="_blank">'
            f'🌐 Website</a>'
        )
    if links.get("menu_url"):
        buttons.append(
            f'<a class="contact-btn btn-menu" href="{links["menu_url"]}" target="_blank">'
            f'📋 Menu</a>'
        )

    return f'<div class="contact-links">{"".join(buttons)}</div>'


# ── Review summary HTML ───────────────────────────────────────────────────────

def render_review_summary_html(data: dict) -> str:
    if not data:
        return "<p style='color:#444;font-size:0.8rem;'>No review summary yet.</p>"

    dim_labels = {
        "food_quality": ("🍽️", "Food Quality"),
        "cleanliness":  ("🧹", "Cleanliness"),
        "service":      ("👨‍💼", "Service"),
        "menu_variety": ("📋", "Menu Variety"),
        "vibe":         ("✨", "Atmosphere"),
    }
    sig_class = {
        "positive": "sig-positive",
        "negative": "sig-negative",
        "mixed":    "sig-mixed",
    }

    rows = []
    dimensions = data.get("dimensions", {})
    for key, (icon, label) in dim_labels.items():
        dim = dimensions.get(key, {})
        signal  = dim.get("signal", "unknown")
        summary = dim.get("summary", "")
        recent  = dim.get("from_recent_reviews")
        if signal == "unknown" or not summary:
            continue
        css     = sig_class.get(signal, "")
        recency = ""
        if recent is True:
            recency = '<span class="recency-recent">recent</span>'
        elif recent is False:
            recency = '<span class="recency-old">older reviews</span>'
        rows.append(
            f'<div class="dim-row">{icon} '
            f'<span class="dim-label {css}">{label}</span>{recency} — '
            f'<span style="color:#aaa;">{summary}</span></div>'
        )

    if not rows:
        rows = ["<p style='color:#555;font-size:0.82rem;'>No text content in reviews.</p>"]

    # Metadata
    meta_parts = []
    avg     = data.get("avg_rating")
    weighted = data.get("weighted_rating")
    count   = data.get("review_count", 0)
    recent_date = (data.get("most_recent_review") or "")[:10]
    oldest_date = (data.get("oldest_review") or "")[:10]

    if avg:
        meta_parts.append(f"Overall: {avg}★")
    if weighted and avg and abs(weighted - avg) >= 0.3:
        trend = "📈" if weighted > avg else "📉"
        meta_parts.append(f"Recent: {weighted}★ {trend}")
    if count:
        meta_parts.append(f"{count} reviews")
    if recent_date:
        meta_parts.append(f"Latest: {recent_date}")
    if oldest_date:
        meta_parts.append(f"Oldest: {oldest_date}")

    meta_html = (
        f'<div class="review-meta">{" · ".join(meta_parts)}</div>'
        if meta_parts else ""
    )

    fake_html = ""
    if data.get("has_fake_signals"):
        fake_html = f'<div class="fake-warning">⚠️ {data.get("disclaimer","")}</div>'
    elif data.get("confidence") == "low":
        fake_html = '<div class="fake-warning">⚠️ Low confidence — very few reviews.</div>'

    return (
        f'<div class="review-panel">'
        f'{"".join(rows)}{meta_html}{fake_html}'
        f'</div>'
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍽️ Connoisseur")
    st.markdown("<p style='color:#888;font-size:0.84rem;'>Restaurant discovery · Pakistan</p>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**Your details**")
    name_val = st.text_input("Name", value=st.session_state.user_name,
                              placeholder="Ahmed", label_visibility="collapsed")
    uid_val  = st.text_input("User ID", value=st.session_state.user_id,
                              placeholder="ahmed_123", label_visibility="collapsed")
    st.caption("Name · User ID for personalised results")

    if name_val != st.session_state.user_name:
        st.session_state.user_name = name_val
    if uid_val != st.session_state.user_id:
        st.session_state.user_id = uid_val

    st.divider()
    st.markdown("<p style='color:#555;font-size:0.78rem;'>"
                "Lahore · Islamabad · Karachi · Rawalpindi</p>",
                unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_search, tab_profile, tab_memory, tab_analytics = st.tabs([
    "🔍 Discover", "👤 My Profile", "🧠 My Memory", "📊 Analytics"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DISCOVER
# ════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.markdown("## Find your next meal")
    st.markdown("<p style='color:#888;margin-top:-0.5rem;'>"
                "Search across Lahore, Islamabad, Karachi and Rawalpindi.</p>",
                unsafe_allow_html=True)

    col_q, col_mode, col_btn = st.columns([5, 2, 1])
    with col_q:
        query = st.text_input("q", placeholder="biryani Lahore · rooftop cafe Islamabad",
                              label_visibility="collapsed", key="query_input")
    with col_mode:
        mode = st.selectbox("mode", ["Quick search", "AI recommendations"],
                            label_visibility="collapsed")
    with col_btn:
        go = st.button("Search", type="primary", use_container_width=True)

    if go and query.strip():
        st.session_state.last_query = query.strip()
        with st.spinner("Searching..." if mode == "Quick search"
                        else "Running 6-agent recommendation pipeline..."):
            if mode == "Quick search":
                st.session_state.search_results = api_search(
                    query.strip(), st.session_state.user_id
                )
            else:
                st.session_state.search_results = api_recommend(
                    query.strip(), st.session_state.user_id
                )

    results = st.session_state.search_results

    if results:
        st.markdown(
            f"<p style='color:#555;font-size:0.84rem;margin-bottom:0.5rem;'>"
            f"{len(results)} result{'s' if len(results)!=1 else ''} for "
            f"<em style='color:#C8A96E;'>{st.session_state.last_query}</em></p>",
            unsafe_allow_html=True
        )

        for r in results:
            rid      = r.get("restaurant_id", 0)
            name     = r.get("name", "Unknown")
            cuisine  = r.get("cuisine", "")
            city     = r.get("city", "")
            area     = r.get("area", "")
            rating   = r.get("rating")
            price    = r.get("price_level", "")
            reasoning = r.get("reasoning", "")
            rrf_score = r.get("rrf_score", 0)

            stars = ""
            if rating:
                stars = "★" * int(rating) + "☆" * (5 - int(rating))

            loc = f"{city}" + (f" · {area}" if area else "")

            # Card
            st.markdown(f"""
<div class="restaurant-card">
  <p class="card-name">{name}</p>
  <p class="card-meta">
    <span class="card-badge">{cuisine}</span>
    <span class="card-badge">{loc}</span>
    {f'<span class="card-badge">{price}</span>' if price else ''}
    {f'<span class="rating-star">{stars}</span> <span style="color:#888;font-size:0.8rem;">{rating}/5</span>' if rating else ''}
  </p>
  {f'<p class="reasoning-text">{reasoning}</p>' if reasoning else ''}
  <span class="score-pill">relevance {rrf_score:.4f}</span>
</div>""", unsafe_allow_html=True)

            # Contact links + review + feedback in columns
            col_contact, col_review, col_fb = st.columns([3, 5, 2])

            with col_contact:
                st.markdown("**Contact**")
                links = api_contact_links(
                    rid,
                    st.session_state.user_name,
                    st.session_state.last_query
                )
                st.markdown(render_contact_links_html(links), unsafe_allow_html=True)

            with col_review:
                with st.expander("📋 Reviews", expanded=False):
                    review_data = api_review_summary(rid)
                    st.markdown(render_review_summary_html(review_data),
                                unsafe_allow_html=True)

            with col_fb:
                st.markdown("**Feedback**")
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button("👍", key=f"up_{rid}",
                                 disabled=rid in st.session_state.feedback_sent):
                        api_feedback(r, 1, st.session_state.last_query,
                                     st.session_state.user_id)
                        st.session_state.feedback_sent.add(rid)
                        st.toast(f"👍 Noted!")
                with fb_col2:
                    if st.button("👎", key=f"dn_{rid}",
                                 disabled=rid in st.session_state.feedback_sent):
                        api_feedback(r, -1, st.session_state.last_query,
                                     st.session_state.user_id)
                        st.session_state.feedback_sent.add(rid)
                        st.toast("👎 Got it.")

            st.markdown("<hr style='border-color:#1A1A1A;margin:0.5rem 0;'>",
                        unsafe_allow_html=True)

    elif st.session_state.last_query:
        st.info("No results found. Try a different query or city.")
    else:
        st.markdown("""
<div style='color:#333;text-align:center;padding:3rem 0;'>
  <p style='font-size:2.5rem;'>🍽️</p>
  <p style='font-size:1rem;color:#555;'>Search for restaurants above.</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROFILE
# ════════════════════════════════════════════════════════════════════════════

with tab_profile:
    st.markdown("## Your taste profile")

    if not st.session_state.user_id:
        st.info("Enter your User ID in the sidebar to see your profile.")
    else:
        if st.button("Refresh", key="refresh_profile"):
            st.rerun()

        profile = api_profile(st.session_state.user_id)

        if not profile:
            st.markdown(
                "<p style='color:#555;'>No profile yet. "
                "Search and use 👍 👎 to build your preference profile.</p>",
                unsafe_allow_html=True
            )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Cuisines you love**")
                for c in profile.get("preferred_cuisines", []):
                    st.markdown(f"✅ {c}")
                if not profile.get("preferred_cuisines"):
                    st.caption("None yet — give some 👍")

                st.markdown("**Cuisines you avoid**")
                for c in profile.get("avoided_cuisines", []):
                    st.markdown(f"❌ {c}")
                if not profile.get("avoided_cuisines"):
                    st.caption("None yet")

            with col_b:
                st.markdown("**Preferred cities**")
                for c in profile.get("preferred_cities", []):
                    st.markdown(f"📍 {c}")

                st.markdown("**Signals recorded**")
                st.metric("Total feedback", profile.get("feedback_count", 0))


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MEMORY
# ════════════════════════════════════════════════════════════════════════════

with tab_memory:
    st.markdown("## What we remember about you")

    if not st.session_state.user_id:
        st.info("Enter your User ID in the sidebar.")
    else:
        memory = api_memory(st.session_state.user_id)

        if not memory:
            st.caption("No memory yet. Start searching.")
        else:
            long_term = memory.get("long_term", {})
            session   = memory.get("session", {})

            summary = long_term.get("memory_summary", "")
            if summary and summary != "No memory summary yet.":
                st.markdown("**Summary**")
                st.markdown(
                    f"<div style='background:#1A1A1A;border:1px solid #2E2E2E;"
                    f"border-radius:10px;padding:0.9rem 1rem;color:#aaa;font-size:0.9rem;'>"
                    f"{summary}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("Memory summary generated after 10 conversations.")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("**This session**")
                for q in session.get("recent_queries", []):
                    st.markdown(f"🔍 {q}")
                if not session.get("recent_queries"):
                    st.caption("No searches this session.")

            with col_m2:
                st.markdown("**Past searches**")
                for q in long_term.get("recent_searches", [])[:8]:
                    st.markdown(f"🕐 {q}")
                if not long_term.get("recent_searches"):
                    st.caption("No past search history.")

            st.markdown("**Recent conversation**")
            for turn in long_term.get("recent_history", [])[-6:]:
                role    = turn.get("role", "")
                content = turn.get("content", "")
                icon    = "🧑" if role == "user" else "🤖"
                st.markdown(
                    f"<p style='font-size:0.84rem;color:#666;'>"
                    f"{icon} <strong style='color:#888;'>{role.title()}:</strong> "
                    f"{content[:200]}</p>",
                    unsafe_allow_html=True
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.markdown("## System analytics")

    if st.button("Refresh", key="refresh_analytics"):
        st.rerun()

    data = api_analytics()

    if not data:
        st.caption("No analytics data yet.")
    else:
        vol = data.get("search_volume", {})
        fb  = data.get("feedback", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total searches",  vol.get("total", 0))
        c2.metric("Searches (24h)",  vol.get("last_24h", 0))
        c3.metric("Total feedback",  fb.get("total", 0))
        c4.metric("Like rate",       f"{fb.get('like_rate_pct', 0)}%")

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Top queries**")
            for item in data.get("top_queries", [])[:8]:
                st.markdown(
                    f"<p style='font-size:0.87rem;'>"
                    f"<span style='color:#C8A96E;'>{item['query']}</span> "
                    f"<span style='color:#555;'>× {item['count']}</span></p>",
                    unsafe_allow_html=True
                )

        with col_r:
            st.markdown("**Most liked cuisines**")
            for item in data.get("cuisine_preferences", {}).get("most_liked", [])[:6]:
                st.markdown(
                    f"<p style='font-size:0.87rem;'>✅ "
                    f"<span style='color:#aaa;'>{item['cuisine']}</span> "
                    f"<span style='color:#555;'>× {item['count']}</span></p>",
                    unsafe_allow_html=True
                )
            st.markdown("**Most avoided cuisines**")
            for item in data.get("cuisine_preferences", {}).get("most_avoided", [])[:6]:
                st.markdown(
                    f"<p style='font-size:0.87rem;'>❌ "
                    f"<span style='color:#aaa;'>{item['cuisine']}</span> "
                    f"<span style='color:#555;'>× {item['count']}</span></p>",
                    unsafe_allow_html=True
                )