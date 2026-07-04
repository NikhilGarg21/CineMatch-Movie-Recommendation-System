import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# =====================================================================
# CSS SYSTEM: LOCKS BOTH HOMEPAGE AND DETAILS REC GRIDS IN PERFECTION
# =====================================================================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.movie-title { font-size: 0.92rem; line-height: 1.2rem; height: 2.4rem; overflow: hidden; margin-top: 8px; font-weight: 500; text-align: left; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 18px; background: rgba(255,255,255,0.8); margin-bottom: 1rem; }

/* FIXES ALL POSTER EDGES/SIZES DYNAMICALLY INTO UNIFORM GRIDS WITHOUT SHIFTING */
.stImage > img {
    width: 100% !important;
    height: 380px !important;
    object-fit: cover !important;
    border-radius: 12px !important;
}
.stButton > button {
    width: 100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

def goto_home():
    st.session_state.view = "home"
    st.rerun()

def goto_details(movie_title: str):
    st.session_state.view = "details"
    st.session_state.selected_id = movie_title
    st.rerun()

def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"

def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            title = m.get("title", "Untitled")
            poster = m.get("poster_url") or "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=500"

            with colset[c]:
                st.image(poster, use_container_width=True)
                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}"):
                    goto_details(title)
                st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)

def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        meta = x.get("meta") or {}
        title = meta.get("title") or x.get("title") or "Untitled"
        poster_url = meta.get("poster_url") or x.get("poster_url")
        cards.append({"title": title, "poster_url": poster_url})
    return cards

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    if st.button("🏠 Home Layout", use_container_width=True):
        goto_home()
    grid_cols = st.slider("Columns per row", 4, 8, 6)

st.title("🎬 CineMatch")
st.divider()

# HOME VIEW
if st.session_state.view == "home":
    typed = st.text_input("Search by movie title...", placeholder="Type keyword (e.g. avenger, batman, iron man)...")

    if typed.strip():
        data, err = api_get_json("/search", params={"query": typed.strip(), "limit": 12})
        if err or data is None:
            st.error(f"Search error: {err}")
        else:
            suggestions = [m["title"] for m in data if "title" in m]
            if suggestions:
                selected = st.selectbox("Matching Suggestions", ["-- Select from list --"] + suggestions, index=0)
                if selected != "-- Select from list --":
                    goto_details(selected)
            
            st.markdown("### Search Results")
            poster_grid(data, cols=grid_cols, key_prefix="search_results")
        st.stop()

    st.markdown("### Trending Movies")
    home_cards, err = api_get_json("/home", params={"limit": 12})
    if err or not home_cards:
        st.error(f"Failed to fetch baseline trends: {err}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# DETAILS VIEW
elif st.session_state.view == "details":
    movie_title = st.session_state.selected_id
    
    a, b = st.columns([3, 1])
    with a:
        st.markdown("### Movie Details")
    with b:
        if st.button("← Back to Trends", use_container_width=True):
            goto_home()

    bundle, err = api_get_json("/movie/bundle", params={"query": movie_title, "tfidf_top_n": 12})
    if err or not bundle:
        st.error("Error matching index profile for this title.")
        st.stop()

    details = bundle.get("movie_details", {})
    left, right = st.columns([1, 2.2], gap="large")

    with left:
        poster_img = details.get("poster_url") or "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=500"
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{poster_img}" style="max-height: 480px; width: auto; object-fit: contain; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            </div>
            """, 
            unsafe_allow_html=True
        )

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {details.get('title','')}")
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown(f"**📅 Year:** {details.get('year','-')}")
            st.markdown(f"**⭐ Rating:** {details.get('rating','N/A')} / 10")
        with col_meta2:
            st.markdown(f"**👥 Cast:** {', '.join(details.get('actors', [])) or 'N/A'}")
            st.markdown(f"**📊 Total Votes:** {details.get('votes','N/A')}")
            
        st.markdown("---")
        st.markdown("### Overview")
        st.write(details.get("plot") or "No text summary compiled for this movie profile.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Similar Movies")
    local_recs = to_cards_from_tfidf_items(bundle.get("tfidf_recommendations"))
    poster_grid(local_recs, cols=grid_cols, key_prefix="details_tfidf")