import streamlit as st
import requests

API_KEY = ""

st.title("Explore Fashion 🔍")

def run_search():
    st.session_state.search_triggered = True
    st.session_state.page = 1  # reset page
    st.session_state.images_to_show = 12

# Input field that triggers on Enter
query = st.text_input(
    "Search fashion images...",
    placeholder="e.g. shoes, jackets, streetwear",
    key="query",
    on_change=run_search
)

# Initialize session state
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "images_to_show" not in st.session_state:
    st.session_state.images_to_show = 12
if "page" not in st.session_state:
    st.session_state.page = 1

# If search triggered
if st.session_state.search_triggered and st.session_state.query:
    # Always search for outfits, filter to fashion category
    search_query = f"{st.session_state.query} outfit"
    url = f"https://pixabay.com/api/?key={API_KEY}&q={requests.utils.quote(search_query)}&image_type=photo&category=fashion&per_page=50"
    response = requests.get(url)

    # Check status BEFORE trying to parse JSON
    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
    else:
        try:
            data = response.json()
        except Exception as e:
            st.error("Failed to parse JSON response")
            st.write("Raw response:")
            st.code(response.text)
            st.stop()

    hits = data.get("hits", [])
    total = len(hits)

    if total > 0:
        st.write(f"Results for **{st.session_state.query}**:")
        cols = st.columns(3)

        for idx, hit in enumerate(hits[: st.session_state.images_to_show]):
            with cols[idx % 3]:
                st.image(hit["webformatURL"], width='stretch')

        # Load more button
        if st.session_state.images_to_show < total:
            if st.button("Load More"):
                st.session_state.images_to_show += 12
                st.rerun()
        else:
            st.info("No more images to load.")
    else:
        st.warning("No results found. Try another keyword 👀")
