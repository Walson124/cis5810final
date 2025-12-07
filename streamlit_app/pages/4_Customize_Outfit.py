import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image

from recommend import recommend_outfit, Outfit

BASE_DIR = Path(__file__).resolve().parents[2]
WARDROBE_CSV = BASE_DIR / "data" / "wardrobe.csv"

st.title("Customize Outfit — Pick required items")

# Load wardrobe CSV
if not WARDROBE_CSV.exists():
    st.warning("No wardrobe data found. Upload and segment clothes first.")
    st.stop()

wardrobe_df = pd.read_csv(WARDROBE_CSV)

# Initialize session state for selected items and filter
if "selected_ids_customize" not in st.session_state:
    st.session_state["selected_ids_customize"] = set()
if "selected_types_customize" not in st.session_state:
    st.session_state["selected_types_customize"] = list(wardrobe_df["type"].unique())

# Filter by type if desired
types = list(wardrobe_df["type"].unique())
st.session_state["selected_types_customize"] = st.multiselect(
    "Filter types to show (optional)", 
    types, 
    default=st.session_state["selected_types_customize"],
    key="type_filter"
)
filtered = wardrobe_df[wardrobe_df["type"].isin(st.session_state["selected_types_customize"])].reset_index(drop=True)

st.write(f"Found {len(wardrobe_df)} items in wardrobe.")
st.write("### Pick required items")

# Display thumbnails in a grid with checkbox under each
cols_per_row = 4
cols = st.columns(cols_per_row)

for i, (_, row) in enumerate(filtered.iterrows()):
    col = cols[i % cols_per_row]
    with col:
        img_path = Path(row["rgba_path"])
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path
        try:
            img = Image.open(img_path)
            if img.mode == "RGBA":
                background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            st.image(img, caption=f"{row['folder_name']} ({row['type']})", width='stretch')
        except Exception as e:
            st.write(f"Error loading {row['folder_name']}")

        cb_key = f"select_{row['id']}"
        st.checkbox(
            "Select", 
            key=cb_key, 
            value=(row['id'] in st.session_state["selected_ids_customize"]),
            on_change=lambda rid=row['id'], ckey=cb_key: (
                st.session_state["selected_ids_customize"].add(rid) 
                if st.session_state[ckey] 
                else st.session_state["selected_ids_customize"].discard(rid)
            )
        )

st.write(f"Selected {len(st.session_state['selected_ids_customize'])} required items.")

k = st.slider("How many outfit choices to sample (k)", min_value=1, max_value=50, value=10)

if st.button("Make recommendation"):
    if not st.session_state["selected_ids_customize"]:
        st.warning("Please select at least one item to require in the outfit.")
    else:
        st.info("Computing recommendation — sampling outfits...")
        outfit, score = recommend_outfit(k, wardrobe_df=wardrobe_df, amt='one', selected_ids=list(st.session_state["selected_ids_customize"]))
        if outfit is None:
            st.error("Could not find a recommendation including the selected pieces.")
        else:
            st.success(f"Recommended outfit (score: {score:.3f})")
            # Display selected items first, highlighted
            st.write("### Required items you selected")
            req = wardrobe_df[wardrobe_df['id'].isin(st.session_state["selected_ids_customize"])]
            req_cols = st.columns(min(4, len(req)))
            for i, (_, r) in enumerate(req.iterrows()):
                with req_cols[i % 4]:
                    img_path = Path(r['rgba_path'])
                    if not img_path.is_absolute():
                        img_path = BASE_DIR / img_path
                    try:
                        img = Image.open(img_path)
                        if img.mode == 'RGBA':
                            background = Image.new('RGBA', img.size, (255,255,255,255))
                            img = Image.alpha_composite(background, img)
                        st.image(img, caption=f"{r['folder_name']} (REQUIRED)", width='stretch')
                    except Exception as e:
                        st.write(f"Error loading {r['folder_name']}")

            # Display recommended outfit
            st.write("### Recommended Outfit")
            pieces = outfit.wardrobe_df.loc[outfit.wardrobe_df['id'].isin(outfit.ids)]
            cols_count = min(3, len(pieces))
            cols = st.columns(cols_count)
            for i, (_, pr) in enumerate(pieces.iterrows()):
                img_path = Path(pr["rgba_path"]) if isinstance(pr["rgba_path"], str) else Path(pr["rgba_path"])
                if not img_path.is_absolute():
                    img_path = BASE_DIR / img_path
                try:
                    img = Image.open(img_path)
                    if img.mode == 'RGBA':
                        background = Image.new('RGBA', img.size, (255,255,255,255))
                        img = Image.alpha_composite(background, img)
                    col_idx = i % cols_count
                    with cols[col_idx]:
                        st.image(img, caption=f"{pr['folder_name']} ({pr['type']})", width='stretch')
                except Exception:
                    st.write(f"Missing {pr['folder_name']}")

            st.write("### Outfit Details")
            for _, r in pieces.iterrows():
                st.write(f"- **{r['type'].capitalize()}**: {r['folder_name']} ({r['class']})")
