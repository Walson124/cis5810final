# pages/Outfit Builder.py
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from recommend.recommend import recommend_outfit, display_outfit, Outfit, render_outfit_on_mannequin
from recommend.wardrobe import add_pieces, OUT_DIR
import recommend.gemini_client as gemini_client


BASE_DIR = Path(__file__).resolve().parents[2]  # go up 2 levels to cis5810final
WARDROBE_CSV = BASE_DIR / "data" / "wardrobe.csv"

st.title("Outfit Builder 👗")
st.write("Generate outfit recommendations from your wardrobe!")

# Load wardrobe CSV (rebuild if missing)
seg_root = OUT_DIR  # this is cis5810final/wardrobe_segmented

if not WARDROBE_CSV.exists():
    # Try to rebuild from existing segmented folders
    if not seg_root.exists():
        st.warning("No wardrobe data found. Go to 'My Wardrobe' to upload and segment clothes first.")
        st.stop()

    seg_folders = [f for f in seg_root.iterdir() if f.is_dir()]
    if not seg_folders:
        st.warning("No segmented wardrobe items found. Go to 'My Wardrobe' to upload and segment clothes first.")
        st.stop()

    with st.spinner("Building wardrobe data from segmented items..."):
        add_pieces(seg_folders)  # overwrites/creates data/wardrobe.csv

# After attempting rebuild, try to load
if not WARDROBE_CSV.exists():
    st.warning("Could not create wardrobe.csv. Check logs or rerun segmentation.")
    st.stop()

wardrobe_df = pd.read_csv(WARDROBE_CSV)

st.write(f"Found {len(wardrobe_df)} items in wardrobe.")

# Optional filters
types = wardrobe_df["type"].unique()
selected_types = st.multiselect("Select clothing types to include:", types, default=list(types))

filtered_df = wardrobe_df[wardrobe_df["type"].isin(selected_types)]
if filtered_df.empty:
    st.warning("No items match the selected types.")
    st.stop()

# # Number of outfits to sample
# k = st.slider("How many outfit samples to evaluate?", min_value=1, max_value=20, value=5)
k = 1

# Button to generate outfit
if st.button("Generate Outfit"):
    st.info("Sampling outfits...")

    # Use the filtered wardrobe, not the full df
    try:
        outfit, score = recommend_outfit(k, wardrobe_df=filtered_df)
    except ValueError as e:
        st.error(f"Could not generate outfit: {e}")
        st.stop()

    # Guard against the case where recommend_outfit returns (None, score)
    if outfit is None:
        st.warning("Couldn't generate a valid outfit. Try adding more items or changing your filters.")
        st.stop()

    st.success(f"Generated outfit! Compatibility score: {score:.3f}")

    # Display outfit images in Streamlit
    pieces = wardrobe_df.loc[wardrobe_df["id"].isin(outfit.ids)]
    n_pieces = len(pieces)
    cols_count = min(3, n_pieces)
    rows_count = (n_pieces + cols_count - 1) // cols_count


    st.write("### Outfit Preview")
    cols = st.columns(cols_count)
    for i, (_, row) in enumerate(pieces.iterrows()):
        # Prefer standardized image if available, fall back to RGBA
        img_col = "standard_image_path" if "standard_image_path" in row.index else "rgba_path"
        img_path = Path(row[img_col])
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path
        try:
            img = Image.open(img_path)
            if img.mode == "RGBA":
                background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            col_idx = i % cols_count
            with cols[col_idx]:
                st.image(img, caption=f"{row['folder_name']} ({row['type']})", width='stretch')
        except Exception as e:
            st.error(f"Error loading image {row['folder_name']}: {e}")

    st.write("### Outfit Details")
    for _, row in pieces.iterrows():
        st.write(f"- **{row['type'].capitalize()}**: {row['folder_name']} ({row['class']})")

    # --- On-mannequin render ---
    mannequin_img_path = None
    try:
        mannequin_img_path = render_outfit_on_mannequin(outfit)
        # st.write("### On-Body Preview")
        # st.image(str(mannequin_img_path), width='stretch')
    except Exception as e:
        st.warning(f"Could not render mannequin preview: {e}")

    # Always call Gemini to beautify the mannequin render
    if mannequin_img_path is not None:
        try:
            client = gemini_client.init_client()
            ai_img = gemini_client.beautify_outfit_preview(client, mannequin_img_path)
            st.write("### Gemini-enhanced Preview")
            st.image(ai_img, width='stretch')
        except Exception as e:
            st.warning(f"Gemini enhancement failed: {e}")

