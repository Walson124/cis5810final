# pages/Outfit Builder.py
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from recommend import recommend_outfit, display_outfit, Outfit

BASE_DIR = Path(__file__).resolve().parents[2]  # go up 2 levels to cis5810final
WARDROBE_CSV = BASE_DIR / "data" / "wardrobe.csv"

st.title("Outfit Builder 👗")
st.write("Generate outfit recommendations from your wardrobe!")

# Load wardrobe CSV
if not WARDROBE_CSV.exists():
    st.warning("No wardrobe data found. Upload and segment clothes first.")
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

# Number of outfits to sample
k = st.slider("How many outfit samples to evaluate?", min_value=1, max_value=20, value=5)

# Button to generate outfit
if st.button("Generate Outfit"):
    st.info("Sampling outfits...")
    
    try:
        outfit, score = recommend_outfit(k)
    except ValueError as e:
        st.error(str(e))  # show the error in Streamlit
    else:
        st.success(f"Generated outfit! Compatibility score: {score:.3f}")

        # Display outfit images in Streamlit
        pieces = wardrobe_df.loc[wardrobe_df["id"].isin(outfit.ids)]
        n_pieces = len(pieces)
        cols_count = min(3, n_pieces)
        rows_count = (n_pieces + cols_count - 1) // cols_count

        st.write("### Outfit Preview")
        cols = st.columns(cols_count)
        for i, (_, row) in enumerate(pieces.iterrows()):
            img_path = Path(row["rgba_path"])
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