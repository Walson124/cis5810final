import sys
from pathlib import Path

# Make project root (cis5810final) visible to Python imports
project_root = Path(__file__).parents[1]  # streamlit_app/.. = cis5810final
sys.path.insert(0, str(project_root))

import streamlit as st
import torch
from segment.predict_and_segment_folder import Models, segment_folder, DET_MODEL, SAM_MODEL, DEFAULT_LABELS
from recommend.recommend import recommend_outfit, display_outfit, Outfit

# Cache the heavy model so it is created only once per session
@st.cache_resource
def load_model():
    print("🔧 Loading Models() once...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Models(
        labels=DEFAULT_LABELS,
        det_model=DET_MODEL,
        sam_model=SAM_MODEL,
        device=device
    )

def get_model():
    """Returns the cached model instance from session state."""
    if "M" not in st.session_state:
        st.session_state["M"] = load_model()
    return st.session_state["M"]

def get_segment_folder_func():
    """Returns the segmentation function, saved in session state."""
    if "segment_fn" not in st.session_state:
        st.session_state["segment_fn"] = segment_folder
    return st.session_state["segment_fn"]
