import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path
project_root = Path(__file__).parents[1]  # cis5810final/
sys.path.insert(0, str(project_root))

st.title("Fashion Recommender 👗")
st.write("Welcome to the main page!")

# Import global shared resources
from streamlit_app.globals import get_model, get_segment_folder_func

# Get the cached model and segmentation function
M = get_model()
segment_fn = get_segment_folder_func()

st.write("Model loaded successfully ✅")
