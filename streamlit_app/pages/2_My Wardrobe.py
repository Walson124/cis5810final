import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import uuid

from globals import get_model, get_segment_folder_func

# Always call these at the top of the page
M = get_model()
segment_folder = get_segment_folder_func()

# Initialize session state variables
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()
if "run_seg" not in st.session_state:
    st.session_state["run_seg"] = False

# Create directories
wardrobe_folder = Path("wardrobe_images")
seg_folder = Path("wardrobe_segmented")
wardrobe_folder.mkdir(exist_ok=True)
seg_folder.mkdir(exist_ok=True)

st.title("Wardrobe 👚")
st.write("Upload clothes or take a photo. Segmentation runs automatically when new items are added.")

with st.expander("Add Clothes"):
    img_file = st.camera_input("Take a photo")
    upload_files = st.file_uploader(
        "Or upload existing photo(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

# Normalize into a list
incoming = []
if img_file:
    incoming.append(img_file)
if upload_files:
    incoming.extend(upload_files)

# ==== HANDLE NEW FILES (only once per file) ====
new_files_saved = False
for file in incoming:
    file_id = getattr(file, "name", None) or str(uuid.uuid4())

    if file_id in st.session_state["processed_files"]:
        continue  # already handled this same image

    st.session_state["processed_files"].add(file_id)

    # Save file
    next_num = len(list(wardrobe_folder.glob("*.png"))) + 1
    img_path = wardrobe_folder / f"clothing_{next_num}.png"
    Image.open(file).save(img_path)

    st.image(str(img_path), caption=f"Added {img_path.name}", width=300)
    new_files_saved = True

# Trigger segmentation once
if new_files_saved:
    st.session_state["run_seg"] = True
    st.rerun()

from recommend.wardrobe import add_pieces

# Actually run segmentation
if st.session_state["run_seg"]:
    with st.spinner("Running segmentation..."):
        segment_folder(wardrobe_folder, out_root=seg_folder, M=M)
        
        # Generate/update wardrobe.csv
        segmented_folders = [f for f in seg_folder.iterdir() if f.is_dir()]
        if segmented_folders:
            add_pieces(segmented_folders)
            
    st.session_state["run_seg"] = False
    st.success("Segmentation complete and CSV updated!")
    st.rerun()

# ==== DISPLAY SEGMENTED RESULTS ====
st.write("### Segmented Wardrobe")
segmented_files = sorted(seg_folder.glob("*/rgba/*.png"))

if segmented_files:
    cols_count = 3
    for i in range(0, len(segmented_files), cols_count):
        cols = st.columns(cols_count)
        for j, f in enumerate(segmented_files[i:i + cols_count]):
            with cols[j]:
                st.image(str(f), width=200, caption=f.name)

                clothing_name = f.parents[1].name  # clothing_#
                if st.button(f"Delete {clothing_name}", key=f"{clothing_name}_{i}_{j}"):
                    # Delete segmented output and original image
                    shutil.rmtree(seg_folder / clothing_name)
                    orig = wardrobe_folder / f"{clothing_name}.png"
                    if orig.exists():
                        orig.unlink()

                    # Remove from processed file list
                    if f"{clothing_name}.png" in st.session_state["processed_files"]:
                        st.session_state["processed_files"].remove(f"{clothing_name}.png")

                    st.rerun()

else:
    st.info("No segmented items yet.")
