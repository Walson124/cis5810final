import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import uuid

from globals import get_model, get_segment_folder_func
from recommend.wardrobe import add_pieces, OUT_DIR, OUTPUT_CSV

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

    # Save file with a fresh clothing_N index that never reuses existing ones
    existing = list(wardrobe_folder.glob("clothing_*.png"))
    nums = []
    for p in existing:
        stem = p.stem  # e.g. "clothing_6"
        try:
            n = int(stem.split("_")[-1])
            nums.append(n)
        except ValueError:
            pass

    next_num = max(nums) + 1 if nums else 1
    img_path = wardrobe_folder / f"clothing_{next_num}.png"
    Image.open(file).save(img_path)


    st.image(str(img_path), caption=f"Added {img_path.name}", width=300)
    new_files_saved = True

# Trigger segmentation once
if new_files_saved:
    st.session_state["run_seg"] = True
    st.rerun()

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

# One card per clothing_* folder
clothing_dirs = sorted([d for d in seg_folder.iterdir() if d.is_dir()])

if clothing_dirs:
    cols_count = 3
    for i in range(0, len(clothing_dirs), cols_count):
        cols = st.columns(cols_count)
        for j, clothing_dir in enumerate(clothing_dirs[i : i + cols_count]):
            clothing_name = clothing_dir.name  # e.g. "clothing_2"

            with cols[j]:
                # Prefer standardized.png if it exists, else first rgba/*.png
                std_path = clothing_dir / "standardized.png"
                if std_path.exists():
                    img_path = std_path
                else:
                    rgba_files = sorted((clothing_dir / "rgba").glob("*.png"))
                    if not rgba_files:
                        st.caption(f"{clothing_name} (no images)")
                        continue
                    img_path = rgba_files[0]

                try:
                    img = Image.open(img_path)
                    # same white-background treatment as Outfit Builder
                    if img.mode == "RGBA":
                        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                        img = Image.alpha_composite(bg, img)
                    st.image(img, width=200, caption=clothing_name)
                except Exception as e:
                    st.error(f"Error loading {clothing_name}: {e}")

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
