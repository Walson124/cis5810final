# wardrobe.py
"""
process outputs from segmentation and store into a csv with cols:
    id,folder_name,class,type,rgba_path,color_vec,clip_emb_path

def add_pieces(img_paths: list[str]):
    # takes a list of folder paths of newly segmented images, and updates csv

"""

import json
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import recommend.gemini_client as gemini_client

from transformers import CLIPProcessor, CLIPModel
import torch, os

BASE_DIR = Path(__file__).resolve().parent.parent
# OUT_DIR = (BASE_DIR / "out").resolve()
OUT_DIR = (BASE_DIR / "wardrobe_segmented").resolve()
OUTPUT_CSV = (BASE_DIR / "data" / "wardrobe.csv").resolve()

USE_GEMINI_STANDARDIZE_IMAGES = True  # flip off if too slow

STANDARDIZED_NAME = "standardized.png"  # one per clothing folder

class Models:
    def __init__(self, device):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.kmeans = KMeans(n_clusters=3, n_init=10)
        self.device = device

    def clip_embedding(self, image: Image, folder_path):  # ensure imput Image is RGBA
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) 
        image = background  # set white background
        inputs = self.clip_proc(images = image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        embedding = emb / emb.norm(dim=1, keepdim=True)  # normalize to compute cosine similarity later on
        
        emb_path = ( folder_path / "embedding.pt" )
        torch.save(embedding.cpu(), emb_path)
        return emb_path
    
    def top_colors(self, image: Image):
        """
        Extract top 3 colors from an RGBA image
        """
        arr = np.array(image)

        arr = arr[arr[:, :, 3] > 0]
        arr = arr[:, :3]  # drop alpha channel
        arr = arr.reshape(-1, 3)
        self.kmeans.fit(arr)
        colors = self.kmeans.cluster_centers_.astype(float)
        return colors


def category(cls):
    tops = {"t-shirt", "shirt", "blouse", "sweater", "hoodie", "cardigan",
    "jacket", "coat", "blazer",}
    bottoms = {"jeans", "trousers", "pants", "shorts", "skirt"}
    fullbody = {"dress"}
    shoes = {"sneakers", "boots", "heels", "sandals"}
    accessories = {"backpack", "hat", "scarf", "gloves", "sunglasses"}

    cls = cls.lower().strip()
    if cls in tops:
        return "top"
    elif cls in bottoms:
        return "bottom"
    elif cls in fullbody:
        return "fullbody"
    elif cls in shoes:
        return "shoes"
    elif cls in accessories:
        return "accessories"
    else:
        return "unknown"

# def process_folder(folder_path, M: Models):
#     """Read manifest.json, extract data, return one row dict."""
#     manifest_path = folder_path / "manifest.json"
#     if not manifest_path.exists():
#         return None

#     with open(manifest_path, "r") as f:
#         manifest = json.load(f)

#     item_id = str(uuid.uuid4())[:8]  # short random id

#     detections = manifest["detections"][0]
#     cls = manifest["clip_topk"][0][0]
#     group_type  = category(cls)
#     # rgba_path = detections["rgba_path"]
#     # #    "rgba_path": "out\\gap_mens_brown_and_tan_shirt\\rgba\\00_jacket.png",
#     # #    OUT_DIR = (BASE_DIR / "wardrobe_segmented").resolve()
#     # # rgba_path = rgba_path.replace("out\\", str(OUT_DIR) + "\\", 1)

    
#     # try:
#     #     image = Image.open(rgba_path).convert("RGBA")
#     # except Exception as e:
#     #     return {"image": str(folder_path.name), "error": f"open_failed: {e}"}

#     # color_vec = M.top_colors(image)
#     # clip_emb_path = M.clip_embedding(image, folder_path)

#     # client = gemini_client.init_client()
#     # json_dict = gemini_client.get_fashion_attributes(client, rgba_path)
#     # description = json_dict['DESCRIPTION']
#     # attributes_path = folder_path / 'attributes.json'
#     # try:
#     #     with open(attributes_path, 'w') as f:
#     #         json.dump(json_dict, f, indent=4)

#     # except Exception as e:
#     #     print(f" Error saving JSON file: {e}")

#     # return {
#     #     "id": item_id,
#     #     "folder_name": folder_path.name,
#     #     "class": cls,
#     #     "type": group_type,
#     #     "rgba_path": str(rgba_path),
#     #     "color_vec": color_vec.tolist(),
#     #     "clip_emb_path": str(clip_emb_path),
#     #     "description": description,
#     #     "attributes_path": attributes_path
#     # }
#     rgba_path = Path(detections["rgba_path"])
#     if not rgba_path.is_absolute():
#         rgba_path = folder_path / "rgba" / rgba_path.name

#     # Load RGBA image
#     try:
#         image = Image.open(rgba_path).convert("RGBA")
#     except Exception as e:
#         return {"image": str(folder_path.name), "error": f"open_failed: {e}"}

#     # --- your existing color + CLIP logic ---
#     color_vec = M.top_colors(image)
#     clip_emb_path = M.clip_embedding(image, folder_path)

#     # --- NEW: standardized render (optional + cached) ---
#     standardized_path = folder_path / STANDARDIZED_NAME
#     if USE_GEMINI_STANDARDIZE_IMAGES and not standardized_path.exists():
#         try:
#             client = gemini_client.init_client()
#             std_img = gemini_client.render_standardized_image(client, rgba_path)
#             # std_img should be a PIL Image
#             standardized_path.parent.mkdir(parents=True, exist_ok=True)
#             std_img.save(standardized_path)
#         except Exception as e:
#             print(f"[WARN] Gemini standardization failed for {folder_path.name}: {e}")

#     # If Gemini failed or flag is off, just fall back to original RGBA
#     if standardized_path.exists():
#         display_path = standardized_path
#     else:
#         display_path = rgba_path

#     # --- existing Gemini attribute logic ---
#     json_dict, attributes_path, description = gemini_client.get_or_create_attributes(
#         client, rgba_path, folder_path
#     )

#     return {
#         "id": item_id,
#         "folder_name": folder_path.name,
#         "class": cls,
#         "type": group_type,
#         "rgba_path": str(rgba_path),
#         "standard_image_path": str(display_path),   # 👈 NEW COLUMN
#         "color_vec": color_vec.tolist(),
#         "clip_emb_path": str(clip_emb_path),
#         "description": description,
#         "attributes_path": attributes_path,
#     }

def process_folder(folder_path, M: Models):
    """Read manifest.json, extract data for one clothing item, return a row dict."""
    folder_path = Path(folder_path)
    manifest_path = folder_path / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    item_id = str(uuid.uuid4())[:8]  # short random id

    detections = manifest["detections"][0]
    cls = manifest["clip_topk"][0][0]
    group_type = category(cls)

    # --- resolve RGBA path ---
    rgba_path = Path(detections.get("rgba_path", ""))
    if not rgba_path.is_absolute():
        # Use the filename from manifest, but under this folder's rgba/ dir
        rgba_path = folder_path / "rgba" / rgba_path.name

    if not rgba_path.exists():
        # Fallback: first PNG in rgba/ if manifest path is weird
        candidates = sorted((folder_path / "rgba").glob("*.png"))
        if not candidates:
            return {"image": str(folder_path.name), "error": "no_rgba_image"}
        rgba_path = candidates[0]

    # --- load image ---
    try:
        image = Image.open(rgba_path).convert("RGBA")
    except Exception as e:
        return {"image": str(folder_path.name), "error": f"open_failed: {e}"}

    # --- color + CLIP embedding ---
    color_vec = M.top_colors(image)
    clip_emb_path = M.clip_embedding(image, folder_path)

    # --- try to init Gemini client (used for both std image + attributes) ---
    client = None
    try:
        client = gemini_client.init_client()
    except Exception as e:
        print(f"[WARN] Failed to initialize Gemini client for {folder_path.name}: {e}")

    # --- optional standardized render (image-to-image) ---
    standardized_path = folder_path / STANDARDIZED_NAME
    if client is not None and USE_GEMINI_STANDARDIZE_IMAGES and not standardized_path.exists():
        try:
            std_img = gemini_client.render_standardized_image(client, rgba_path)
            standardized_path.parent.mkdir(parents=True, exist_ok=True)
            std_img.save(standardized_path)
        except Exception as e:
            print(f"[WARN] Gemini standardization failed for {folder_path.name}: {e}")

    # choose which image to display in UI
    if standardized_path.exists():
        display_path = standardized_path
    else:
        display_path = rgba_path

    # --- attributes / description (robust to Gemini failure) ---
    attributes_path = folder_path / "attributes.json"
    json_dict = None
    description = ""

    if client is not None:
        try:
            json_dict, attributes_path, description = gemini_client.get_or_create_attributes(
                client, rgba_path, folder_path
            )
        except Exception as e:
            print(f"[WARN] Gemini attributes failed for {folder_path.name}: {e}")

    if json_dict is None:
        # Either no client or call failed: try to reuse existing JSON or create stub
        if attributes_path.exists():
            try:
                with open(attributes_path, "r") as f:
                    json_dict = json.load(f)
                description = json_dict.get("DESCRIPTION", "")
            except Exception as e:
                print(f"[WARN] Failed to read attributes.json for {folder_path.name}: {e}")
                json_dict = {"DESCRIPTION": ""}
                description = ""
        else:
            json_dict = {"DESCRIPTION": ""}
            description = ""
            attributes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(attributes_path, "w") as f:
                json.dump(json_dict, f, indent=4)

    # --- final row for wardrobe.csv ---
    return {
        "id": item_id,
        "folder_name": folder_path.name,
        "class": cls,
        "type": group_type,
        "rgba_path": str(rgba_path),
        "standard_image_path": str(display_path),
        "color_vec": color_vec.tolist(),
        "clip_emb_path": str(clip_emb_path),
        "description": description,
        "attributes_path": str(attributes_path),
    }


def add_pieces(img_paths: list[str]):
    """
    Takes a list of folder paths (typically *all* clothing_* folders)
    and rebuilds wardrobe.csv from scratch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = Models(device=device)

    rows = []
    for path in img_paths:
        path = Path(path)
        if path.is_dir():
            row = process_folder(path, M)
            if row:
                rows.append(row)
                print(f"{row['folder_name']} -> ok")

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved wardrobe data to {OUTPUT_CSV}")
    print(df.head())

def main():
    # process entire folder in 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = Models(device=device)
    
    subfolders = OUT_DIR.iterdir()
    add_pieces(subfolders)


    rows = []
    for subfolder in OUT_DIR.iterdir():
        if subfolder.is_dir():
            row = process_folder(subfolder, M)
            if row:
                rows.append(row)
                
                print(f"{row['folder_name']} -> ok")

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved wardrobe data to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
