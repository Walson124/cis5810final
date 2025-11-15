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

from transformers import CLIPProcessor, CLIPModel
import torch, os

BASE_DIR = Path(__file__).resolve().parent.parent
# OUT_DIR = (BASE_DIR / "out").resolve()
OUT_DIR = (BASE_DIR / "wardrobe_segmented").resolve()
OUTPUT_CSV = (BASE_DIR / "data" / "wardrobe.csv").resolve()

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

def process_folder(folder_path, M: Models):
    """Read manifest.json, extract data, return one row dict."""
    manifest_path = folder_path / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    item_id = str(uuid.uuid4())[:8]  # short random id

    detections = manifest["detections"][0]
    cls = manifest["clip_topk"][0][0]
    group_type  = category(cls)
    rgba_path = detections["rgba_path"]  
    #"rgba_path": "out\\american_vintage_men_s_black_jacket\\rgba\\00_jacket.png",
    
    try:
        image = Image.open(rgba_path).convert("RGBA")
    except Exception as e:
        return {"image": str(folder_path.name), "error": f"open_failed: {e}"}

    color_vec = M.top_colors(image)
    clip_emb_path = M.clip_embedding(image, folder_path)

    return {
        "id": item_id,
        "folder_name": folder_path.name,
        "class": cls,
        "type": group_type,
        "rgba_path": str(rgba_path),
        "color_vec": color_vec.tolist(),
        "clip_emb_path": str(clip_emb_path)
    }

def add_pieces(img_paths: list[str]):
    # takes a list of folder paths of newly segmented images, and updates csv
    
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

    if OUTPUT_CSV.exists():
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
    else:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)  # <-- create folder if missing
        df.to_csv(OUTPUT_CSV, index=False)


def main():
    # process entire folder in 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = Models(device=device)
    
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
