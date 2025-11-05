import json
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
# Anchor BASE_DIR to the repository/project root (one level above `recommend/`)
# This makes paths in the manifest like "out\\..." resolve correctly from repo root.
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = (BASE_DIR / "out").resolve()
OUTPUT_CSV = (BASE_DIR / "data" / "wardrobe.csv").resolve()

def extract_top_colors(image_path, n_colors=3):
    """Extract top N colors from an RGBA image, return normalized RGB vector."""
    image = Image.open(image_path).convert("RGBA")
    arr = np.array(image)

    # remove transparent pixels (alpha > 0)
    arr = arr[arr[:, :, 3] > 0]
    arr = arr[:, :3]  # drop alpha channel
    arr = arr.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(arr)
    colors = kmeans.cluster_centers_.astype(float)

    # flatten and normalize 0–1:
    # color_vec = colors.flatten() / 255.0

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


def process_folder(folder_path):
    """Read manifest.json, extract data, return one row dict."""
    manifest_path = folder_path / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    detections = manifest["detections"][0]
    cls = manifest["clip_topk"][0][0]
    rgba_path = detections["rgba_path"]  
    #"rgba_path": "out\\american_vintage_men_s_black_jacket\\rgba\\00_jacket.png",

    color_vec = extract_top_colors(BASE_DIR / rgba_path)
    group_type = category(cls)

    item_id = str(uuid.uuid4())[:8]  # short random id
    return {
        "id": item_id,
        "folder_name": folder_path.name,
        "class": cls,
        "type": group_type,
        "rgba_path": str(rgba_path),
        "color_vec": color_vec.tolist(),
    }


def main():
    rows = []
    for subfolder in OUT_DIR.iterdir():
        if subfolder.is_dir():
            row = process_folder(subfolder)
            if row:
                rows.append(row)
                print(f"processing {row['folder_name']}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved wardrobe data to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
