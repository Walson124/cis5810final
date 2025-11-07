"""Batch compute CLIP image embeddings for items listed in data/wardrobe.csv.

This script:
 - Loads `data/wardrobe.csv` (default) and looks for rows missing `clip_embedding`.
 - Loads CLIP model and processor once.
 - Batches images through CLIP and computes image embeddings via `model.get_image_features`.
 - Writes embeddings back into the CSV (JSON-serialized list in `clip_embedding` column).
 - Optionally writes a compressed .npz of embeddings keyed by item id.

Usage (from repo root):
    python recommend\compute_clip_embeddings.py --batch-size 32

Requirements:
    pip install torch transformers pandas pillow numpy tqdm
"""

import json
from pathlib import Path
import argparse
import math
import sys

import pandas as pd
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Compute CLIP embeddings for wardrobe items")
    p.add_argument("--csv", type=str, default=None, help="Path to wardrobe.csv (defaults to repo/data/wardrobe.csv)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP (default: 32)")
    p.add_argument("--device", type=str, default=None, help="torch device (cuda/cpu). Auto-detect if not set")
    p.add_argument("--force", action="store_true", help="Recompute embeddings even if present")
    p.add_argument("--save-npz", type=str, default=None, help="Optionally save embeddings to this npz file (path)")
    return p.parse_args()


def load_clip(device):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        print("Missing dependencies: please install torch and transformers.")
        raise

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


def is_missing_embedding(val):
    if pd.isna(val):
        return True
    s = str(val).strip()
    if s == "" or s.lower() == "none":
        return True
    # sometimes a JSON list string '[]' or 'nan' or something else
    if s == '[]':
        return True
    return False


def compute_embeddings(csv_path: Path, batch_size: int = 32, device: str = None, force: bool = False, save_npz: str = None):
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = Path(csv_path) if csv_path is not None else (repo_root / "data" / "wardrobe.csv")
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Always add columns if missing
    if "clip_embedding" not in df.columns:
        df["clip_embedding"] = None
    if "color_vec" not in df.columns:
        df["color_vec"] = None

    device = device or ("cuda" if ("torch" in sys.modules and __import__("torch").cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    model, proc = load_clip(device)
    import torch

    # Prepare all rows for processing
    idxs = df.index.tolist()
    repo_root = Path(__file__).resolve().parent.parent

    # Batch CLIP embedding computation
    batch_size = max(1, batch_size)
    emb_dim = None
    embeddings_dict = {}

    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start : start + batch_size]
        images = []
        valid_indices = []
        for i in batch_idxs:
            rgba_path = df.at[i, "rgba_path"]
            img_path = (repo_root / Path(rgba_path)).resolve()
            if not img_path.exists():
                print(f"Warning: image not found for row {i}: {img_path} (skipping)")
                continue
            try:
                images.append(Image.open(img_path).convert("RGB"))
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading image for row {i}: {e}")

        if not images:
            continue

        # CLIP embedding
        inputs = proc(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats.cpu().numpy()
        if emb_dim is None:
            emb_dim = feats.shape[1]

        for idx_row, emb, img in zip(valid_indices, feats, images):
            embeddings_dict[str(df.at[idx_row, "id"])] = emb.astype(float)
            df.at[idx_row, "clip_embedding"] = json.dumps(emb.astype(float).tolist())

            # KMeans color extraction (top 3 colors)
            try:
                arr = np.array(img.convert("RGBA"))
                arr = arr[arr[:, :, 3] > 0]
                arr = arr[:, :3]
                arr = arr.reshape(-1, 3)
                kmeans = KMeans(n_clusters=3, n_init=10)
                kmeans.fit(arr)
                colors = kmeans.cluster_centers_.astype(float)
                df.at[idx_row, "color_vec"] = json.dumps(colors.tolist())
            except Exception as e:
                print(f"Error computing KMeans for row {idx_row}: {e}")
                df.at[idx_row, "color_vec"] = None

        # close images
        for im in images:
            im.close()

        print(f"Processed rows {start}..{start+len(valid_indices)-1}")

    # save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"Wrote updated CSV: {csv_path}")

    # optional: save consolidated npz mapping id->embedding
    if save_npz:
        arr_ids = []
        arr_embs = []
        for k, v in embeddings_dict.items():
            arr_ids.append(k)
            arr_embs.append(v)
        arr_embs = np.stack(arr_embs, axis=0)
        np.savez_compressed(save_npz, ids=np.array(arr_ids), embs=arr_embs)
        print(f"Saved embeddings npz: {save_npz}")


def main():
    args = parse_args()
    device = args.device
    if device is None:
        # basic auto-detect
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    compute_embeddings(csv_path=args.csv, batch_size=args.batch_size, device=device, force=args.force, save_npz=args.save_npz)


if __name__ == "__main__":
    main()
