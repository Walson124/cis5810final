'''
recommendation:
- tweak probability of generating a specific outfit in `sample_outfit`
- 

'''

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm
from itertools import combinations
import torch

BASE_DIR = Path(__file__).resolve().parents[1]  # cis5810final/
WARDROBE_CSV = BASE_DIR / 'data' / 'wardrobe.csv'

def load_wardrobe():
    """Load wardrobe CSV safely, raises informative error if not present."""
    if not WARDROBE_CSV.exists():
        raise FileNotFoundError(f"{WARDROBE_CSV} not found. Run wardrobe.py first!")
    df = pd.read_csv(WARDROBE_CSV)
    return df

# keep a set of used outfit IDs so we don't repeat ?

class Outfit:
    def __init__(self, ids, wardrobe_df=None):
        self.ids = ids
        self.wardrobe_df = wardrobe_df if wardrobe_df is not None else load_wardrobe()
        self.emb_c_score = self.compute_compatibility_emb()
        self.color_c_score = self.compute_compatibility_color()

    def embeddings(self):
        pieces = self.wardrobe_df.loc[self.wardrobe_df['id'].isin(self.ids)]
        embs = [torch.load(p).flatten() for p in pieces['clip_emb_path'].tolist()]
        return torch.stack(embs)

    def compute_compatibility_emb(self):
        embs = self.embeddings()
        cos_sim_matrix = torch.mm(embs, embs.T)
        mask = ~torch.eye(embs.size(0), dtype = bool)
        score = cos_sim_matrix[mask].mean()  # score computed based on mean of off-diagonal cos sim
        return score

    def compute_compatibility_color(self):
        pieces = self.wardrobe_df.loc[self.wardrobe_df['id'].isin(self.ids)]
        color_mats = [np.array(eval(c), dtype=float) for c in pieces['color_vec']]
        scores = [np.dot(p1.flatten(), p2.flatten()) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                  for p1, p2 in combinations(color_mats, 2)]
        return float(np.clip(np.mean(scores), 0, 1))


def sample_outfit(used_combos, wardrobe_df=None, max_attempts=50):
    wardrobe_df = wardrobe_df if wardrobe_df is not None else load_wardrobe()
    types = wardrobe_df["type"].unique()

    for attempt in range(max_attempts):
        pieces = []

        # Decide whether to use fullbody
        use_fullbody = "fullbody" in types and np.random.rand() < 0.5
        if use_fullbody and len(wardrobe_df[wardrobe_df["type"] == "fullbody"]) > 0:
            pieces.append(wardrobe_df[wardrobe_df["type"] == "fullbody"].sample(1).iloc[0])
        else:
            if "top" in types and len(wardrobe_df[wardrobe_df["type"] == "top"]) > 0:
                pieces.append(wardrobe_df[wardrobe_df["type"] == "top"].sample(1).iloc[0])
            if "bottom" in types and len(wardrobe_df[wardrobe_df["type"] == "bottom"]) > 0:
                pieces.append(wardrobe_df[wardrobe_df["type"] == "bottom"].sample(1).iloc[0])

        if "shoes" in types and len(wardrobe_df[wardrobe_df["type"] == "shoes"]) > 0:
            pieces.append(wardrobe_df[wardrobe_df["type"] == "shoes"].sample(1).iloc[0])
        if "accessories" in types and len(wardrobe_df[wardrobe_df["type"] == "accessories"]) > 0 and np.random.rand() < 0.2:
            pieces.append(wardrobe_df[wardrobe_df["type"] == "accessories"].sample(1).iloc[0])

        # Generate the combination ID
        ids = tuple(sorted(row["id"] for row in pieces))

        # Check if this combination is already used
        if ids not in used_combos:
            used_combos.add(ids)
            return Outfit(list(ids), wardrobe_df)

    # If we reach here, we couldn't find a new outfit
    raise ValueError("Could not sample a new outfit — not enough wardrobe items or all combinations used")


def print_outfit(outfit):
    pieces = outfit.wardrobe_df.loc[outfit.wardrobe_df["id"].isin(outfit.ids)]
    for _, row in pieces.iterrows():
        print(f"{row['type'].upper():<10} | {row['folder_name']} ({row['class']})")


def recommend_outfit(k, wardrobe_df=None):
    wardrobe_df = wardrobe_df if wardrobe_df is not None else load_wardrobe()
    used_combos = set()
    best_score, best_outfit = 0, None

    for _ in range(k):
        outfit = sample_outfit(used_combos, wardrobe_df)
        score = outfit.color_c_score
        print_outfit(outfit)
        print(f"compatibility score(color): {score:.3f}")
        if score > best_score:
            best_score, best_outfit = score, outfit

    return best_outfit, best_score


def display_outfit(outfit, figsize=(12, 8)):
    if not outfit:
        print("No outfit to display")
        return

    fig = plt.figure(figsize=figsize)
    pieces = outfit.wardrobe_df.loc[outfit.wardrobe_df['id'].isin(outfit.ids)]
    n_pieces = len(pieces)
    cols = min(3, n_pieces)
    rows = (n_pieces + cols - 1) // cols

    for i, (_, row) in enumerate(pieces.iterrows(), start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.axis('off')
        try:
            img_path = Path(row['rgba_path'])
            if not img_path.is_absolute():
                img_path = BASE_DIR / img_path
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            plt.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Missing\n{row['folder_name']}", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def main():
    outfit1, score1 = recommend_outfit(5)
    # print_outfit(outfit1)
    # print(f"\nOutfit compatibility score: {score1:.3f}")
    display_outfit(outfit1)
    
if __name__ == "__main__":
    main()