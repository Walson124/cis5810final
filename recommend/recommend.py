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

BASE_DIR = Path(__file__).resolve().parent.parent
WARDROBE_CSV = (BASE_DIR / 'data/wardrobe.csv').resolve()
wardrobe_df = pd.read_csv(WARDROBE_CSV)

# keep a set of used outfit IDs so we don't repeat ?

class Outfit:
    def __init__(self, ids):  # ids is a list of ids for each piece (to look up inwardrove.df)
        self.ids = ids
        self.emb_c_score = self.compute_compatibility_emb()
        self.color_c_score = self.compute_compatibility_color()
        # save id for each outfit?

    def embeddings(self):
        "returns embeddings of outfit in the form of an array of tensors"
        pieces = wardrobe_df.loc[wardrobe_df['id'].isin(self.ids)]
        embs = [torch.load(p).flatten() for p in pieces['clip_emb_path'].tolist()]  # flatten each to 1D
        embs = torch.stack(embs)  # shape: [num_pieces, embedding_dim]
        return embs

    def compute_compatibility_emb(self):
        embs = self.embeddings()
        cos_sim_matrix = torch.mm(embs, embs.T)
        mask = ~torch.eye(embs.size(0), dtype = bool)
        score = cos_sim_matrix[mask].mean()  # score computed based on mean of off-diagonal cos sim
        return score

    def compute_compatibility_color(self):
        """evaluate color compatibility among pieces (list of pd.Series)"""
        pieces = wardrobe_df.loc[wardrobe_df['id'].isin(self.ids)]
        color_mats = []
        for _, row in pieces.iterrows():
            c = row["color_vec"]
            arr = np.array(eval(c), dtype=float)  # if stored as list; use json.loads(c) if stored as string
            color_mats.append(arr)

        def cosine_similarity(a, b):
            v1, v2 = a.flatten(), b.flatten()
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        scores = [cosine_similarity(p1, p2) for p1, p2 in combinations(color_mats, 2)]
        score = float(np.clip(np.mean(scores), 0, 1))
        self.color_c_score = score
        return score


def sample_outfit(used_combos):
    """
    Sample a valid, non_used outfit.
    returns a Outfit object
    """

    pieces = []
    types = wardrobe_df["type"].unique()

    use_fullbody = "fullbody" in types and np.random.rand() < 0.5
    if use_fullbody:
        pieces.append(wardrobe_df[wardrobe_df["type"] == "fullbody"].sample(1).iloc[0])
    else:
        if "top" in types:
            pieces.append(wardrobe_df[wardrobe_df["type"] == "top"].sample(1).iloc[0])
        if "bottom" in types:
            pieces.append(wardrobe_df[wardrobe_df["type"] == "bottom"].sample(1).iloc[0])

    if "shoes" in types:
        pieces.append(wardrobe_df[wardrobe_df["type"] == "shoes"].sample(1).iloc[0])

    # optional accessory
    if "accessories" in types and np.random.rand() < 0.2:
        pieces.append(wardrobe_df[wardrobe_df["type"] == "accessories"].sample(1).iloc[0])

    ids = tuple(sorted(row["id"] for row in pieces))

    if ids in used_combos:
        return sample_outfit(used_combos)
    used_combos.add(ids)

    outfit = Outfit(list(ids))
    return outfit


def print_outfit(outfit: Outfit):
    pieces = wardrobe_df.loc[wardrobe_df["id"].isin(outfit.ids)]

    for _, row in pieces.iterrows():
        part = row["type"]
        print(f"{part.upper():<10} | {row['folder_name']} ({row['class']})")

def recommend_outfit(k):
    ''' sample k outfits and return best one '''
    used_combos = set()
    
    best_score, best_outfit = 0, None
    
    for i in range(k):
        outfit = sample_outfit(used_combos)

        score = outfit.color_c_score  # alternative to use emb_c_score
        print_outfit(outfit)
        print(f"compatibility score(emb): {score:.3f}")
        if score > best_score:
            best_score = score
            best_outfit = outfit

    return best_outfit, best_score


def display_outfit(outfit, figsize=(12, 8)):
    """Display outfit pieces in a grid layout with labels."""
    if not outfit:
        print("No outfit to display")
        return

    fig = plt.figure(figsize=figsize)
    pieces = wardrobe_df.loc[wardrobe_df['id'].isin(outfit.ids)]
    n_pieces = len(pieces)

    cols = min(3, n_pieces)
    rows = (n_pieces + cols - 1) // cols  # ceiling division

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
            print(f"Error loading image for {row['folder_name']}: {e}")
            ax.text(0.5, 0.5, f"Missing\n{row['folder_name']}", 
                    ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def main():
    outfit1, score1 = recommend_outfit(5)
    # print_outfit(outfit1)
    # print(f"\nOutfit compatibility score: {score1:.3f}")
    display_outfit(outfit1)
    
if __name__ == "__main__":
    main()