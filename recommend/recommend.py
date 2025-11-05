'''
recommendation:
define evaluation function for compatibity
take in DF or csv, 
and has functions that sample DF for valid outfits (top bottm shoes)
actual functions callable to give recommendations
- output ids of suggested recommendations
'''

from pathlib import Path
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations

BASE_DIR = Path(__file__).resolve().parent.parent
WARDROBE_CSV = (BASE_DIR / 'data/wardrobe.csv').resolve()
wardrobe_df = pd.read_csv(WARDROBE_CSV)

# keep a set of used outfit IDs so we don't repeat ?

def score_compatibility(pieces):
    """evaluate color compatibility among pieces (list of pd.Series)"""

    color_mats = []
    for row in pieces:
        c = row["color_vec"]
        color_mats.append(np.array(eval(c), dtype=float))

    def cosine_similarity(a, b):
        v1, v2 = a.flatten(), b.flatten()
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    scores = [cosine_similarity(p1, p2) for p1, p2 in combinations(color_mats, 2)]
    return float(np.clip(np.mean(scores), 0, 1))


def sample_outfit(used_combos):
    """
    Sample a valid, non_used outfit.
    returns a dictionary mapping of type to the row for a piece
    """

    outfit = {}
    types = wardrobe_df["type"].unique()

    use_fullbody = "fullbody" in types and np.random.rand() < 0.5
    if use_fullbody:
        outfit["fullbody"] = wardrobe_df[wardrobe_df["type"] == "fullbody"].sample(1).iloc[0]
    else:
        if "top" in types:
            outfit["top"] = wardrobe_df[wardrobe_df["type"] == "top"].sample(1).iloc[0]
        if "bottom" in types:
            outfit["bottom"] = wardrobe_df[wardrobe_df["type"] == "bottom"].sample(1).iloc[0]

    if "shoes" in types:
        outfit["shoes"] = wardrobe_df[wardrobe_df["type"] == "shoes"].sample(1).iloc[0]

    # optional accessory
    if "accessories" in types and np.random.rand() < 0.2:
        outfit["accessory"] = wardrobe_df[wardrobe_df["type"] == "accessories"].sample(1).iloc[0]

    ids = tuple(sorted(row["id"] for row in outfit.values()))
    if ids in used_combos:
        return sample_outfit(used_combos)
    used_combos.add(ids)

    return outfit


def print_outfit(outfit):
    for part, row in outfit.items():
        print(f"{part.upper():<10} | {row['folder_name']} ({row['class']})")


def recommend_outfit(k):
    ''' sample k outfits and return best one '''
    used_combos = set()
    
    best_score, best_outfit = 0, None
    
    for i in range(k):
        outfit = sample_outfit(used_combos)

        pieces = list(outfit.values())
        score = score_compatibility(pieces)
        # print_outfit(outfit)
        print(f"compatibility score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_outfit = outfit

    return best_outfit, best_score

    def display_outfit(outfit, figsize=(12, 8)):
        """Display outfit pieces in a grid layout with labels."""
        if not outfit:
            print("No outfit to display")
            return

        # Layout config
        fig = plt.figure(figsize=figsize)
    
        # Determine grid size based on outfit pieces
        n_pieces = len(outfit)
        if n_pieces <= 3:
            rows, cols = 1, n_pieces
        else:
            rows = (n_pieces + 2) // 3  # ceiling division
            cols = min(3, n_pieces)
    
        for idx, (part, row) in enumerate(outfit.items()):
            # Create subplot
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.axis('off')
        
            # Load and display image
            try:
                img_path = BASE_DIR / row['rgba_path']
                img = Image.open(img_path)
            
                # Convert RGBA to RGB with white background
                if img.mode == 'RGBA':
                    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(background, img)
            
                plt.imshow(img)
            
                # Add title with part type
                plt.title(f"{part.title()}\n{row['class']}", pad=10)
            
            except Exception as e:
                print(f"Error loading image for {part}: {e}")
                # Show empty placeholder
                ax.text(0.5, 0.5, f"Missing\n{part}", 
                       ha='center', va='center', transform=ax.transAxes)
    
        plt.tight_layout()
        plt.show()

def main():
    outfit1, score1 = recommend_outfit(5)
    print_outfit(outfit1)
        print(f"\nOutfit compatibility score: {score1:.3f}")
        display_outfit(outfit1)
    
if __name__ == "__main__":
    main()