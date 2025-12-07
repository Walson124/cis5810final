'''
recommendation:
- tweak probability of generating a specific outfit in `sample_outfit`
- 

TODO: add functinality to: 
- sample outfit based on preexiting piece: ✅
- compute compatibility of specific input pieces (by id perhaps) ✅
- tweak weights on each metric, ensure proper scoring
- implement a nearest neighbor search amongst entire wardrobe, instead of sampling and testing
- implement tfidf with SpaCy or Word2Vec embeddings
- attempt implementing attribute logic to retrieve natrual language senetences instead of descrete keywords for each attribute for downstream sentece to sentence matching
'''

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm
from itertools import combinations
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import argparse


BASE_DIR = Path(__file__).resolve().parents[1]  # cis5810final/
WARDROBE_CSV = BASE_DIR / 'data' / 'wardrobe.csv'

SCORE_WEIGHTS = {   # modify here 
            'embedding': 0.1,
            'color': 0.4,
            'tfidf': 0.2,
            'attribute': 0.3
        }

def load_wardrobe():
    """Load wardrobe CSV safely, raises informative error if not present."""
    if not WARDROBE_CSV.exists():
        raise FileNotFoundError(f"{WARDROBE_CSV} not found. Run wardrobe.py first!")
    df = pd.read_csv(WARDROBE_CSV)
    return df


class Outfit:

    def __init__(self, ids, wardrobe_df=None):
        self.ids = ids
        self.wardrobe_df = wardrobe_df if wardrobe_df is not None else load_wardrobe()
        self.pieces = self.wardrobe_df.loc[self.wardrobe_df['id'].isin(self.ids)]
        self.scores = {
            'embedding': self.compatibility_emb(),
            'color': self.compatibility_color(),
            'tfidf': self.compatibility_tfidf(),
            'attribute': self.compatibility_attribute()
        }
        self.score = 0
        
        for metric, score_value in self.scores.items():
            weight = SCORE_WEIGHTS[metric]
            weighted_score = score_value * weight
            self.score += weighted_score

    def embeddings(self):
        pieces = self.wardrobe_df.loc[self.wardrobe_df['id'].isin(self.ids)]
        embs = [torch.load(p, weights_only=True).flatten() for p in pieces['clip_emb_path'].tolist()]
        return torch.stack(embs)

    def compatibility_emb(self):
        embs = self.embeddings()
        cos_sim_matrix = torch.mm(embs, embs.T)
        mask = ~torch.eye(embs.size(0), dtype = bool)
        score = cos_sim_matrix[mask].mean()  # score computed based on mean of off-diagonal cos sim
        return score

    def compatibility_color(self):
        color_mats = [np.array(eval(c), dtype=float) for c in self.pieces['color_vec']]
        scores = [np.dot(p1.flatten(), p2.flatten()) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                  for p1, p2 in combinations(color_mats, 2)]
        return float(np.clip(np.mean(scores), 0, 1))
   
    def compatibility_tfidf(self):
        descriptions = self.pieces["description"].astype(str).tolist()

        vectorizer = TfidfVectorizer()

        tfidf = vectorizer.fit_transform(descriptions)
        sim_matrix = cosine_similarity(tfidf)
        # mask out diagonal (self similarity)
        mask = ~np.eye(sim_matrix.shape[0], dtype=bool)

        return float(sim_matrix[mask].mean())
    
    def compatibility_attribute(self):
        pieces = self.wardrobe_df.loc[self.wardrobe_df['id'].isin(self.ids)]
        outfit_list = []

        for i, row in pieces.iterrows():
            try:
                with open(row['attributes_path'], 'r') as file:
                    outfit_list.append(json.load(file))
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading {row['attributes_path']}: {e}")
        
        return score_attributes(
            outfit_list=outfit_list,
            weights= {
                'Dominant_Color': 3, 
                'Texture_Primary': 2,
                'Fit_Silhouette': 2,
                'Formality_Level': 4
            }
        )
    
    def print_scores(self):
        print(f"scores:")
        print(f"- embedding: {self.scores['embedding']}")
        print(f"- color: {self.scores['color']}")
        print(f"- tfidf: {self.scores['tfidf']}")
        print(f"- attributes: {self.scores['attribute']}")
        print(f"OVERALL: {self.score}")

########################## attribute matching logic ############################

def check_attribute_match(query_value, rule_list):
    """Checks if a single query value is compatible with a list of allowed values."""
    if isinstance(rule_list, list):
        # Handles list-based rules (e.g., color, texture, fit)
        if isinstance(query_value, list):
            # If the query itself is a list (like Style_Tags), check if ANY tag is in the rules
            return any(val in rule_list for val in query_value)
        else:
            # Simple check if the single value is in the rules list
            return query_value in rule_list
    # Note: You may add logic here for exact matches or range checks (e.g., Formality Level)
    return False

def score_attributes(outfit_list, weights):
    total_score = 0
    max_possible_score = 0
    
    # 1. Iterate over all pairs (Item A vs Item B)
    for i in range(len(outfit_list)):
        for j in range(len(outfit_list)):
            if i == j:
                continue # Skip comparing an item to itself
                
            query_item = outfit_list[i]['QUERY_VECTOR']
            rule_item = outfit_list[j]['KEY_VECTOR']
            
            # 2. Map QUERY attributes to KEY rules
            mappings = {
                'Dominant_Color': 'Rule_Color',
                'Texture_Primary': 'Rule_Texture',
                'Fit_Silhouette': 'Rule_Fit',
                'Formality_Level': 'Rule_Formality'
                # Add any other attribute mappings here
            }
            
            # 3. Calculate score for the A vs B pair
            pair_score = 0
            for query_key, rule_key in mappings.items():
                
                weight = weights.get(query_key, 1) # Default weight of 1
                max_possible_score += weight # Sum up total possible points
                
                # Check if the rule exists on the rule item
                if rule_key in rule_item:
                    is_compatible = check_attribute_match(
                        query_item.get(query_key), 
                        rule_item.get(rule_key)
                    )
                    if is_compatible:
                        pair_score += weight

            total_score += pair_score

    print(f'max_possible_score: {max_possible_score}; total_score: {total_score}')

    # 4. Normalize the score
    if max_possible_score > 0:
        compatibility_percentage = (total_score / max_possible_score)
        return compatibility_percentage
    return 0

########################## outfit utils ############################

def display_outfit(outfit, figsize=(12, 8)):
    if not outfit:
        print("No outfit to display")
        return

    fig = plt.figure(figsize=figsize)
    pieces = outfit.wardrobe_df.loc[outfit.wardrobe_df['id'].isin(outfit.ids)]
    n_pieces = len(pieces)
    cols = min(4, n_pieces)
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

def print_outfit(outfit):
    pieces = outfit.wardrobe_df.loc[outfit.wardrobe_df["id"].isin(outfit.ids)]
    for _, row in pieces.iterrows():
        print(f"{row['type'].upper():<10} | {row['folder_name']} ({row['class']})")
    print('\n')


########################## outfit recommendation ############################

def score_selected_outfit(outfit_ids, wardrobe_df=None):
    return Outfit(list(outfit_ids), wardrobe_df).score


def sample_outfit(used_combos, wardrobe_df=None, max_attempts=50, selected_ids=[]):
    '''
    Sample outfit by randomly selecting pieces and putting them together
    Given a list of ids `selected_pieces`, can create outfit on top of those items
    '''
    wardrobe_df = wardrobe_df if wardrobe_df is not None else load_wardrobe()
    types = wardrobe_df["type"].unique()
    if selected_ids:
        selected_pieces = [
            row for _, row in wardrobe_df.loc[wardrobe_df["id"].isin(selected_ids)].iterrows()
        ]
    else:
        selected_pieces = []

    for attempt in range(max_attempts):
        pieces = list(selected_pieces)
        selected_types = {p['type'] for p in pieces}

        # Decide whether to use fullbody
        use_fullbody = ("fullbody" in types) and (np.random.rand() < 0.5) and ('fullbody' not in selected_types)
        if (use_fullbody) and (len(wardrobe_df[wardrobe_df["type"] == "fullbody"]) > 0):
            pieces.append(wardrobe_df[wardrobe_df["type"] == "fullbody"].sample(1).iloc[0])
        else:
            if ("top" in types) and (len(wardrobe_df[wardrobe_df["type"] == "top"]) > 0) and ('top' not in selected_types):
                pieces.append(wardrobe_df[wardrobe_df["type"] == "top"].sample(1).iloc[0])
            if ("bottom" in types) and (len(wardrobe_df[wardrobe_df["type"] == "bottom"]) > 0) and ('bottom' not in selected_types):
                pieces.append(wardrobe_df[wardrobe_df["type"] == "bottom"].sample(1).iloc[0])

        if ("shoes" in types) and (len(wardrobe_df[wardrobe_df["type"] == "shoes"]) > 0) and ('shoes' not in selected_types):
            pieces.append(wardrobe_df[wardrobe_df["type"] == "shoes"].sample(1).iloc[0])
        if ("accessories" in types) and (len(wardrobe_df[wardrobe_df["type"] == "accessories"]) > 0) and (np.random.rand() < 0.5) and ('accessories' not in selected_types):
            pieces.append(wardrobe_df[wardrobe_df["type"] == "accessories"].sample(1).iloc[0])

        # Generate the combination ID
        ids = tuple(sorted(row["id"] for row in pieces))

        # Check if this combination is already used
        if ids not in used_combos:
            used_combos.add(ids)
            return Outfit(list(ids), wardrobe_df)

    # If we reach here, we couldn't find a new outfit
    raise ValueError("Could not sample a new outfit — not enough wardrobe items or all combinations used")

def recommend_outfit(k, wardrobe_df=None, amt='one', selected_ids=[]):
    wardrobe_df = wardrobe_df if wardrobe_df is not None else load_wardrobe()
    used_combos = set()
    best_score, best_outfit = 0, None
    outfits = []

    for _ in range(k):
        try:
            outfit = sample_outfit(used_combos, wardrobe_df=wardrobe_df, selected_ids=selected_ids)
        except ValueError:
            if amt == 'many':
                outfits.sort(key=lambda x: x.score, reverse=True)
                return outfits
            return best_outfit, best_score  # amt == 'one'

        outfits.append(outfit)
    
        score = outfit.score
        print_outfit(outfit)
        if score > best_score:
            best_score, best_outfit = score, outfit

    
    if (amt == 'many'):
        outfits.sort(key=lambda x: x.score, reverse=True)
        return outfits
    
    return best_outfit, best_score  # if (amt == 'one'): 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_ids", type=str, help="Ids of the outfit piece you'd like incorporated", nargs="+")

    args = parser.parse_args()

    if args.selected_ids:
        outfits = recommend_outfit(50, amt='one', selected_ids=args.selected_ids)
        print(f'input selection id: {args.selected_ids}')
        print(f'args.selected_ids: {score_selected_outfit(args.selected_ids)}')
    else: 
        outfits = recommend_outfit(50, amt='one')

    

    outfit = outfits[0]
    print('------FINAL OUTFIT------')
    print_outfit(outfit=outfit)   
    outfit.print_scores()
    display_outfit(outfit)

# Paths for mannequin compositing (keep these where they were)
MANNEQUIN_PATH = BASE_DIR / "assets" / "mannequin_front.png"
MANNEQUIN_OUT_DIR = BASE_DIR / "outfits"


def _crop_around_content(img: Image.Image, bg_threshold: int = 245) -> Image.Image:
    """
    Crop away mostly-white background so we keep just the garment.

    bg_threshold: 0–255, higher = more aggressive cropping.
    """
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)

    # mask of "non-white" pixels
    non_white = (arr[:, :, 0] < bg_threshold) | \
                (arr[:, :, 1] < bg_threshold) | \
                (arr[:, :, 2] < bg_threshold)

    if not non_white.any():
        # everything is white, nothing to crop
        return img

    ys, xs = np.where(non_white)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # small padding
    pad = 5
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(img.height, y_max + pad)
    x_max = min(img.width, x_max + pad)

    return img.crop((x_min, y_min, x_max, y_max))


def _fit_into_box(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """
    Crop around the garment and then resize to fit inside (box_w, box_h)
    while preserving aspect ratio.
    """
    img = _crop_around_content(img)
    w, h = img.size
    if w == 0 or h == 0:
        return img
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def render_outfit_on_mannequin(outfit, out_path: Path | None = None) -> Path:
    """
    Render an outfit onto a mannequin silhouette and save to disk.

    Returns the path to the rendered image.
    """
    MANNEQUIN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = Image.open(MANNEQUIN_PATH).convert("RGBA")
    canvas = base.copy()
    W, H = canvas.size

    # (x, y, w, h) boxes – tuned for a front-facing full-body mannequin.
    # You can tweak these numbers later if needed.
    layout = {
        "accessories": (int(W * 0.33), int(H * 0.06), int(W * 0.34), int(H * 0.12)),
        "top":         (int(W * 0.25), int(H * 0.20), int(W * 0.50), int(H * 0.30)),
        "bottom":      (int(W * 0.30), int(H * 0.48), int(W * 0.40), int(H * 0.28)),
        "shoes":       (int(W * 0.35), int(H * 0.80), int(W * 0.30), int(H * 0.13)),
    }

    # vertical alignment within each box
    v_align = {
        "accessories": "top",
        "top":         "top",
        "bottom":      "bottom",
        "shoes":       "bottom",
    }

    pieces = outfit.wardrobe_df.loc[outfit.wardrobe_df["id"].isin(outfit.ids)]

    for _, row in pieces.iterrows():
        ctype = row["type"]
        if ctype not in layout:
            continue

        x, y, bw, bh = layout[ctype]

        # Prefer standardized image if present, else fallback to RGBA
        col_name = (
            "standard_image_path"
            if "standard_image_path" in outfit.wardrobe_df.columns
            else "rgba_path"
        )
        img_col_val = row.get(col_name, row.get("rgba_path"))
        if pd.isna(img_col_val):
            continue

        img_path = Path(img_col_val)
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path

        if not img_path.exists():
            print(f"[WARN] Mannequin: image not found {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"[WARN] Mannequin: failed to open {img_path}: {e}")
            continue

        img = _fit_into_box(img, bw, bh)
        iw, ih = img.size

        # Horizontal center in the box
        dest_x = x + (bw - iw) // 2

        # Vertical alignment (top / bottom / center)
        align = v_align.get(ctype, "center")
        if align == "top":
            dest_y = y
        elif align == "bottom":
            dest_y = y + (bh - ih)
        else:
            dest_y = y + (bh - ih) // 2

        canvas.alpha_composite(img, dest=(dest_x, dest_y))

    # Default output path: stable per outfit IDs
    if out_path is None:
        ids_str = "_".join(sorted(map(str, outfit.ids)))
        out_path = MANNEQUIN_OUT_DIR / f"mannequin_{ids_str}.png"

    canvas.save(out_path)
    return out_path

    
if __name__ == "__main__":
    main()