import ast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Display color swatches from data/wardrobe.csv")
    p.add_argument("--n", type=int, default=12, help="How many rows to display (default: 12)")
    p.add_argument("--file", type=str, default=None, help="Path to wardrobe.csv (optional)")
    return p.parse_args()


def load_df(csv_path: Path):
    df = pd.read_csv(csv_path)
    # parse color_vec column which is stored as Python list literal
    def _parse(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None

    if "color_vec" in df.columns:
        df["color_vec_parsed"] = df["color_vec"].apply(_parse)
    else:
        df["color_vec_parsed"] = None
    return df


def show_swatches(df, n=12):
    rows = df.dropna(subset=["color_vec_parsed"]).head(n)
    if rows.empty:
        print("No color data to display.")
        return

    m = len(rows)
    cols = max(1, max(len(c) for c in rows["color_vec_parsed"]))

    fig, axes = plt.subplots(nrows=m, ncols=cols, figsize=(cols * 2, m * 0.9))

    # normalize axes shape for single-row or single-col
    if m == 1:
        axes = np.atleast_2d(axes)
    if cols == 1:
        axes = np.atleast_2d(axes).T

    for i, (_, row) in enumerate(rows.iterrows()):
        colors = row["color_vec_parsed"] or []
        for j in range(cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            if j < len(colors):
                rgb = np.array(colors[j], dtype=float) / 255.0
                # clamp
                rgb = np.clip(rgb, 0, 1)
                ax.imshow(np.ones((10, 10, 3)) * rgb.reshape(1, 1, 3))
            else:
                ax.imshow(np.ones((10, 10, 3)) * np.array([1.0, 1.0, 1.0]))

        # label the row on the left
        label = row.get("folder_name") or row.get("id") or "item"
        fig.text(0.01, 1 - (i + 0.5) / m, label, va="center", ha="left", fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)
    plt.show()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.file) if args.file else (repo_root / "data" / "wardrobe.csv")
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}\nRun the wardrobe script first or pass --file path")
        return

    df = load_df(csv_path)
    show_swatches(df, n=args.n)


if __name__ == "__main__":
    main()
