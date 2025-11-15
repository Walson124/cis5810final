import os, json, argparse, sys, traceback
from pathlib import Path
import streamlit as st
import shutil

import numpy as np
import torch, cv2
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    pipeline,
    SamModel, SamProcessor
)

# ---------------- Defaults ----------------
DET_MODEL      = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL      = "facebook/sam-vit-base"
DET_THRESH     = 0.15
NMS_IOU        = 0.50
GLOBAL_IOU     = 0.90
MIN_AREA_FRAC  = 0.01
TOPK_CLIP      = 1                  # pass CLIP’s top-k labels to DINO
VALID_EXTS     = {".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif"}

DEFAULT_LABELS = [
    "t-shirt", "shirt", "blouse", "sweater", "hoodie", "cardigan",
    "jacket", "coat", "blazer",
    "jeans", "trousers", "pants", "shorts", "skirt", "dress",
    "sneakers", "boots", "heels", "sandals",
    "bag", "backpack", "hat", "scarf", "gloves", "sunglasses"
]
SYN = {
    "sneakers": ["sneaker", "running shoe", "trainer"],
    "boots": ["boot", "ankle boot"],
    "heels": ["heel", "high heel", "stiletto"],
    "t-shirt": ["t shirt", "tee", "tee shirt"],
    "trousers": ["trousers", "pants", "slacks"],
    "sunglasses": ["sunglasses", "shades"],
}

# ---------------- Utils ----------------
def make_dirs_for_image(out_root: Path, image_stem: str):
    base = out_root / image_stem
    (base / "masks").mkdir(parents=True, exist_ok=True)
    (base / "rgba").mkdir(parents=True, exist_ok=True)
    (base / "viz").mkdir(parents=True, exist_ok=True)
    return base

def to_xyxy(b): return [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
def box_area_xyxy(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    area_a = max(0, ax2-ax1)*max(0, ay2-ay1)
    area_b = max(0, bx2-bx1)*max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def contains_xyxy(outer, inner, frac=0.9):
    ox1,oy1,ox2,oy2 = outer; ix1,iy1,ix2,iy2 = inner
    ix1c,iy1c = max(ox1,ix1), max(oy1,iy1)
    ix2c,iy2c = min(ox2,ix2), min(oy2,iy2)
    iw,ih = max(0,ix2c-ix1c), max(0,iy2c-iy1c)
    inter = iw*ih
    inner_area = max(0, ix2-ix1)*max(0, iy2-iy1)
    return inner_area > 0 and inter/(inner_area + 1e-6) >= frac

def per_class_nms(dets, iou=0.5):
    by_cls = {}
    for d in dets: by_cls.setdefault(d["label"], []).append(d)
    kept = []
    for cls, items in by_cls.items():
        boxes = np.array([to_xyxy(it["box"]) for it in items], dtype=np.float32)
        scores = np.array([it["score"] for it in items], dtype=np.float32)
        if len(boxes) == 0: continue
        xywh = boxes.copy()
        xywh[:,2] = boxes[:,2] - boxes[:,0]
        xywh[:,3] = boxes[:,3] - boxes[:,1]
        idxs = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou
        )
        if isinstance(idxs, np.ndarray): idxs = idxs.flatten().tolist()
        elif isinstance(idxs, list): idxs = [i[0] if isinstance(i,(list,np.ndarray)) else i for i in idxs]
        for i in idxs: kept.append(items[i])
    return kept

def global_dedupe(dets, iou_thr=0.90):
    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets_sorted:
        box = to_xyxy(d["box"])
        drop = any(iou_xyxy(box, to_xyxy(k["box"])) >= iou_thr or
                   contains_xyxy(to_xyxy(k["box"]), box, 0.9)
                   for k in kept)
        if not drop: kept.append(d)
    return kept

def keep_topk(dets, k=1):
    return sorted(dets, key=lambda d: d["score"], reverse=True)[:k]

def expand_with_synonyms(labels):
    out, seen = [], set()
    for l in labels:
        for x in [l] + SYN.get(l, []):
            if x not in seen:
                seen.add(x); out.append(x)
    return out

# ---------------- Models (load once) ----------------
class Models:
    def __init__(self, labels, det_model, sam_model, device):
        # CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = labels

        # Grounding-DINO
        self.detector = pipeline(
            task="zero-shot-object-detection",
            model=det_model,
            device=(0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1)
        )

        # SAM
        self.sam = SamModel.from_pretrained(sam_model).to(device)
        self.sam_proc = SamProcessor.from_pretrained(sam_model)
        self.device = device

    def clip_topk(self, image_pil, k):
        inputs = self.clip_proc(text=self.labels, images=[image_pil], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            probs = out.logits_per_image.softmax(dim=1)[0]
        idx = torch.topk(probs, k=min(k, len(self.labels))).indices.tolist()
        return [(self.labels[i], float(probs[i])) for i in idx]

# ---------------- Core per-image processing ----------------
def process_image(image_path: Path, out_root: Path, M: Models,
                  det_thresh=DET_THRESH, nms_iou=NMS_IOU, global_iou=GLOBAL_IOU,
                  min_area_frac=MIN_AREA_FRAC, topk_clip=TOPK_CLIP, keep_k=None):
    try:
        im_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"image": str(image_path), "error": f"open_failed: {e}"}

    W, H = im_pil.size
    image_np = np.array(im_pil)[:, :, ::-1]  # BGR

    image_stem = image_path.stem
    out_base = make_dirs_for_image(out_root, image_stem)

    # 1) CLIP → choose labels
    topk = M.clip_topk(im_pil, k=topk_clip)
    clip_labels = [t[0] for t in topk]
    det_labels = expand_with_synonyms(clip_labels)

    # 2) Grounding-DINO → boxes
    raw = M.detector(im_pil, candidate_labels=det_labels, threshold=det_thresh)
    raw = [d for d in raw if d["score"] >= det_thresh]
    if not raw:
        return {
            "image": str(image_path),
            "clip_topk": topk,
            "detections": [],
            "note": "no_detections"
        }

    dets = per_class_nms(raw, iou=nms_iou)
    dets = global_dedupe(dets, iou_thr=global_iou)
    if keep_k is not None:
        dets = keep_topk(dets, k=keep_k)

    # 3) SAM → best mask per box
    manifest_items = []
    for i, d in enumerate(dets):
        box = to_xyxy(d["box"])
        if box_area_xyxy(box) / (W*H) < min_area_frac:
            continue

        inputs = M.sam_proc(images=im_pil, input_boxes=[[box]], return_tensors="pt").to(M.device)
        with torch.no_grad():
            out = M.sam(**inputs)

        masks_raw = M.sam_proc.post_process_masks(
            out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )[0][0]  # (num_masks, H, W)

        # choose mask by IoU score, fallback: largest area
        try:
            scores = out.iou_scores[0, 0]
            best_idx = int(scores.argmax().item())
        except Exception:
            areas = masks_raw.flatten(1).sum(dim=1)
            best_idx = int(areas.argmax().item())

        mask = (masks_raw[best_idx].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        # light cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        mask = cv2.medianBlur(mask, 3)

        # Save files
        mask_path = out_base / "masks" / f"{i:02d}_{d['label']}.png"
        cv2.imwrite(str(mask_path), mask)

        rgba = Image.new("RGBA", im_pil.size)
        rgba.paste(im_pil, (0, 0))
        rgba.putalpha(Image.fromarray(mask))
        rgba_path = out_base / "rgba" / f"{i:02d}_{d['label']}.png"
        rgba.save(str(rgba_path))

        viz = image_np.copy()
        m3 = mask > 0
        overlay = viz.copy()
        overlay[m3] = (0.6*overlay[m3] + 0.4*np.array((24,200,255), dtype=np.uint8)).astype(np.uint8)
        viz = cv2.addWeighted(overlay, 0.35, viz, 0.65, 0)
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(viz, (x1,y1), (x2,y2), (0,180,255), 2)
        txt = f"{d['label']} {d['score']:.2f}"
        cv2.putText(viz, txt, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(viz, txt, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        viz_path = out_base / "viz" / f"{i:02d}_{d['label']}.jpg"
        cv2.imwrite(str(viz_path), viz)

        manifest_items.append({
            "index": i,
            "class": d["label"],
            "score": float(d["score"]),
            "box_xyxy": [float(b) for b in box],
            "mask_path": str(mask_path),
            "rgba_path": str(rgba_path),
            "viz_path": str(viz_path)
        })

    # per-image manifest
    per_image_manifest = {
        "image": str(image_path),
        "clip_topk": topk,
        "detections": manifest_items
    }
    with open(out_base / "manifest.json", "w") as f:
        json.dump(per_image_manifest, f, indent=2)

    return per_image_manifest


##########################
# Segment folder function
##########################
def segment_folder(images_dir, out_root="out", labels=None, M=None):
    if M is None:
        # fallback — but normally shouldn't be used
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if labels is None:
            labels = DEFAULT_LABELS
        M = Models(labels=labels, det_model=DET_MODEL, sam_model=SAM_MODEL, device=device)
    
    images_dir = Path(images_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    results = []

    for p in paths:
        img_stem = p.stem
        seg_img_folder = out_root / img_stem / "rgba"

        # Skip already processed items
        if seg_img_folder.exists() and any(seg_img_folder.iterdir()):
            results.append({"image": str(p), "skipped": True})
            continue

        try:
            # result = process_image(
            #     image_path=p,
            #     out_root=out_root,
            #     M=M,
            #     det_thresh=0.25,
            #     nms_iou=0.50,
            #     global_iou=0.60,
            #     min_area_frac=0.003,
            #     topk_clip=5,
            #     keep_k=2
            # )
            result = process_image(
                image_path=p, 
                out_root=out_root, 
                M=M, 
                det_thresh=DET_THRESH, 
                nms_iou=NMS_IOU, 
                global_iou=GLOBAL_IOU, 
                min_area_frac=MIN_AREA_FRAC, 
                topk_clip=TOPK_CLIP, 
                keep_k=1 
            )
            # verify output creation
            out_base = out_root / p.stem
            if not (out_base.exists() and any(out_base.rglob("*"))):
                raise RuntimeError("Segmentation produced no files")

        except Exception as e:
            result = {"image": str(p), "error": str(e)}
            # cleanup partial
            out_base = out_root / p.stem
            if out_base.exists():
                shutil.rmtree(out_base)

        results.append(result)

    return results


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser("Folder mode: CLIP → Grounding-DINO → SAM")
    ap.add_argument("--images_dir", required=True, help="Folder containing images")
    ap.add_argument("--out", default="out", help="Output root folder")
    ap.add_argument("--labels", nargs="*", default=DEFAULT_LABELS, help="CLIP candidate labels")
    ap.add_argument("--topk_clip", type=int, default=TOPK_CLIP)
    ap.add_argument("--det_thresh", type=float, default=DET_THRESH)
    ap.add_argument("--nms_iou", type=float, default=NMS_IOU)
    ap.add_argument("--global_iou", type=float, default=GLOBAL_IOU)
    ap.add_argument("--min_area_frac", type=float, default=MIN_AREA_FRAC)
    ap.add_argument("--keep_k", type=int, default=1, help="Keep only top-K detections per image")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_root   = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load models once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = Models(labels=args.labels, det_model=DET_MODEL, sam_model=SAM_MODEL, device=device)

    # Collect images
    paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not paths:
        print(f"No images found in {images_dir}. Valid extensions: {sorted(VALID_EXTS)}")
        sys.exit(1)

    global_manifest = {"images_dir": str(images_dir), "results": []}
    for p in paths:
        try:
            result = process_image(
                image_path=p, out_root=out_root, M=M,
                det_thresh=args.det_thresh, nms_iou=args.nms_iou,
                global_iou=args.global_iou, min_area_frac=args.min_area_frac,
                topk_clip=args.topk_clip, keep_k=args.keep_k
            )
        except Exception as e:
            result = {"image": str(p), "error": f"exception: {e}", "traceback": traceback.format_exc()}

        global_manifest["results"].append(result)
        print(f"Processed: {p.name}  →  {('OK' if 'detections' in result else 'ERR')}")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(global_manifest, f, indent=2)

    print(f"Done. Wrote outputs under: {out_root}")

if __name__ == "__main__":
    # Optional: put models on another disk
    # os.environ["TRANSFORMERS_CACHE"] = "/mnt/otherdisk/hf_cache"
    # os.environ["HF_HOME"] = "/mnt/otherdisk/hf_home"
    main()
