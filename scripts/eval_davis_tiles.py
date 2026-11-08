"""Sanity-check the converted Davis model on his own labelled training tiles.

These are TRAINING tiles, so scores are optimistic — the point is only to prove the
conversion is faithful (if it can't find his own labelled features, the
preprocessing doesn't match what ArcGIS did). Usage:
    python scripts/eval_davis_tiles.py /path/to/Training_Data [norm]
norm = 'tv'  : 0-1 input, torchvision's built-in ImageNet normalisation (default)
       'raw' : 0-255 input, normalisation disabled
"""
import sys, glob, os
from pathlib import Path
import numpy as np, rasterio, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.davis import build_model, CLASSES

root = Path(sys.argv[1]); norm = sys.argv[2] if len(sys.argv) > 2 else "tv"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(Path(__file__).resolve().parents[1] / "models/davis2021/RCNN_2_Output.pth", dev)
if norm == "raw":
    model.transform.image_mean = [0.0, 0.0, 0.0]; model.transform.image_std = [1.0, 1.0, 1.0]
name_to_id = {v: k for k, v in CLASSES.items()}
stats = {c: [0, 0] for c in CLASSES.values()}  # found, total
fp = 0; n_tiles = 0
for cname in CLASSES.values():
    for lab in sorted(glob.glob(str(root / "labels" / cname / "*.tif"))):
        img_p = root / "images" / Path(lab).name
        if not img_p.exists(): continue
        gt = rasterio.open(lab).read(1) > 0
        if not gt.any(): continue
        x = rasterio.open(img_p).read([1, 2, 3]).astype("float32")
        x = torch.from_numpy(x / 255.0 if norm == "tv" else x).to(dev)
        with torch.no_grad(): out = model([x])[0]
        n_tiles += 1; stats[cname][1] += 1; hit = False
        for lbl, sc, m in zip(out["labels"].cpu(), out["scores"].cpu(), out["masks"].cpu()):
            if sc < 0.5: continue
            pm = m[0].numpy() > 0.5
            iou = (pm & gt).sum() / max((pm | gt).sum(), 1)
            if CLASSES[int(lbl)] == cname and iou > 0.25: hit = True
            elif iou < 0.05: fp += 1
        stats[cname][0] += hit
print(f"norm={norm} device={dev} tiles={n_tiles}")
for c, (f, t) in stats.items(): print(f"  {c:7s} found {f}/{t} = {f / max(t, 1):.0%}")
print(f"  extra detections not on any labelled feature (score>=0.5): {fp}")
