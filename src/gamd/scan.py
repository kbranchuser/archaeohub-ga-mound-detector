"""Load a trained detector and run it over a band stack with overlapping chips."""
import math
from pathlib import Path
import torch
from .davis import build_model, adapt_channels, CLASSES
from .bands import RECIPES
from .dem import DAVIS_RES_M

ROOT = Path(__file__).resolve().parents[2]
CHIP, STRIDE = 200, 100


def load(ckpt, dev):
    """Returns (model, recipe). ckpt = path to a trained checkpoint, or "davis" for the base model."""
    model = build_model(ROOT / "models/davis2021/RCNN_2_Output.pth", dev)
    if ckpt == "davis":
        return model.eval(), "davis"
    ck = torch.load(ckpt, map_location=dev, weights_only=True); recipe = ck.get("recipe", "davis")
    adapt_channels(model, RECIPES[recipe]).load_state_dict(ck["model"])
    return model.eval(), recipe


def scan(model, dev, bands):
    """Sliding 200 px chips (stride 100); detections >= 0.3 as [centre_row, centre_col, score, width_m], merged within 30 m."""
    n = bands.shape[1]; dets = []
    starts = list(range(0, n - CHIP + 1, STRIDE)) + ([n - CHIP] if (n - CHIP) % STRIDE else [])
    for y0 in starts:
        for x0 in starts:
            x = torch.from_numpy(bands[:, y0:y0 + CHIP, x0:x0 + CHIP].astype("float32") / 255.0).to(dev)
            with torch.no_grad(): p = model([x])[0]
            for lbl, sc, box in zip(p["labels"].cpu(), p["scores"].cpu(), p["boxes"].cpu()):
                if CLASSES[int(lbl)] not in ("Mound", "Ring") or sc < 0.3: continue
                cx, cy = float(box[0] + box[2]) / 2 + x0, float(box[1] + box[3]) / 2 + y0
                dets.append([cy, cx, float(sc), float(box[2] - box[0]) * DAVIS_RES_M])
    dets.sort(key=lambda d: -d[2]); keep = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) * DAVIS_RES_M > 30 for k in keep): keep.append(d)
    return keep
