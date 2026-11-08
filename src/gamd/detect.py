"""Sliding-window inference over a DEM, mirroring Esri's tiling: 200 px chips,
overlapping by `pad` px, keeping a detection only from the chip whose centre
region contains it (so nothing is counted twice)."""
import numpy as np, torch
from .davis import CLASSES
from .dem import davis_bands


def detect(model, z, res, chip=200, pad=50, min_score=0.3, device="cpu", **band_kw):
    bands = davis_bands(z, res, **band_kw)
    H, W = z.shape; step = chip - 2 * pad; out = []
    for y0 in range(-pad, H - pad, step):
        for x0 in range(-pad, W - pad, step):
            ys, xs = max(y0, 0), max(x0, 0)
            tile = np.zeros((3, chip, chip), "float32")
            sub = bands[:, ys:y0 + chip, xs:x0 + chip]
            tile[:, ys - y0:ys - y0 + sub.shape[1], xs - x0:xs - x0 + sub.shape[2]] = sub
            # Davis-faithful local DEM stretch is per chip (see davis_bands) — recompute band 3 here
            zc = z[ys:y0 + chip, xs:x0 + chip]
            if zc.size:
                b = davis_bands(zc, res, **band_kw)[2]
                tile[2, ys - y0:ys - y0 + b.shape[0], xs - x0:xs - x0 + b.shape[1]] = b
            with torch.no_grad():
                r = model([torch.from_numpy(tile / 255.0).to(device)])[0]
            for lbl, sc, box, m in zip(r["labels"].cpu(), r["scores"].cpu(), r["boxes"].cpu(), r["masks"].cpu()):
                if sc < min_score: continue
                cx, cy = float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2
                if not (pad <= cx < chip - pad and pad <= cy < chip - pad): continue  # owned by a neighbour chip
                bx = box.numpy() + [x0, y0, x0, y0]
                out.append(dict(cls=CLASSES[int(lbl)], score=float(sc), box=bx.tolist(),
                                diam_m=float(np.mean([bx[2] - bx[0], bx[3] - bx[1]]) * res)))
    return sorted(out, key=lambda d: -d["score"])
