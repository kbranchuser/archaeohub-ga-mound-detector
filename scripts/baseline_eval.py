"""Baseline: what does the unmodified Davis model say about each verified feature?

For every row in data/ga_mounds_verified.csv (mounds, reconstructions, look-alikes),
fetch a 300 m 3DEP tile centred on it at Davis's 1.524 m pixels (one 200 px chip),
run the model, and report the best detection whose centre falls within `near_m`
of the feature. Scoring:
  mound / reconstruction -> HIT if a Mound or Ring detection (score >= thr) is near centre
  lookalike              -> FALSE ALARM if a Mound or Ring detection is near centre
Usage: python scripts/baseline_eval.py [dem_levels_per_m] [thr]
"""
import sys, csv, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.davis import build_model, CLASSES
from gamd.dem import fetch_dem, davis_bands, DAVIS_RES_M

ROOT = Path(__file__).resolve().parents[1]
k = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
near_m = 40.0
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(ROOT / "models/davis2021/RCNN_2_Output.pth", dev)
cache = ROOT / "data/baseline_tiles"; cache.mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader(open(ROOT / "data/ga_mounds_verified.csv")) if r["label"] in ("mound", "reconstruction", "lookalike")]
out = []
for r in rows:
    key = f"{float(r['lat']):.5f}_{float(r['lon']):.5f}.npy"
    if (cache / key).exists():
        z = np.load(cache / key)
    else:
        z, _, _ = fetch_dem(float(r["lat"]), float(r["lon"]), 200 * DAVIS_RES_M); np.save(cache / key, z)
    x = torch.from_numpy(davis_bands(z, dem_levels_per_m=k).astype("float32") / 255.0).to(dev)
    with torch.no_grad(): p = model([x])[0]
    best = None
    for lbl, sc, box in zip(p["labels"].cpu(), p["scores"].cpu(), p["boxes"].cpu()):
        cx, cy = float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2
        if np.hypot(cx - 100, cy - 100) * DAVIS_RES_M > near_m: continue
        if best is None or sc > best[1]: best = (CLASSES[int(lbl)], float(sc))
    is_arch = bool(best and best[0] in ("Mound", "Ring") and best[1] >= thr)
    out.append(dict(feature=r["feature"], label=r["label"], top=best[0] if best else "-", score=round(best[1], 2) if best else 0.0,
                    result=("HIT" if is_arch else "miss") if r["label"] != "lookalike" else ("FALSE ALARM" if is_arch else "ok")))
for o in out: print(f"{o['label']:14s} {o['result']:11s} top={o['top']:6s} {o['score']:.2f}  {o['feature']}")
m = [o for o in out if o["label"] == "mound"]; la = [o for o in out if o["label"] == "lookalike"]
print(f"\nk={k} thr={thr}: original mounds found {sum(o['result']=='HIT' for o in m)}/{len(m)}; "
      f"look-alikes wrongly flagged {sum(o['result']=='FALSE ALARM' for o in la)}/{len(la)}")
json.dump(out, open(ROOT / f"data/baseline_k{int(k)}.json", "w"), indent=1)
