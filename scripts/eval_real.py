"""Score a model on the verified Georgia features, same protocol as baseline_eval.py:
one 200 px chip (1.524 m) centred on each feature; a Mound/Ring detection with
score >= thr whose centre is within 40 m counts.

    python scripts/eval_real.py [checkpoint.pt | davis] [dem_levels_per_m] [thr]
"""
import sys, csv, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.davis import build_model, adapt_channels, CLASSES
from gamd.bands import make_bands, RECIPES, PAD
from gamd.dem import fetch_dem, davis_bands, DAVIS_RES_M

ROOT = Path(__file__).resolve().parents[1]
NEAR_M = 40.0


from gamd.scan import load  # noqa: E402,F401  (moved to src/gamd/scan.py; kept importable here)


def evaluate(model, dev, k=10.0, thr=0.5, groups=None, recipe="davis"):
    pad = 0 if recipe == "davis" else PAD   # relief recipes need context around the chip
    cache = ROOT / ("data/baseline_tiles" if recipe == "davis" else "data/eval_tiles_p75"); cache.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ROOT / "data/ga_mounds_verified.csv")) if r["label"] in ("mound", "reconstruction", "lookalike")]
    out = []
    for r in rows:
        group = r["site"].replace(" 5km", "").strip()
        if groups and group not in groups: continue
        key = f"{float(r['lat']):.5f}_{float(r['lon']):.5f}.npy"
        if (cache / key).exists(): z = np.load(cache / key)
        else: z, _, _ = fetch_dem(float(r["lat"]), float(r["lon"]), (200 + 2 * pad) * DAVIS_RES_M); np.save(cache / key, z)
        kw = dict(dem_levels_per_m=k) if recipe == "davis" else {}
        x = torch.from_numpy(make_bands(z, DAVIS_RES_M, recipe, pad=pad, **kw).astype("float32") / 255.0).to(dev)
        with torch.no_grad(): p = model([x])[0]
        best = None
        for lbl, sc, box in zip(p["labels"].cpu(), p["scores"].cpu(), p["boxes"].cpu()):
            cx, cy = float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2
            if np.hypot(cx - 100, cy - 100) * DAVIS_RES_M > NEAR_M: continue
            if CLASSES[int(lbl)] not in ("Mound", "Ring"): continue
            if best is None or sc > best: best = float(sc)
        hit = best is not None and best >= thr
        out.append(dict(feature=r["feature"], group=group, label=r["label"], score=round(best or 0.0, 2),
                        result=("HIT" if hit else "miss") if r["label"] != "lookalike" else ("FALSE ALARM" if hit else "ok")))
    return out


def summary(out):
    m = [o for o in out if o["label"] == "mound"]; la = [o for o in out if o["label"] == "lookalike"]
    return dict(mounds_found=sum(o["result"] == "HIT" for o in m), mounds=len(m),
                false_alarms=sum(o["result"] == "FALSE ALARM" for o in la), lookalikes=len(la))


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "davis"
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, recipe = load(ckpt, dev)
    out = evaluate(model, dev, k, thr, recipe=recipe)
    for o in out: print(f"{o['label']:14s} {o['result']:11s} {o['score']:.2f}  {o['feature']}")
    s = summary(out)
    print(f"\n{ckpt} ({recipe}) k={k} thr={thr}: mounds found {s['mounds_found']}/{s['mounds']}; look-alikes wrongly flagged {s['false_alarms']}/{s['lookalikes']}")
