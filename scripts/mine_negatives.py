"""Learn from mistakes (hard-negative mining), step 1: scan random ordinary Georgia
ground with a trained model and collect what it flags.

    python scripts/mine_negatives.py CKPT N_AREAS OUT_DIR [workers]

Areas are 600 m squares >= 3 km from every known mound site AND every Archaeohub
site (data/archaeohub_sites_exclude.csv), so the user's own places never become
"not a mound" lessons. Saves each area's DEM + precomputed relief6 bands and
OUT_DIR/flags.json; review with scripts/review_flags.py. Nothing is labelled
until a person accepts a flag as clearly modern or natural.
"""
import sys, csv, json, math, time
from pathlib import Path
import numpy as np, torch
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamd.dem import fetch_dem, DAVIS_RES_M
from gamd.bands import make_bands
from gamd.davis import CLASSES
from eval_real import load
from build_training_set import region_of, known_points, km, ROOT

CHIP, STRIDE = 200, 100


def exclusion_points():
    pts = known_points()
    for r in csv.reader(open(ROOT / "data/archaeohub_sites_exclude.csv")):
        try: pts.append((float(r[0]), float(r[1])))
        except (ValueError, IndexError): pass
    return pts


def sample_areas(n, seed=7):
    from shapely.geometry import shape, Point
    ga = shape([f for f in json.load(open(ROOT / "data/us_states.json"))["features"] if f["properties"]["name"] == "Georgia"][0]["geometry"])
    ex = exclusion_points(); rng = np.random.default_rng(seed); out = []
    minx, miny, maxx, maxy = ga.bounds
    regions = ("mountains", "piedmont", "coastal_plain")
    while len(out) < n:
        lat, lon = float(rng.uniform(miny, maxy)), float(rng.uniform(minx, maxx))
        if region_of(lat, lon) != regions[len(out) % 3] or not ga.contains(Point(lon, lat)): continue
        if min(km((lat, lon), p) for p in ex) < 3: continue
        out.append((lat, lon))
    return out


from gamd.scan import scan  # noqa: E402,F401  (moved to src/gamd/scan.py)


if __name__ == "__main__":
    ckpt, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, recipe = load(ckpt, dev); assert recipe == "relief6"
    flags = json.load(open(out / "flags.json")) if (out / "flags.json").exists() else []
    done = {f["area"] for f in flags} | {p.stem for p in out.glob("area_*.npz")}
    t0 = time.time()
    for i, (lat, lon) in enumerate(sample_areas(n)):
        aid = f"area_{i:04d}"
        if aid in done: continue
        try: z, tr, epsg = fetch_dem(lat, lon, 600, res_m=1.0)
        except Exception as e: print(aid, "fetch failed", e); continue
        if np.isnan(z).mean() > 0.2: continue
        z = z.astype("float64"); g = int(round(600 / DAVIS_RES_M))
        zg = ndi.zoom(np.where(np.isfinite(z), z, np.nanmedian(z)), g / z.shape[0], order=1, mode="nearest", prefilter=False)[:g, :g]
        bands = make_bands(zg, DAVIS_RES_M, "relief6")
        np.savez_compressed(out / f"{aid}.npz", z=z.astype("float32"), bands=bands)
        hits = scan(model, dev, bands)
        for k, (cy, cx, sc, w) in enumerate(hits):
            flags.append(dict(area=aid, lat=lat, lon=lon, epsg=epsg, transform=list(tr)[:6], region=region_of(lat, lon),
                              k=k, row_grid=round(cy, 1), col_grid=round(cx, 1), score=round(sc, 3), width_m=round(w, 1)))
        if not hits: flags.append(dict(area=aid, lat=lat, lon=lon, region=region_of(lat, lon), k=None))
        json.dump(flags, open(out / "flags.json", "w"))
        print(f"[{i + 1}/{n} {time.time() - t0:.0f}s] {aid} {region_of(lat, lon):13s} flags >=0.5: {sum(h[2] >= 0.5 for h in hits)}  (>=0.3: {len(hits)})", flush=True)
