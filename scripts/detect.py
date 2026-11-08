"""Find possible mounds in a square of US LiDAR — standalone, no Archaeohub needed.

    python scripts/detect.py --point 34.1250,-84.8075 --size 800 --out results/etowah
    python scripts/detect.py --csv places.csv --out results          # name,lat,lon[,size_m]

Each run writes a folder with spots.csv, spots.geojson, the relief views (views/*.png)
and a close-up per spot (spots/spot_N.png). Elevation comes from USGS 3DEP (1 m),
fetched on the fly.
"""
import argparse, base64, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def save(res, out):
    (out / "views").mkdir(parents=True, exist_ok=True); (out / "spots").mkdir(exist_ok=True)
    for im in res["images"]:
        (out / "views" / im["name"]).write_bytes(base64.b64decode(im.pop("png")))
    cols = ["n", "lat", "lon", "score", "models_agree", "models_total", "height_m", "width_m", "landcover", "building_m"]
    feats = []
    with open(out / "spots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for s in res["flags"]:
            (out / "spots" / f"spot_{s['n']}.png").write_bytes(base64.b64decode(s.pop("chip_png")))
            w.writerow(s)
            feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[s["lon"], s["lat"]]),
                              properties={k: s.get(k) for k in cols if k not in ("lat", "lon")}))
    json.dump(dict(type="FeatureCollection", features=feats), open(out / "spots.geojson", "w"), indent=1)
    json.dump({k: v for k, v in res.items() if k not in ("images", "flags")}, open(out / "run.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", help="lat,lon of the square's centre")
    ap.add_argument("--csv", help="file with name,lat,lon[,size_m] columns")
    ap.add_argument("--size", type=float, default=1000, help="square side in metres (default 1000)")
    ap.add_argument("--name", default="scan")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt", nargs="*", help="model checkpoint(s); default runs/final/all/ckpt_final.pt")
    a = ap.parse_args()
    import torch
    from gamd.reading import run_reading, default_ckpts
    from gamd.scan import load
    jobs = []
    if a.point:
        lat, lon = (float(v) for v in a.point.replace(" ", "").split(","))
        jobs.append(dict(name=a.name, lat=lat, lon=lon, size_m=a.size))
    if a.csv:
        jobs += [dict(name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]), size_m=float(r.get("size_m") or a.size)) for r in csv.DictReader(open(a.csv))]
    if not jobs: ap.error("give --point or --csv")
    ckpts = a.ckpt or default_ckpts()
    if not ckpts: sys.exit("No model found: put the checkpoint at runs/final/all/ckpt_final.pt or pass --ckpt.")
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    models = [load(c, dev)[0] for c in ckpts]; names = [Path(c).name for c in ckpts]
    for j in jobs:
        out = Path(a.out) / j["name"].replace(" ", "_").replace("/", "-") if a.csv else Path(a.out)
        t0 = time.time()
        res = run_reading(j, models, names, dev, lambda p, s: print(f"  {p:3.0f}%  {s}", flush=True))
        save(res, out)
        print(f"{j['name']}: {len(res['flags'])} spot(s) in {time.time() - t0:.0f}s -> {out}/", flush=True)
