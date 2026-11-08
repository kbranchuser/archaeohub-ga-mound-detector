"""Private scan of chosen locations with several models (results never leave data/).

    python scripts/scan_sites.py SITES_CSV OUT_DIR CKPT [CKPT ...]
SITES_CSV: name,lat,lon[,size_m]. For each site: 3DEP at 1 m (default 1 km square),
relief6 bands, every model run over overlapping chips; flags from different models
within 30 m are merged and reported with how many models agree. Writes
OUT_DIR/<name>.png (hillshade, numbered flags) and OUT_DIR/flags.json.
"""
import sys, csv, json, math
from pathlib import Path
import numpy as np, torch
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamd.dem import fetch_dem_tiled, hillshade, DAVIS_RES_M
from gamd.bands import make_bands
from eval_real import load
from mine_negatives import scan

sites_csv, out = sys.argv[1], Path(sys.argv[2]); ckpts = sys.argv[3:]; out.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
models = [load(c, dev)[0] for c in ckpts]
try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
except OSError: font = None
from pyproj import Transformer
done = {x["site"] for x in json.load(open(out / "flags.json"))} if (out / "flags.json").exists() else set()
report = json.load(open(out / "flags.json")) if (out / "flags.json").exists() else []
for r in csv.DictReader(open(sites_csv)):
    if r["name"] in done: continue
    lat, lon, size = float(r["lat"]), float(r["lon"]), float(r.get("size_m") or 1000)
    try: z, tr, epsg = fetch_dem_tiled(lat, lon, size, res_m=1.0)
    except Exception as e: print(f"{r['name']}: elevation download failed ({e}); skipped, rerun later", flush=True); continue
    z = z.astype("float64")
    g = int(round(size / DAVIS_RES_M)); zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    bands = make_bands(ndi.zoom(zf, g / z.shape[0], order=1, mode="nearest", prefilter=False)[:g, :g], DAVIS_RES_M, "relief6")
    hits = []
    for mi, m in enumerate(models):
        for cy, cx, sc, w in scan(m, dev, bands):
            if sc >= 0.5: hits.append((cy, cx, sc, mi))
    groups = []
    for h in sorted(hits, key=lambda h: -h[2]):
        gg = next((gr for gr in groups if math.hypot(h[0] - gr[0][0], h[1] - gr[0][1]) * DAVIS_RES_M < 30), None)
        (gg.append(h) if gg else groups.append([h]))
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    hs = hillshade(zf, 1.0)[0]; im = Image.fromarray(hs.astype("uint8")).convert("RGB").resize((1000, 1000)); d = ImageDraw.Draw(im)
    s = 1000 / g; flags = []
    for k, gr in enumerate(groups, 1):
        cy, cx = np.mean([h[0] for h in gr]), np.mean([h[1] for h in gr])
        agree = len({h[3] for h in gr}); best = max(h[2] for h in gr)
        x, y = tr * (cx * DAVIS_RES_M, cy * DAVIS_RES_M); flon, flat = to_ll.transform(x, y)
        flags.append(dict(n=k, lat=round(flat, 6), lon=round(flon, 6), models_agree=agree, of=len(models), best_score=round(best, 2)))
        col = (255, 40, 40) if agree >= 3 else (255, 170, 0) if agree == 2 else (120, 160, 255)
        d.ellipse([cx * s - 18, cy * s - 18, cx * s + 18, cy * s + 18], outline=col, width=3)
        d.text((cx * s + 20, cy * s - 12), f"{k} ({agree}/{len(models)})", fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((10, 10), f"{r['name']}  {size:.0f} m  red=3-4 models agree, orange=2, blue=1", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(out / f"{r['name'].replace(' ', '_').replace('/', '-')}.png")
    report.append(dict(site=r["name"], lat=lat, lon=lon, flags=flags))
    json.dump(report, open(out / "flags.json", "w"), indent=1)
    print(f"{r['name']:40s} flags: {len(flags)}  agreed by >=3 models: {sum(f['models_agree'] >= 3 for f in flags)}", flush=True)
json.dump(report, open(out / "flags.json", "w"), indent=1)
