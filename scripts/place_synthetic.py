"""Step 6.4: sample areas across Georgia's regions, find level real ground, place
synthetic mounds, and save placements + a map + example tiles.

    python scripts/place_synthetic.py OUT_DIR [areas_per_region] [seed]
Writes OUT_DIR/placements.json, OUT_DIR/map.png, OUT_DIR/examples.png
"""
import sys, json, csv, math
from pathlib import Path
import numpy as np
from shapely.geometry import shape, Point
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade
from gamd import synth
from gamd.place import suitability, choose_spots
from gamd.landcover import fetch_nlcd, modern_mask

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
per_region = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
only = sys.argv[4] if len(sys.argv) > 4 else None   # run a single region (for parallel runs)
AREA = 600  # m

ga = shape([f for f in json.load(open(ROOT / "data/us_states.json"))["features"] if f["properties"]["name"] == "Georgia"][0]["geometry"])
REGIONS = {  # rough physiographic boxes (lat_min, lat_max, lon_min, lon_max)
    "mountains":     (34.45, 34.98, -85.40, -83.15),
    "piedmont":      (32.90, 34.30, -85.10, -82.20),
    "coastal_plain": (30.70, 32.40, -84.90, -81.60),
}
known = []
for f in ("ga_mounds_verified.csv", "ga_mound_candidates.csv"):
    for r in csv.DictReader(open(ROOT / "data" / f)):
        try: known.append((float(r["lat"]), float(r["lon"])))
        except (ValueError, KeyError): pass
def far_from_known(lat, lon, km=3.0):
    return all(math.hypot(lat - a, (lon - b) * math.cos(math.radians(lat))) * 111 > km for a, b in known)

areas = []
for region, (la0, la1, lo0, lo1) in REGIONS.items():
    if only and region != only: continue
    n = 0; attempts = 0
    while n < per_region and attempts < per_region * 20:
        attempts += 1
        lat, lon = float(rng.uniform(la0, la1)), float(rng.uniform(lo0, lo1))
        if not ga.contains(Point(lon, lat)) or not far_from_known(lat, lon): continue
        try: z, tr, epsg = fetch_dem(lat, lon, AREA, res_m=1.0)
        except Exception as e: print("skip", region, lat, lon, e); continue
        if np.isnan(z).mean() > 0.2: continue
        try: modern = modern_mask(fetch_nlcd(lat, lon, AREA, 1.0), 1.0)
        except Exception as e: print("skip (land cover)", region, lat, lon, e); continue
        ok = suitability(z) & ~modern          # graded pads, lots, roads read as level ground
        np.save(out / f"modern_{region}_{n + 1}.npy", modern)
        want = []; params = []
        for _ in range(int(rng.integers(4, 7))):
            dz, m, p = synth.random_mound(260, rng=rng)
            ys, xs = np.nonzero(m)
            want.append(max(int(np.hypot(np.ptp(xs), np.ptp(ys)) / 2) + 2, 8)); params.append((dz, m, p))
        spots = choose_spots(ok, want, rng)
        if not spots: continue
        n += 1
        np.save(out / f"area_{region}_{n}.npy", z.astype("float32"))
        areas.append(dict(region=region, idx=n, lat=lat, lon=lon, suitable_frac=round(float(ok.mean()), 3),
                          mounds=[dict(row=r, col=c, radius_px=rad, kind=str(params[want.index(rad)][2]["kind"]),
                                       height=round(float(params[want.index(rad)][2]["height"]), 2)) for r, c, rad in spots]))
        print(f"{region:13s} #{n} {lat:.4f},{lon:.4f} level/usable {ok.mean():5.1%}  placed {len(spots)}/{len(want)}")
if only:
    json.dump(areas, open(out / f"placements_{only}.json", "w"), indent=1); sys.exit(0)
for f in sorted(out.glob("placements_*.json")): areas += json.load(open(f))
json.dump(areas, open(out / "placements.json", "w"), indent=1)

# map of Georgia with sample areas
W = 700; minx, miny, maxx, maxy = ga.bounds
sx = lambda lon: (lon - minx) / (maxx - minx) * (W - 40) + 20
sy = lambda lat: (maxy - lat) / (maxy - miny) * (W * 1.1 - 40) + 20
mp = Image.new("RGB", (W, int(W * 1.1)), "white"); d = ImageDraw.Draw(mp)
d.polygon([(sx(x), sy(y)) for x, y in ga.exterior.coords], outline=(90, 90, 90))
col = {"mountains": (40, 120, 40), "piedmont": (200, 120, 0), "coastal_plain": (30, 90, 200)}
for a in areas:
    x, y = sx(a["lon"]), sy(a["lat"]); r = 3 + len(a["mounds"])
    d.ellipse([x - r, y - r, x + r, y + r], fill=col[a["region"]])
for la, lo in known: x, y = sx(lo), sy(la); d.line([(x - 4, y - 4), (x + 4, y + 4)], fill=(200, 0, 0), width=2); d.line([(x - 4, y + 4), (x + 4, y - 4)], fill=(200, 0, 0), width=2)
d.text((20, int(W * 1.1) - 60), "dots = sample areas (green mountains, orange Piedmont, blue coastal plain)\nsize = mounds placed; red x = known sites (kept 3 km away)", fill=(0, 0, 0))
mp.save(out / "map.png")

# examples: first area of each region with mounds inserted
T = 300; ex = Image.new("L", (T * 3, T * 2), 255); dd = ImageDraw.Draw(ex)
for k, region in enumerate(REGIONS):
    a = next((a for a in areas if a["region"] == region), None)
    if not a: continue
    z = np.load(out / f"area_{region}_{a['idx']}.npy").astype("float64"); zz = z.copy(); ok = suitability(z) & ~np.load(out / f"modern_{region}_{a['idx']}.npy")
    for mnd in a["mounds"]:
        dz, m, p = synth.random_mound(260, rng=np.random.default_rng(mnd["row"] * 7 + mnd["col"]))
        r0, c0 = mnd["row"] - 130, mnd["col"] - 130
        ys, xs = slice(max(r0, 0), min(r0 + 260, z.shape[0])), slice(max(c0, 0), min(c0 + 260, z.shape[1]))
        zz[ys, xs] += dz[ys.start - r0:ys.stop - r0, xs.start - c0:xs.stop - c0]
    hs = hillshade(zz, 1.0)[0]; okv = np.where(ok, 255, 90).astype("uint8")
    ex.paste(Image.fromarray(hs.astype("uint8")).resize((T, T)), (k * T, 0)); ex.paste(Image.fromarray(okv).resize((T, T)), (k * T, T))
    dd.text((k * T + 4, 4), f"{region}: {len(a['mounds'])} fakes placed", fill=255); dd.text((k * T + 4, T + 4), "white = suitable ground", fill=0)
ex.save(out / "examples.png")
print("areas:", len(areas), "mounds placed:", sum(len(a["mounds"]) for a in areas))
