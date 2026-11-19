"""House-pad negatives: real buildings as "not a mound" examples (Microsoft footprints).

    python scripts/build_house_negatives.py N [tiles_per_region]

Bare-earth LiDAR drops the house but keeps its graded pad, the detector's commonest
false alarm. This writes N scenes (600 m at 1 m, like the mined negatives) centred on
randomly chosen Georgia buildings in rural to suburban settings, >= 3 km from every
known mound site and Archaeohub site. Each building in a scene is a look-alike focus
point with supervised background, so training sees thousands of real pads.
Scenes: data/training/v1/scenes/houses_NNNN (group "house_pads", train only).
"""
import sys, json, math, gzip
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem
from gamd.dataset import save_scene
from gamd.buildings import quadkey, _tile_urls, _get, CACHE
from build_training_set import region_of, km, ROOT
from mine_negatives import exclusion_points

SC = ROOT / "data/training/v1/scenes"
SIZE, MAX_FOCUS, EDGE = 600, 15, 60


def ga_tiles():
    """Level-9 quadkeys over Georgia, grouped by region of their centre."""
    from shapely.geometry import shape, Point
    ga = shape([f for f in json.load(open(ROOT / "data/us_states.json"))["features"] if f["properties"]["name"] == "Georgia"][0]["geometry"])
    minx, miny, maxx, maxy = ga.bounds; out = {}
    for lat in np.arange(miny, maxy, 0.25):
        for lon in np.arange(minx, maxx, 0.25):
            if ga.contains(Point(lon, lat)): out.setdefault(quadkey(lat, lon), (lat, lon))
    by = {}
    for qk, (lat, lon) in out.items(): by.setdefault(region_of(lat, lon), []).append(qk)
    return by, ga


def centroids(qk):
    """(lat, lon) of every building in a quadkey tile (downloads + caches the tile)."""
    pts = []
    for k, url in enumerate(_tile_urls(qk)):
        fp = CACHE / f"{qk}_{k}.geojsonl.gz"
        if not fp.exists(): fp.write_bytes(_get(url))
        for line in gzip.open(fp, "rt"):
            g = json.loads(line)["geometry"]
            ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
            pts.append((sum(p[1] for p in ring) / len(ring), sum(p[0] for p in ring) / len(ring)))
    return np.array(pts)


if __name__ == "__main__":
    n = int(sys.argv[1]); per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = np.random.default_rng(23)
    by, ga = ga_tiles(); ex = exclusion_points()
    regions = ("mountains", "piedmont", "coastal_plain")
    tiles = {r: list(rng.choice(sorted(by[r]), size=min(per, len(by[r])), replace=False)) for r in regions}
    print("tiles:", {r: len(t) for r, t in tiles.items()}, flush=True)
    pools = {}
    for r in regions:
        arr = np.concatenate([c for c in (centroids(q) for q in tiles[r]) if len(c)])
        pools[r] = arr; print(f"{r}: {len(arr):,} buildings", flush=True)
    done = {p.stem for p in SC.glob("houses_*.json")}; made = len(done); k = 0; tries = 0
    while made < n and tries < n * 40:
        tries += 1; r = regions[k % 3]; pool = pools[r]
        lat, lon = pool[int(rng.integers(len(pool)))]
        k += 1
        # rural-to-suburban: 3-60 buildings within the scene; away from known sites
        d = np.hypot((pool[:, 0] - lat) * 111000, (pool[:, 1] - lon) * 111000 * math.cos(math.radians(lat)))
        near = pool[d < SIZE / 2 - EDGE]
        if not (3 <= len(near) <= 60) or min(km((lat, lon), p) for p in ex) < 3: continue
        sid = f"houses_{made:04d}"
        try: z, tr, epsg = fetch_dem(float(lat), float(lon), SIZE, res_m=1.0)
        except Exception as e: print(sid, "fetch failed", e, flush=True); continue
        if np.isnan(z).mean() > 0.2: continue
        from pyproj import Transformer
        to_xy = Transformer.from_crs(4326, epsg, always_xy=True); inv = ~tr; focus = []
        for blat, blon in near[rng.permutation(len(near))][:MAX_FOCUS]:
            col, row = inv * to_xy.transform(blon, blat)
            if EDGE <= row < SIZE - EDGE and EDGE <= col < SIZE - EDGE:
                focus.append(dict(feature="building footprint", kind="house pad", row=int(row), col=int(col)))
        if not focus: continue
        meta = dict(id=sid, type="real", lat=float(lat), lon=float(lon), epsg=epsg, transform=list(tr)[:6], res=1.0,
                    region=r, group="house_pads", split="train_only", background_supervised=True,
                    instances=[], focus=focus, decoys=[], empty=True)
        save_scene(SC / sid, z, np.zeros(z.shape, "uint16"), np.zeros(z.shape, bool), meta)
        made += 1
        if made % 10 == 0: print(f"{made}/{n} scenes ({sum(len(json.loads((SC / f'houses_{i:04d}.json').read_text())['focus']) for i in range(made)):,} pads)", flush=True)
    print(f"done: {made} house scenes", flush=True)
