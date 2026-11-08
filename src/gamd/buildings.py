"""Distance from points to the nearest building (Microsoft Global ML Building Footprints).

Bare-earth LiDAR removes houses but keeps the graded pad under them, which reads
as a low platform — the detector's commonest false alarm. Footprints come in
level-9 quadkey tiles (~70 km); each is fetched once and cached in data/cache/buildings.
"""
import csv, gzip, io, json, math, urllib.request
from pathlib import Path

INDEX = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
CACHE = Path(__file__).resolve().parents[2] / "data/cache/buildings"


def quadkey(lat, lon, z=9):
    s = math.sin(math.radians(lat))
    x = int((lon + 180) / 360 * 2 ** z); y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 2 ** z)
    return "".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(z - 1, -1, -1))


def _get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "ga-mound-detector"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()


def _tile_urls(qk):
    CACHE.mkdir(parents=True, exist_ok=True)
    idx = CACHE / "index_us.csv"
    if not idx.exists():
        rows = [r for r in csv.reader(io.StringIO(_get(INDEX).decode())) if r and r[0] == "UnitedStates"]
        with open(idx, "w", newline="") as f: csv.writer(f).writerows(rows)
    return [r[2] for r in csv.reader(open(idx)) if r[1] == qk]


def footprints(south, west, north, east):
    """Building polygons (lists of lon/lat rings) overlapping the box."""
    qks = {quadkey(la, lo) for la in (south, north) for lo in (west, east)}
    out = []
    for qk in sorted(qks):
        for k, url in enumerate(_tile_urls(qk)):
            fp = CACHE / f"{qk}_{k}.geojsonl.gz"
            if not fp.exists(): fp.write_bytes(_get(url))
            for line in gzip.open(fp, "rt"):
                g = json.loads(line)["geometry"]
                ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
                lo, la = ring[0]
                if south - 0.002 <= la <= north + 0.002 and west - 0.002 <= lo <= east + 0.002: out.append(ring)
    return out


def nearest_m(points, rings, epsg):
    """Metres from each (lat, lon) point to the closest footprint edge (0 = inside); None if no buildings."""
    from pyproj import Transformer
    from shapely.geometry import Polygon, Point
    from shapely.strtree import STRtree
    if not rings: return [None] * len(points)
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    polys = [Polygon([t.transform(lo, la) for lo, la in r]) for r in rings]
    tree = STRtree(polys)
    out = []
    for la, lo in points:
        p = Point(*t.transform(lo, la)); i = tree.nearest(p)
        out.append(round(float(polys[i].distance(p)), 1))
    return out
