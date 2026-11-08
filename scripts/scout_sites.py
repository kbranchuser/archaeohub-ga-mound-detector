"""Scout out-of-state mound parks for training positives (Georgia stays the test).

    python scripts/scout_sites.py OUT_DIR [size_m]

For each park in SITES (published park location, public), fetch 3DEP around it,
find compact rises above a 141 m top-hat ground (same rule as the Georgia
measurements), and write OUT_DIR/<site>.png (hillshade, numbered candidates) and
OUT_DIR/candidates.csv. A person then marks which numbers are mounds
(data/out_of_state_mounds.csv); nothing is labelled automatically. Also reports
the DEM's effective detail, since some areas only have 10 m elevation.
"""
import sys, csv, math
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade
from gamd.bands import local_relief

# name, state, lat, lon, published mound count / note (approximate park centres)
SITES = [
    ("Moundville", "AL", 33.0055, -87.6305, "~29 mounds; Mound B ~17.7 m"),
    ("Winterville", "MS", 33.4833, -91.0614, "~12+ mounds; Mound A ~16.8 m"),
    ("Emerald", "MS", 31.6350, -91.2463, "huge platform + 2 secondary mounds"),
    ("Toltec", "AR", 34.6469, -92.0655, "Plum Bayou; Mound A ~15 m, Mound B ~11.5 m"),
    ("Parkin", "AR", 35.2766, -90.5560, "one large platform mound by the St. Francis R."),
    ("Pinson", "TN", 35.4979, -88.6828, "Sauls Mound ~22 m + others (Middle Woodland)"),
    ("Shiloh", "TN", 35.1400, -88.3190, "7 platform mounds on a bluff over the Tennessee R."),
    ("Chucalissa", "TN", 35.0628, -90.1300, "platform mound on a Mississippi R. bluff"),
    ("Lake Jackson", "FL", 30.4990, -84.3127, "platform mounds; Mound 2 ~11 m"),
    ("Letchworth-Love", "FL", 30.5195, -83.9983, "large Woodland platform ~14.6 m"),
    ("Kincaid", "IL", 37.0817, -88.4930, "many platform mounds on the Ohio R. floodplain"),
    ("Angel", "IN", 37.9435, -87.4539, "11 mounds; Mound A ~13 m"),
]


# Georgia parks already in the test set: scouted for their OTHER mounds (more test positives)
GA_SITES = [
    ("Etowah", "GA", 34.1250, -84.8075, "6 mounds A-F; B and C large platforms near A"),
    ("Kolomoki", "GA", 31.4690, -84.9399, "Mound A + B-H (burial, conical, low platforms)"),
    ("Ocmulgee", "GA", 32.8390, -83.6070, "Great Temple, Lesser Temple, Funeral, McDougal, Southeast, Cornfield, Earth Lodge"),
]
# New Georgia test sites (published locations; never DINAA)
GA_NEW = [
    ("Singer-Moye", "GA", 32.07846, -84.8352, "8 mounds, ~1-14 m (New Georgia Encyclopedia); UGA-owned"),
    ("Roods Landing", "GA", 32.02445, -85.04444, "8 platform mounds, Mound A 7.6 m, 44x38 m (Wikipedia/NRHP)"),
    ("Scull Shoals", "GA", 33.7290, -83.2890, "2 mounds in Oconee River bottoms, ~1 m alluvium over site (Oconee NF)"),
]


def candidates(z, res, min_h=1.5):
    zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    rel = local_relief(ndi.gaussian_filter(zf, 0.7), res)
    lab, n = ndi.label(ndi.binary_opening(rel > min_h, iterations=2))
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == k; area = m.sum() * res * res
        if not (150 <= area <= 40000): continue
        r = rel[sl]; h = float(r[m].max())
        ys, xs = np.nonzero(m)
        fill = m.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))     # compact blobs fill their box
        if fill < 0.45: continue
        py, px = np.unravel_index(np.where(m, r, -1).argmax(), m.shape)
        out.append(dict(row=int(py + sl[0].start), col=int(px + sl[1].start), height_m=round(h, 1), area_m2=int(area), fill=round(float(fill), 2)))
    return sorted(out, key=lambda c: -c["height_m"])


def detail(z):
    """Fraction of 1-px curvature that survives a 3 px blur: ~0 for interpolated 10 m data."""
    zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    lap = ndi.laplace(zf); return float(np.std(lap - ndi.uniform_filter(lap, 3)))


if __name__ == "__main__":
    from pyproj import Transformer
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    size = float(sys.argv[2]) if len(sys.argv) > 2 else 2400
    if len(sys.argv) > 3 and sys.argv[3] == "ga": SITES[:] = GA_SITES
    if len(sys.argv) > 3 and sys.argv[3] == "ganew": SITES[:] = GA_NEW
    res = 1.524; rows = []
    for name, st, lat, lon, note in SITES:
        z, tr, epsg = fetch_dem(lat, lon, size, res_m=res)
        cs = candidates(z, res)[:40]
        to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
        hs = hillshade(np.where(np.isfinite(z), z, np.nanmedian(z)), res)[0]
        im = Image.fromarray(hs.astype("uint8")).convert("RGB"); d = ImageDraw.Draw(im)
        for i, c in enumerate(cs, 1):
            x, y = tr * (c["col"] + 0.5, c["row"] + 0.5); c["lon"], c["lat"] = (round(v, 5) for v in to_ll.transform(x, y))
            d.ellipse([c["col"] - 14, c["row"] - 14, c["col"] + 14, c["row"] + 14], outline=(255, 40, 40), width=3)
            d.text((c["col"] + 16, c["row"] - 8), str(i), fill=(255, 255, 0))
            rows.append(dict(site=name, state=st, n=i, **c))
        im.resize((1000, 1000)).save(out / f"{name.replace(' ', '_')}.png")
        print(f"{name:16s} {st}  detail {detail(z):.3f}  candidates {len(cs):2d}  top heights {[c['height_m'] for c in cs[:6]]}  ({note})", flush=True)
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
