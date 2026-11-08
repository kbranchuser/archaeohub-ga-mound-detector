"""Measure verified mounds from 1 m 3DEP: height, footprint, long/short axis,
flat-top fraction, flank steepness, roundness. Writes data/mound_measurements.csv
and a contact sheet with each measured outline drawn on the hillshade.

Ground level = morphological opening (top-hat), so bluffs and broad landforms
stay ground; the mound = connected area containing the summit that rises
> max(0.3 m, 15% of its height) above that ground."""
import sys, csv, math
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade

ROOT = Path(__file__).resolve().parents[1]
FRAC = 0.15  # outline = where the mound rises above 15% of its own height (separates touching mounds)
rows = [r for r in csv.DictReader(open(ROOT / "data/ga_mounds_verified.csv")) if r["label"] == "mound"]
out, tiles = [], []
for r in rows:
    lat, lon = float(r["lat"]), float(r["lon"])
    z, tr, _ = fetch_dem(lat, lon, 320, res_m=1.0); z = z.astype("float64")
    H, W = z.shape; c = H // 2
    win = z[c - 25:c + 25, c - 25:c + 25]; py, px = np.unravel_index(np.nanargmax(win), win.shape); py += c - 25; px += c - 25
    yy, xx = np.mgrid[0:H, 0:W]; d = np.hypot(yy - py, xx - px)
    # Ground = morphological opening with a 141 m flat window (separable min then max):
    # removes anything narrower than the window (mounds) but keeps wide landforms
    # such as the bluff Ocmulgee's Great Temple sits on. rel = top-hat.
    zf = np.where(np.isfinite(z), z, np.nanmin(z))
    ground = ndi.maximum_filter(ndi.minimum_filter(zf, size=141), size=141)
    rel = zf - ground
    h = float(rel[py, px]); thr = max(0.3, FRAC * h)
    lab, _ = ndi.label(ndi.binary_opening(rel > thr, iterations=2)); m = lab == lab[py, px]
    m = ndi.binary_fill_holes(m) & (d < 110)
    ys, xs = np.nonzero(m); area = m.sum()
    ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    proj = np.vstack([xs - xs.mean(), ys - ys.mean()]).T @ evec
    long_ax, short_ax = np.ptp(proj[:, 1]) + 1, np.ptp(proj[:, 0]) + 1
    gy, gx = np.gradient(ndi.gaussian_filter(z, 1.0)); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    top = m & (rel > 0.85 * h) & (slope < 6)
    flank = m & (rel > 0.2 * h) & (rel < 0.8 * h)
    circ = 4 * math.pi * area / max(ndi.binary_dilation(m).sum() - area, 1) ** 2  # area / perimeter^2
    rect_fill = area / (long_ax * short_ax)   # ~0.785 ellipse/circle, ~1.0 rectangle
    out.append(dict(site=r["site"], feature=r["feature"], height_m=round(h, 1), long_m=int(long_ax), short_m=int(short_ax),
                    footprint_m2=int(area), top_frac=round(top.sum() / area, 2), top_m2=int(top.sum()),
                    flank_slope_deg=round(float(np.median(slope[flank])), 1) if flank.any() else None,
                    elong=round(long_ax / short_ax, 2), rect_fill=round(rect_fill, 2), condition=r["condition"]))
    hs, _ = hillshade(z, 1.0); im = Image.fromarray(hs.astype("uint8")).convert("RGB"); dr = ImageDraw.Draw(im)
    edge = m & ~ndi.binary_erosion(m); im_a = np.array(im); im_a[edge] = (255, 40, 40); im_a[top & ~ndi.binary_erosion(top)] = (40, 200, 255)
    t = Image.fromarray(im_a).crop((c - 110, c - 110, c + 110, c + 110)).resize((260, 260)); ImageDraw.Draw(t).text((4, 4), r["feature"][:30], fill=(255, 255, 0))
    tiles.append(t)
with open(ROOT / "data/mound_measurements.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
for o in out: print(f"{o['feature'][:34]:34s} h={o['height_m']:5.1f}m  {o['long_m']}x{o['short_m']}m  top={o['top_frac']:.2f} ({o['top_m2']} m2)  flank={o['flank_slope_deg']}°  elong={o['elong']}  rectfill={o['rect_fill']}")
sheet = Image.new("RGB", (260 * 4, 260 * 2), "white")
for k, t in enumerate(tiles): sheet.paste(t, ((k % 4) * 260, (k // 4) * 260))
sheet.save(sys.argv[1] if len(sys.argv) > 1 else "measure.png")
