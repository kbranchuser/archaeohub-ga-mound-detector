"""Search an area of 3DEP LiDAR for compact, mound-like rises and make a contact sheet.
    python scripts/area_search.py NAME LAT LON SIZE_M OUT.png [min_relief_m]"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from pyproj import Transformer
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade

name, lat0, lon0, size, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
thr = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
res = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0
z, tr, epsg = fetch_dem(lat0, lon0, size, res_m=res)
zz = np.nan_to_num(z.astype("float64"), nan=float(np.nanmean(z))); zs = ndi.gaussian_filter(zz, 0.8)
lrm = zs - ndi.uniform_filter(zs, size=41)
gy, gx = np.gradient(zs, res); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
tf = Transformer.from_crs(f"EPSG:{epsg}", 4326, always_xy=True)
lab, _ = ndi.label(lrm > thr); rows = []
for i, sl in enumerate(ndi.find_objects(lab), 1):
    m = lab[sl] == i; a = m.sum() * res * res
    if not 200 <= a <= 12000: continue
    rr, cc = np.nonzero(m); ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([rr, cc])))); el = (ev[1] / max(ev[0], 1e-6)) ** .5
    fill = m.sum() / m.size
    if el > 1.6 or fill < 0.55: continue
    r0 = rr.mean() + sl[0].start; c0 = cc.mean() + sl[1].start
    bg = float(np.median(slope[max(int(r0) - 25, 0):int(r0) + 25, max(int(c0) - 25, 0):int(c0) + 25]))
    if bg > 6: continue
    x, y = tr * (c0, r0); lon, lat = tf.transform(x, y)
    h = float(lrm[sl][m].max())
    rows.append(dict(h=round(h, 1), lat=round(lat, 5), lon=round(lon, 5), area=int(a), el=round(el, 2), px=[int(c0), int(r0)], score=round(h / el * fill, 2)))
rows.sort(key=lambda r: -r["score"])
hs, _ = hillshade(z, res); T, half = 220, 50; top = rows[:15]
sheet = Image.new("RGB", (T * 5, T * 3), "white"); d = ImageDraw.Draw(sheet)
for k, r in enumerate(top):
    x, y = r["px"]; x0, y0 = max(x - half, 0), max(y - half, 0)
    t = Image.fromarray(hs[y0:y0 + 2 * half, x0:x0 + 2 * half].astype("uint8")).resize((T, T))
    sheet.paste(t, ((k % 5) * T, (k // 5) * T)); d.rectangle([(k % 5) * T, (k // 5) * T, (k % 5) * T + 150, (k // 5) * T + 14], fill=(0, 0, 0))
    d.text(((k % 5) * T + 3, (k // 5) * T + 2), f"{name[0]}{k+1} {r['h']}m {r['area']}m2", fill=(255, 255, 0))
sheet.save(out)
print(name, f"{len(rows)} candidates;", json.dumps([{k: r[k] for k in ('h', 'lat', 'lon', 'area')} for r in top[:10]]))
