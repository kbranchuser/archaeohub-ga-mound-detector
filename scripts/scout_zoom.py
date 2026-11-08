"""Zoomed, numbered view of a scouted park's core (see scout_sites.py).
    python scripts/scout_zoom.py SCOUT_DIR NAME [half_m] [lat lon]"""
import sys, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade
from scout_sites import SITES, GA_SITES, GA_NEW
d = Path(sys.argv[1]); name = sys.argv[2]; half = float(sys.argv[3]) if len(sys.argv) > 3 else 500
site = next(s for s in SITES + GA_SITES + GA_NEW if s[0] == name)
lat, lon = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (site[2], site[3])
res = 1.0
z, tr, epsg = fetch_dem(lat, lon, 2 * half, res_m=res)
from pyproj import Transformer
to = Transformer.from_crs(4326, epsg, always_xy=True); inv = ~tr
hs = hillshade(np.where(np.isfinite(z), z, np.nanmedian(z)), res)[0]
im = Image.fromarray(hs.astype("uint8")).convert("RGB").resize((1000, 1000)); s = 1000 / z.shape[0]; dr = ImageDraw.Draw(im)
try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
except OSError: font = None
for c in csv.DictReader(open(d / "candidates.csv")):
    if c["site"] != name: continue
    col, row = inv * to.transform(float(c["lon"]), float(c["lat"]))
    if not (0 <= row < z.shape[0] and 0 <= col < z.shape[1]): continue
    x, y = col * s, row * s
    dr.ellipse([x - 16, y - 16, x + 16, y + 16], outline=(255, 40, 40), width=3)
    dr.text((x + 18, y - 12), f"{c['n']} ({float(c['height_m']):.0f}m)", fill=(255, 230, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
dr.text((10, 10), f"{name}  {2*half:.0f} m across", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
im.save(d / f"zoom_{name.replace(' ', '_')}.png")
