"""Real mounds (top) vs 10 fully realistic random fakes (bottom two rows)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade
from gamd import synth
N = 220; rng = np.random.default_rng(21)
real = [("Etowah A", 34.1250, -84.8075), ("Kolomoki A", 31.46909, -84.93987), ("Lamar A", 32.81218, -83.59269),
        ("Lamar B", 32.81245, -83.59069), ("Kolomoki conical", 31.46858, -84.94339)]
bgs = [fetch_dem(la, lo, N, res_m=1.0)[0].astype("float64") for la, lo in
       [(34.6815, -83.7065), (32.8105, -83.5960), (31.4660, -84.9440)]]  # level ground: Nacoochee hayfield, Lamar floodplain, Kolomoki field
tex_src = fetch_dem(32.8105, -83.5960, 600, res_m=1.0)[0].astype("float64")  # real Etowah-area ground for texture
T = 240; sheet = Image.new("L", (T * 5, T * 3), 255); d = ImageDraw.Draw(sheet)
for k, (n, la, lo) in enumerate(real):
    z = fetch_dem(la, lo, N, res_m=1.0)[0].astype("float64")
    sheet.paste(Image.fromarray(hillshade(z, 1.0)[0].astype("uint8")).resize((T, T)), (k * T, 0)); d.text((k * T + 4, 4), "REAL " + n, fill=255)
for k in range(10):
    bg = bgs[k % 3]; dz, m, p = synth.random_mound(N, rng=rng)
    z, m, fx = synth.realistic(bg, dz, m, p, texture_src=tex_src, rng=rng)
    col, row = k % 5, 1 + k // 5
    sheet.paste(Image.fromarray(hillshade(z, 1.0)[0].astype("uint8")).resize((T, T)), (col * T, row * T))
    d.text((col * T + 4, row * T + 4), f"FAKE {p['kind']} {p['height']:.1f}m {'+'.join(fx)}", fill=255)
sheet.save(sys.argv[1])
