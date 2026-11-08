"""Real mound (top row) vs its synthetic twin set into real empty ground (bottom row),
same 220 m extent and lighting; plus random synthetic mounds."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade
from gamd import synth

N = 220; rng = np.random.default_rng(7)
REAL = [("Etowah A", 34.1250, -84.8075, lambda: synth.platform(N, height=19, base_l=120, base_w=110, flank_deg=32, corner_frac=0.3, rot_deg=-8, ramp=True, erosion_m=2.5, rng=rng)),
        ("Kolomoki A", 31.46909, -84.93987, lambda: synth.platform(N, height=18.8, base_l=104, base_w=78, flank_deg=34, corner_frac=0.35, rot_deg=80, erosion_m=2.5, rng=rng)),
        ("Lamar A", 32.81218, -83.59269, lambda: synth.platform(N, height=7.7, base_l=53, base_w=49, flank_deg=30, corner_frac=0.4, rot_deg=10, erosion_m=2, rng=rng)),
        ("Lamar B", 32.81245, -83.59069, lambda: synth.platform(N, height=6.1, base_l=40, base_w=35, flank_deg=29, corner_frac=0.95, rot_deg=0, erosion_m=2, rng=rng)),
        ("Kolomoki conical", 31.46858, -84.94339, lambda: synth.conical(N, height=5.6, core_r=11, apron_r=30, apron_frac=0.3, rng=rng))]
bg, _, _ = fetch_dem(34.6815, -83.7065, N, res_m=1.0); bg = bg.astype("float64")  # empty hayfield, Nacoochee valley
tex = lambda: synth.lidar_texture(bg.shape, amp=0.04, rng=rng)
T = 240; cols = len(REAL); sheet = Image.new("L", (T * cols, T * 3), 255); d = ImageDraw.Draw(sheet)
for k, (name, la, lo, make) in enumerate(REAL):
    z, _, _ = fetch_dem(la, lo, N, res_m=1.0)
    sheet.paste(Image.fromarray(hillshade(z.astype("float64"), 1.0)[0].astype("uint8")).resize((T, T)), (k * T, 0))
    dz, m = make(); zz = bg + dz + tex()
    sheet.paste(Image.fromarray(hillshade(zz, 1.0)[0].astype("uint8")).resize((T, T)), (k * T, T))
    d.text((k * T + 4, 4), "REAL " + name, fill=255); d.text((k * T + 4, T + 4), "FAKE " + name, fill=255)
for k in range(cols):  # row 3: random draws from the calibrated ranges
    dz, m, p = synth.random_mound(N, rng=rng)
    zz = bg + dz + synth.lidar_texture(bg.shape, amp=0.05, furrow_amp=0.04 if p["kind"] == "plowed" else 0, rot_deg=p["rot_deg"], rng=rng)
    im = Image.fromarray(hillshade(zz, 1.0)[0].astype("uint8")).convert("RGB")
    a = np.array(im); edge = m & ~np.pad(m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:], 1); a[edge] = (255, 60, 60)
    sheet.paste(Image.fromarray(a).convert("L").resize((T, T)), (k * T, 2 * T))
    d.text((k * T + 4, 2 * T + 4), f"RANDOM {p['kind']} {p['height']:.1f}m" + (" cut" if p.get("damaged") else ""), fill=255)
sheet.save(sys.argv[1] if len(sys.argv) > 1 else "synth_demo.png")
