"""Look at published mound locations in 3DEP, one tight view per point.

    python scripts/check_candidates.py CANDIDATES_CSV OUT_DIR [size_m]

scout_sites.py searches a 2.4 km box for any compact rise, which drowns real
mounds in dissected terrain (Singer-Moye). Here the location is already
published, so each view is small (default 400 m), centred on the point with a
crosshair, and reports how high the ground within 30 m of it stands above its
surroundings. A person decides
from the picture; nothing is labelled automatically.

Reads columns site, feature, lat, lon (blank lat/lon rows are skipped).
Writes OUT_DIR/<n>_<site>.png (hillshade | local relief) and OUT_DIR/checked.csv.
"""
import sys, csv, math, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamd.dem import fetch_dem, hillshade
from gamd.bands import local_relief
from scout_sites import detail

RES = 1.0
PEAK_R = 30   # metres around the published point searched for its highest relief


if __name__ == "__main__":
    from pyproj import Transformer
    src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    size = float(sys.argv[3]) if len(sys.argv) > 3 else 400
    rows = [r for r in csv.DictReader(open(src)) if (r.get("lat") or "").strip() and (r.get("lon") or "").strip()]
    results = []
    for i, r in enumerate(rows, 1):
        lat, lon = float(r["lat"]), float(r["lon"])
        name = f"{i:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', (r['site'] + ' ' + (r.get('feature') or '')).strip())[:50]}"
        try:
            # Fetch twice the view: the 141 m top-hat behind local_relief needs
            # context past the edges, or a whole floodplain reads as one "rise".
            zbig, tr, epsg = fetch_dem(lat, lon, size * 2, res_m=RES)
        except Exception as e:  # noqa: BLE001 — log and keep going
            print(f"{name}: fetch failed: {e}", flush=True); continue
        zb = np.where(np.isfinite(zbig), zbig, np.nanmedian(zbig))
        q = zb.shape[0] // 4
        zf = zb[q:q + int(size / RES), q:q + int(size / RES)]
        c0 = zf.shape[1] / 2; r0 = zf.shape[0] / 2
        # How far the ground within PEAK_R of the published point stands above
        # its surroundings (local relief): a quick hint, the picture decides.
        rel = local_relief(zb, RES)[q:q + zf.shape[0], q:q + zf.shape[1]]
        yy, xx = np.ogrid[:zf.shape[0], :zf.shape[1]]
        disc = (yy - r0) ** 2 + (xx - c0) ** 2 <= (PEAK_R / RES) ** 2
        peak = float(rel[disc].max())
        py, px = np.unravel_index(np.where(disc, rel, -1e9).argmax(), rel.shape)
        hs = Image.fromarray(hillshade(zf, RES)[0].astype("uint8")).convert("RGB")
        rl = Image.fromarray((np.clip((rel + 2) / 6, 0, 1) * 255).astype("uint8")).convert("RGB")
        for im in (hs, rl):
            d = ImageDraw.Draw(im)
            d.line([c0 - 12, r0, c0 + 12, r0], fill=(0, 200, 255), width=2)
            d.line([c0, r0 - 12, c0, r0 + 12], fill=(0, 200, 255), width=2)
            d.ellipse([c0 - PEAK_R, r0 - PEAK_R, c0 + PEAK_R, r0 + PEAK_R], outline=(0, 200, 255), width=1)
        both = Image.new("RGB", (hs.width * 2 + 6, hs.height), (0, 0, 0))
        both.paste(hs, (0, 0)); both.paste(rl, (hs.width + 6, 0))
        ImageDraw.Draw(both).text((6, 4), f"{r['site']} | {r.get('feature') or ''} | {int(size)} m", fill=(255, 255, 0))
        both.save(out / f"{name}.png")
        results.append(dict(n=i, site=r["site"], feature=r.get("feature", ""), lat=lat, lon=lon,
                            detail=round(detail(zf), 3), peak_relief_m=round(peak, 1),
                            peak_offset_m=round(math.hypot(px - c0, py - r0) * RES), image=f"{name}.png"))
        print(f"{name:55s} detail {results[-1]['detail']:.3f}  peak relief {peak:5.1f} m  ({results[-1]['peak_offset_m']} m from point)", flush=True)
    if results:
        with open(out / "checked.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
