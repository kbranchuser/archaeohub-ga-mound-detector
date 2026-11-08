"""One reading: scan a square of USGS 3DEP LiDAR with the mound model(s).

`run_reading(job, models, names, dev, progress)` fetches 1 m elevation for the square
(plus a context border), computes the relief6 layers, runs every model, merges flags,
measures each spot, tags it with land cover and nearest building, and returns flags,
close-ups and relief views as base64 PNGs. Used by scripts/detect.py, area_scan.py and
the Archaeohub helper.
"""
import base64, glob, io, math, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from .dem import hillshade

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD, MERGE_M = 0.5, 30


# --- images ------------------------------------------------------------------
def png_b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def relief_rgb(hs, rel, lim=1.5):
    """Hillshade tinted by local relief: raised = warm, cut = cool, flat = grey."""
    t = np.clip(rel / lim, -1, 1)[..., None]
    warm, cool = np.array([224, 112, 79], float), np.array([90, 140, 210], float)
    tint = np.where(t > 0, warm, cool); a = np.abs(t) * 0.75
    g = np.repeat((hs / 255.0)[..., None], 3, 2) * 255
    return np.clip(g * (1 - a) + (g * 0.35 + tint * 0.65) * a, 0, 255).astype("uint8")


def ramp(v, stops):
    """Colour a 0-255 uint8 band through [(pos 0-1, (r,g,b)), ...] stops."""
    x = v.astype(float) / 255.0
    pos = np.array([p for p, _ in stops]); cols = np.array([c for _, c in stops], float)
    return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], -1).astype("uint8")


def stretch(a, lo=2, hi=98):
    """Contrast-stretch a grey band to its 2–98 % range."""
    a = a.astype(float); l, h = np.percentile(a, [lo, hi])
    return (np.clip((a - l) / max(h - l, 1e-6), 0, 1) * 255).astype("uint8")


HEAT = [(0, (20, 18, 24)), (0.35, (110, 40, 90)), (0.7, (225, 110, 60)), (1, (250, 235, 150))]
DIVERGE = [(0, (60, 110, 200)), (0.5, (225, 222, 215)), (1, (215, 85, 55))]
HYPSO = [(0, (40, 90, 60)), (0.35, (130, 160, 90)), (0.65, (205, 180, 120)), (1, (245, 240, 230))]


def band_views(bands, zf, hs, side):
    """Every relief layer the model reads (relief6), plus two context views, as map images."""
    g = lambda a: resize(np.ascontiguousarray(a), side)
    lrm, opos, slope, oneg, geo, svf = bands[:6]
    multi = np.stack([hillshade(zf, 1.0, az=a)[0] for a in (315, 45, 180)], -1).astype("uint8")
    zr = zf - np.nanmin(zf); elev = (255 * zr / max(float(np.nanmax(zr)), 1e-6)).astype("uint8")
    elev_rgb = (ramp(elev, HYPSO).astype(float) * (0.55 + 0.45 * hs[..., None] / 255)).astype("uint8")
    v = [("view_height.png", "Height above ground", ramp(lrm, HEAT)),
         ("view_openness_pos.png", "Openness ＋ (raised = bright)", stretch(opos)),
         ("view_openness_neg.png", "Openness − (cuts = dark)", stretch(255 - oneg)),
         ("view_slope.png", "Slope (steep = dark)", stretch(255 - slope)),
         ("view_landform.png", "Landform (raised = warm, hollow = cool)", ramp(geo, DIVERGE)),
         ("view_skyview.png", "Sky view (directional)", stretch(svf))]
    out = [dict(name=n, label=l, kind="view", png=png_b64(g(a))) for n, l, a in v]
    out.append(dict(name="view_multihillshade.png", label="Multi-direction hillshade", kind="view", png=png_b64(resize(multi, side))))
    out.append(dict(name="view_elevation.png", label="Elevation (colour)", kind="view", png=png_b64(resize(elev_rgb, side))))
    return out


def resize(arr, n):
    return np.asarray(Image.fromarray(arr).resize((n, n), Image.LANCZOS))


# --- one reading -------------------------------------------------------------
NLCD_NAMES = {11: "open water", 21: "developed, open space (lawns, parks)", 22: "developed, low intensity", 23: "developed, medium intensity",
              24: "developed, high intensity", 31: "bare ground", 41: "deciduous forest", 42: "evergreen forest", 43: "mixed forest",
              52: "shrubland", 71: "grassland", 81: "pasture / hay", 82: "cropland", 90: "woody wetland", 95: "herbaceous wetland"}


def add_context_tags(flags, lat, lon, full_m, epsg, pix):
    """Tag each spot with its land cover (NLCD 2021) and distance to the nearest building."""
    try:
        from .landcover import fetch_nlcd
        nl = fetch_nlcd(lat, lon, full_m, 1.0)
        for f, (py, px) in zip(flags, pix):
            c = int(nl[min(py, nl.shape[0] - 1), min(px, nl.shape[1] - 1)]); f["landcover"] = NLCD_NAMES.get(c, f"class {c}")
    except Exception as e:
        print("  land cover unavailable:", e, flush=True)
    try:
        from .buildings import footprints, nearest_m
        la = [f["lat"] for f in flags]; lo = [f["lon"] for f in flags]
        if flags:
            rings = footprints(min(la), min(lo), max(la), max(lo))
            for f, d in zip(flags, nearest_m([(f["lat"], f["lon"]) for f in flags], rings, epsg)): f["building_m"] = d
    except Exception as e:
        print("  building footprints unavailable:", e, flush=True)


MARGIN_M = 120   # extra ground around every square: relief layers need ~115 m of context, or the edges distort


def run_reading(job, models, names, dev, progress):
    """Scan one square. Elevation is fetched with a MARGIN_M border so relief layers are
    right up to the edge; flags in the border are dropped and images cover the square only."""
    from .dem import fetch_dem_tiled, DAVIS_RES_M
    from .bands import make_bands, local_relief
    from .scan import scan
    from pyproj import Transformer
    t0 = time.time()
    lat, lon, size = float(job["lat"]), float(job["lon"]), float(job["size_m"])
    progress(5, "Downloading USGS 3DEP elevation")
    z, tr, epsg = fetch_dem_tiled(lat, lon, size + 2 * MARGIN_M, res_m=1.0)
    if np.isnan(z).mean() > 0.5: raise RuntimeError("No 1 m LiDAR elevation covers most of this square.")
    zf = np.where(np.isfinite(z), z, np.nanmedian(z)).astype("float64")
    N = zf.shape[0]; M = MARGIN_M; n = N - 2 * M                      # 1 m grid: full, border, inner square
    progress(25, "Computing relief layers")
    G = int(round(N / DAVIS_RES_M)); Mp = M / DAVIS_RES_M; g = int(round(n / DAVIS_RES_M)); m0 = int(round(Mp))
    bands_full = make_bands(ndi.zoom(zf, G / N, order=1, mode="nearest", prefilter=False)[:G, :G], DAVIS_RES_M, "relief6")
    hits = []
    for mi, m in enumerate(models):
        progress(35 + 45 * mi / len(models), f"Running model {mi + 1} of {len(models)}")
        for cy, cx, sc, w in scan(m, dev, bands_full):
            if sc >= THRESHOLD and Mp <= cy < G - Mp and Mp <= cx < G - Mp:   # inside the square only
                hits.append((cy, cx, sc, mi, w))
    groups = []
    for h in sorted(hits, key=lambda h: -h[2]):
        gg = next((gr for gr in groups if math.hypot(h[0] - gr[0][0], h[1] - gr[0][1]) * DAVIS_RES_M < MERGE_M), None)
        (gg.append(h) if gg else groups.append([h]))

    progress(85, "Measuring spots and making images")
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    zs = ndi.gaussian_filter(zf, 0.7)
    rel = local_relief(zs, 1.0)                          # height above local ground (for measuring)
    slrm = zs - ndi.gaussian_filter(zs, 15)              # signed local relief (for pictures: rises warm, cuts cool)
    hs = hillshade(zf, 1.0)[0]
    flags, pix = [], []
    for k, gr in enumerate(groups, 1):
        cy, cx = np.mean([h[0] for h in gr]), np.mean([h[1] for h in gr])
        best = max(gr, key=lambda h: h[2])
        x, y = tr * (cx * DAVIS_RES_M, cy * DAVIS_RES_M); flon, flat = to_ll.transform(x, y)
        py, px = int(round(cy * DAVIS_RES_M)), int(round(cx * DAVIS_RES_M))   # 1 m grid (full)
        pix.append((py, px))
        r = max(8, int(best[4] / 2))
        win = rel[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
        h = int(min(max(60, best[4] * 0.9), N // 2)); y0, x0 = min(max(py - h, 0), N - 2 * h), min(max(px - h, 0), N - 2 * h)
        chip = resize(relief_rgb(hs[y0:y0 + 2 * h, x0:x0 + 2 * h], slrm[y0:y0 + 2 * h, x0:x0 + 2 * h]), 240)
        im = Image.fromarray(chip); d = ImageDraw.Draw(im); s = 240 / (2 * h)
        cxp, cyp, rr = (px - x0) * s, (py - y0) * s, max(10, best[4] / 2 * s * 1.2)
        d.ellipse([cxp - rr, cyp - rr, cxp + rr, cyp + rr], outline=(255, 255, 255), width=2)
        bar = 50 if h <= 90 else 100
        d.line([8, 232, 8 + bar * s, 232], fill=(255, 255, 255), width=3); d.text((10, 216), f"{bar} m", fill=(255, 255, 255))
        flags.append(dict(n=k, lat=round(flat, 7), lon=round(flon, 7), score=round(float(best[2]), 3),
                          models_agree=len({h[3] for h in gr}), models_total=len(models),
                          height_m=round(float(np.nanmax(win)), 2) if win.size else None, width_m=round(float(best[4]), 1),
                          fx=round((float(cx) - Mp) / g, 4), fy=round((float(cy) - Mp) / g, 4), chip_png=png_b64(np.asarray(im))))
    progress(90, "Checking land cover and nearby buildings")
    add_context_tags(flags, lat, lon, size + 2 * MARGIN_M, epsg, pix)
    corners = [to_ll.transform(*(tr * c)) for c in [(M, M), (M + n, M), (M + n, M + n), (M, M + n)]]   # NW, NE, SE, SW of the square
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    side = min(n, 1000)
    inner = (slice(M, M + n), slice(M, M + n))
    zf_i, hs_i, slrm_i = zf[inner], hs[inner], slrm[inner]
    bands = bands_full[:, m0:m0 + g, m0:m0 + g]
    images = [
        dict(name="overview_relief.png", label="Local relief (raised = warm, cut = cool)", kind="relief", png=png_b64(resize(relief_rgb(hs_i, slrm_i), side))),
        dict(name="overview_hillshade.png", label="Hillshade", kind="view", png=png_b64(resize(hs_i.astype("uint8"), side))),
        dict(name="overview_model.png", label="What the model sees", kind="view",
             png=png_b64(resize(np.ascontiguousarray(bands[:3].transpose(1, 2, 0)), side))),
    ] + band_views(bands, zf_i, hs_i, side)
    return dict(
        model=dict(name=", ".join(names) if len(names) <= 2 else f"{len(names)}-model ensemble", files=names, threshold=THRESHOLD, inputs="relief6", edge_margin_m=MARGIN_M),
        bounds=[min(lons), min(lats), max(lons), max(lats)],
        corners=[[round(a, 7), round(b, 7)] for a, b in corners],
        stats=dict(runtime_s=round(time.time() - t0, 1), area_km2=round((size / 1000) ** 2, 2), dem="USGS 3DEP 1 m", device=dev),
        images=images, flags=flags,
    )


def default_ckpts():
    final = ROOT / "runs/final/all/ckpt_final.pt"
    if final.exists(): return [str(final)]
    return sorted(glob.glob(str(ROOT / "runs/mined6/fold*/ckpt_final.pt")))
