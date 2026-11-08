"""Synthetic ("artificial") mounds for training data, shaped from measured Georgia mounds.

Each generator returns (dz, mask): a height-change raster (metres, 1 px = `res` m)
to ADD to a real DEM, and the boolean footprint used as the training label.

Calibration (scripts/measure_mounds.py, 1 m 3DEP):
  platform: flanks ~29-34 deg regardless of size; height 6-19 m; base 35-120 m;
            squarish with rounded corners. A truncated pyramid with a constant
            flank angle reproduces the measured flat-top share (Etowah A 22% vs
            23% measured, Kolomoki A 13% vs 13%, Lamar A 22% vs 20%).
  conical:  steep core cone on a broad gentle apron (Kolomoki conical: 5.6 m,
            63 x 56 m incl. apron).
  plowed:   low broad rises 0.6-1.5 m over 15-45 m (Romanian burial-mound study,
            Niculiță 2020) - matches our unconfirmed 'possible' mounds.
"""
import math
import numpy as np
from scipy import ndimage as ndi


def _grid(size_px, res, cx=0.0, cy=0.0, rot_deg=0.0):
    """Ground coordinates (m) of each pixel, rotated so the mound's long axis is x."""
    c = (np.arange(size_px) - (size_px - 1) / 2) * res
    xx, yy = np.meshgrid(c, c)
    xx, yy = xx - cx, yy - cy
    t = math.radians(rot_deg)
    return xx * math.cos(t) + yy * math.sin(t), -xx * math.sin(t) + yy * math.cos(t)


def _rounded_rect_inside(x, y, half_l, half_w, corner_r):
    """Distance INSIDE a rounded rectangle (positive inside, negative outside), metres."""
    qx, qy = np.abs(x) - (half_l - corner_r), np.abs(y) - (half_w - corner_r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - corner_r
    return -outside


def platform(size_px, res=1.0, height=10.0, base_l=70.0, base_w=60.0, flank_deg=31.0,
             corner_frac=0.25, rot_deg=0.0, ramp=False, ramp_width=12.0, erosion_m=2.0, wobble_m=1.0, rng=None):
    """Flat-topped Mississippian platform mound (truncated pyramid, rounded corners, optional ramp)."""
    rng = rng or np.random.default_rng()
    x, y = _grid(size_px, res, rot_deg=rot_deg)
    corner = corner_frac * min(base_l, base_w) / 2
    inside = _rounded_rect_inside(x, y, base_l / 2, base_w / 2, corner)
    wob = ndi.gaussian_filter(rng.normal(0, 1, inside.shape), 14 / res)        # outline wobble (NMC-style)
    inside = inside + wob / (wob.std() + 1e-9) * wobble_m
    dz = np.clip(inside * math.tan(math.radians(flank_deg)), 0, height)
    slump = ndi.gaussian_filter(rng.normal(0, 1, inside.shape), 3 / res)     # uneven, slumped flanks
    dz = dz + slump / (slump.std() + 1e-9) * 0.02 * height * (dz > 0) * (dz < height)
    if ramp:  # causeway down one long side at ~1/3 of the flank angle, from the summit edge outward
        side = float(rng.choice([-1, 1]))
        top_edge = base_w / 2 - height / math.tan(math.radians(flank_deg))   # where the flat top ends
        run = height / math.tan(math.radians(flank_deg * 0.6))   # Etowah-like: short, ~18-20 deg
        along = side * y - top_edge                                       # 0 at the summit edge, + outward
        rz = height * (1 - along / run)
        on = (np.abs(x) < ramp_width / 2) & (along > 0) & (along < run)
        dz = np.where(on, np.maximum(dz, rz), dz)
    if erosion_m > 0:
        dz = ndi.gaussian_filter(dz, erosion_m / res)
    return dz, dz > max(0.3, 0.15 * height)


def conical(size_px, res=1.0, height=5.5, core_r=12.0, apron_r=30.0, apron_frac=0.3,
            elong=1.1, rot_deg=0.0, erosion_m=2.5, rng=None):
    """Conical/burial mound: steep core cone on a broad gentle apron."""
    rng = rng or np.random.default_rng()
    x, y = _grid(size_px, res, rot_deg=rot_deg)
    r = np.hypot(x / elong, y)
    w = ndi.gaussian_filter(rng.normal(0, 1, r.shape), 5 / res); r = r + w / (w.std() + 1e-9) * 1.0
    core = np.clip(1 - r / core_r, 0, None) ** 1.2 * (1 - apron_frac) * height
    apron = np.clip(1 - r / apron_r, 0, None) ** 1.6 * apron_frac * height / (1 - core_r / apron_r) ** 1.6
    dz = np.maximum(core + np.minimum(apron, apron_frac * height), apron)
    dz = ndi.gaussian_filter(dz, erosion_m / res)
    dz = dz * (height / max(dz.max(), 1e-6))            # smoothing lowers the peak; restore target height
    return dz, dz > max(0.3, 0.15 * height)


def plowed(size_px, res=1.0, height=1.0, diam=30.0, plow_stretch=1.3, rot_deg=0.0, rng=None):
    """Low, plow-spread mound: a broad smooth rise, smeared along the plow direction."""
    x, y = _grid(size_px, res, rot_deg=rot_deg)
    r = np.hypot(x / plow_stretch, y) / (diam / 2)
    dz = height * np.exp(-2.2 * r ** 2)
    return dz, dz > max(0.2, 0.25 * height)


def damage_cut(dz, mask, res=1.0, width_m=15.0, depth_frac=1.0, rot_deg=None, offset_m=None, rng=None):
    """Cut a straight corridor (road/railroad) through a mound, like Ocmulgee's Lesser Temple."""
    rng = rng or np.random.default_rng()
    n = dz.shape[0]; rot = rng.uniform(0, 180) if rot_deg is None else rot_deg
    off = rng.uniform(-0.3, 0.3) * n * res if offset_m is None else offset_m
    x, y = _grid(n, res, rot_deg=rot)
    dist = np.abs(y - off) - width_m / 2                 # <0 inside the corridor floor
    keep = np.clip(dist / (1.5 * max(dz.max(), 1.0)) + 0.0, 0, 1)   # 0 on floor -> 1 beyond the side slope
    floor = dz * (1 - depth_frac)
    dz = floor + (dz - floor) * keep
    return dz, mask & (dist > 0)


def lidar_texture(shape, res=1.0, amp=0.06, furrow_amp=0.0, furrow_m=1.5, rot_deg=0.0, rng=None):
    """Fine bare-earth roughness, plus optional plow furrows."""
    rng = rng or np.random.default_rng()
    t = ndi.gaussian_filter(rng.normal(0, 1, shape), 1.2 / res)
    t = t / (t.std() + 1e-9) * amp
    if furrow_amp > 0:
        x, _ = _grid(shape[0], res, rot_deg=rot_deg)
        t = t + furrow_amp * np.sin(2 * math.pi * x / furrow_m)
    return t


def random_mound(size_px, res=1.0, rng=None):
    """One mound drawn from the calibrated ranges. Returns (dz, mask, params)."""
    rng = rng or np.random.default_rng()
    kind = rng.choice(["platform", "conical", "plowed"], p=[0.45, 0.25, 0.30])
    rot = float(rng.uniform(0, 180))
    if kind == "platform":
        h = float(rng.uniform(3, 19)); L = float(rng.uniform(max(30, 4 * h), max(45, 7 * h)))
        p = dict(height=h, base_l=L, base_w=float(L * rng.uniform(0.75, 1.0)), flank_deg=float(rng.uniform(26, 35)),
                 corner_frac=float(rng.uniform(0.15, 0.6)), rot_deg=rot, ramp=bool(rng.random() < 0.3), ramp_width=float(L * rng.uniform(0.2, 0.35)),
                 erosion_m=float(rng.uniform(1.5, 3.5)), wobble_m=float(rng.uniform(0.5, 2)))
        dz, m = platform(size_px, res, rng=rng, **p)
    elif kind == "conical":
        h = float(rng.uniform(2, 7)); core = float(rng.uniform(8, 15))
        p = dict(height=h, core_r=core, apron_r=float(core * rng.uniform(1.8, 2.8)), apron_frac=float(rng.uniform(0.15, 0.4)),
                 elong=float(rng.uniform(1.0, 1.25)), rot_deg=rot, erosion_m=float(rng.uniform(2, 4)))
        dz, m = conical(size_px, res, rng=rng, **p)
    else:
        p = dict(height=float(rng.uniform(0.8, 1.5)), diam=float(rng.uniform(15, 45)), plow_stretch=float(rng.uniform(1.0, 1.5)), rot_deg=rot)
        dz, m = plowed(size_px, res, rng=rng, **p)
    if kind != "plowed" and rng.random() < 0.15:
        dz, m = damage_cut(dz, m, res, width_m=float(rng.uniform(8, 20)), depth_frac=float(rng.uniform(0.6, 1.0)), rng=rng)
        p["damaged"] = True
    p["kind"] = kind
    return dz, m, p


# ---------------------------------------------------------------- realism (6.3)
def forest_sparse(z, keep_frac=0.15, cell_m=1.0, rng=None):
    """Mimic thin ground returns under canopy: keep a random fraction of cells and
    re-interpolate linearly (TIN), which leaves the faceted look of forested LiDAR."""
    from scipy.interpolate import LinearNDInterpolator
    rng = rng or np.random.default_rng()
    H, W = z.shape; keep = rng.random((H, W)) < keep_frac
    keep[[0, -1], :] = True; keep[:, [0, -1]] = True
    ys, xs = np.nonzero(keep)
    f = LinearNDInterpolator(np.c_[xs, ys], z[ys, xs])
    yy, xx = np.mgrid[0:H, 0:W]
    return f(xx, yy)


def summit_path(dz, res=1.0, width_m=2.0, depth_m=0.4, rng=None):
    """A worn footpath/stair line running up a flank onto the summit."""
    rng = rng or np.random.default_rng()
    n = dz.shape[0]; ang = rng.uniform(0, 180)
    x, y = _grid(n, res, rot_deg=ang)
    on = (np.abs(y) < width_m / 2) & (x > 0) & (dz > 0.2)
    return np.where(on, np.maximum(dz - depth_m, 0), dz)


def real_microrelief(real_dem, size_px, rng=None, sigma_px=6):
    """Fine texture (DEM minus its smoothed self) cut from a real LiDAR patch."""
    rng = rng or np.random.default_rng()
    H, W = real_dem.shape
    y0, x0 = rng.integers(0, H - size_px), rng.integers(0, W - size_px)
    p = real_dem[y0:y0 + size_px, x0:x0 + size_px].astype("float64")
    r = p - ndi.gaussian_filter(p, sigma_px)
    return np.clip(r, -0.2, 0.2) * 0.6                   # keep fine texture, drop banks/ditches


def realistic_parts(shape, dz, mask, params, res=1.0, texture_src=None, rng=None):
    """Per-mound realism without forest: returns (dz, mask, tex, effects), where dz
    carries any path/cut and tex is the surface texture to add over the window."""
    rng = rng or np.random.default_rng(); fx = []
    kind = params.get("kind")
    if kind == "platform" and rng.random() < 0.35:
        dz = summit_path(dz, res, rng=rng); fx.append("path")
    if kind not in (None, "plowed") and not params.get("damaged") and rng.random() < 0.15:
        dz, mask = damage_cut(dz, mask, res, width_m=float(rng.uniform(8, 20)), depth_frac=float(rng.uniform(0.6, 1.0)), rng=rng); fx.append("cut")
    plowed_field = kind == "plowed" or rng.random() < 0.25
    tex = lidar_texture(shape, res, amp=float(rng.uniform(0.03, 0.08)),
                        furrow_amp=float(rng.uniform(0.02, 0.06)) if plowed_field else 0.0,
                        rot_deg=float(rng.uniform(0, 180)), rng=rng)
    if plowed_field: fx.append("furrows")
    if texture_src is not None and rng.random() < 0.6:
        tex = tex + real_microrelief(texture_src, shape[0], rng=rng); fx.append("real-texture")
    return dz, mask, tex, fx


def realistic(bg, dz, mask, params, res=1.0, texture_src=None, rng=None):
    """Combine a synthetic mound with real ground and apply random realism effects.
    Returns (z, mask, effects)."""
    rng = rng or np.random.default_rng()
    dz, mask, tex, fx = realistic_parts(bg.shape, dz, mask, params, res, texture_src, rng)
    z = bg + dz + tex
    if rng.random() < 0.3:
        z = forest_sparse(z, keep_frac=float(rng.uniform(0.05, 0.25)), rng=rng); fx.append("forest")
    return z, mask, fx
