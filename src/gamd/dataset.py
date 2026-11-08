"""Training scenes (step 6.5).

A scene is a 600 m square of real 1 m bare-earth elevation with labels:
  z        float32 elevation (m), synthetic mounds already inserted
  inst     uint16 instance raster (0 = none, k = instances[k-1])
  ignore   bool, pixels the trainer must not count as background
plus a JSON sidecar (location, region, group/split, instance list, effects).
Raw elevation is stored, not image bands, so any input recipe (Davis bands,
VAT, local relief) can be computed at training time; chips are cut at random
from the scene there too.
"""
import json, math
import numpy as np
from scipy import ndimage as ndi
from . import synth

WIN = 260  # px window a synthetic mound is drawn in (random_mound size)


def _feather(n, support, edge_frac=0.25):
    """Blend weight for a window: 1 where the mound or its apron is, fading to 0
    at the window edge so added texture leaves no square seam."""
    i = np.minimum(np.arange(n), np.arange(n)[::-1]) / (edge_frac * n)
    w_edge = np.clip(np.minimum.outer(i, i), 0, 1)
    w_sup = np.clip(ndi.gaussian_filter(ndi.binary_dilation(support, iterations=10).astype("float32"), 6) * 1.5, 0, 1)
    return np.maximum(w_edge, w_sup)


def _paste(z, row, col, n, fn):
    """Apply fn(window_view, (ys, xs) into window) to the part of an n×n window
    centred on (row, col) that falls inside z."""
    r0, c0 = row - n // 2, col - n // 2
    ys, xs = slice(max(r0, 0), min(r0 + n, z.shape[0])), slice(max(c0, 0), min(c0 + n, z.shape[1]))
    wy, wx = slice(ys.start - r0, ys.stop - r0), slice(xs.start - c0, xs.stop - c0)
    return fn(ys, xs, wy, wx)


def insert_mound(z, inst, k, row, col, dz, mask, params, texture_src, rng, res=1.0):
    """Add one synthetic mound (with path/cut/texture realism) into z in place and
    write instance id k into inst. Returns the realism effects applied."""
    dz, mask, tex, fx = synth.realistic_parts((WIN, WIN), dz, mask, params, res, texture_src, rng)
    w = _feather(WIN, dz > 0.02)
    def f(ys, xs, wy, wx):
        z[ys, xs] += dz[wy, wx] + w[wy, wx] * tex[wy, wx]
        inst[ys, xs][mask[wy, wx]] = k
    _paste(z, row, col, WIN, f)
    return fx


def insert_decoy(z, row, col, texture_src, rng, res=1.0):
    """Same texture/furrow treatment as a mound gets, with no mound, so 'extra
    texture' is never a shortcut for 'mound here'."""
    _, _, tex, fx = synth.realistic_parts((WIN, WIN), np.zeros((WIN, WIN)), np.zeros((WIN, WIN), bool),
                                          {"kind": None}, res, texture_src, rng)
    n = int(rng.integers(60, 140)); yy, xx = np.mgrid[:WIN, :WIN]
    blob = np.hypot(yy - WIN / 2, xx - WIN / 2) < n / 2
    w = _feather(WIN, blob)
    def f(ys, xs, wy, wx):
        z[ys, xs] += w[wy, wx] * tex[wy, wx]
    _paste(z, row, col, WIN, f)
    return fx


def real_mound_mask(z, py, px, frac=0.15, ground_win=141, max_r=110):
    """Outline of a real mound around summit (py, px): the connected area rising
    > max(0.3 m, frac × height) above a top-hat ground (same rule as
    scripts/measure_mounds.py and the synthetic masks). Returns (mask, height)."""
    zf = np.where(np.isfinite(z), z, np.nanmin(z))
    ground = ndi.maximum_filter(ndi.minimum_filter(zf, size=ground_win), size=ground_win)
    rel = zf - ground; h = float(rel[py, px])
    lab, _ = ndi.label(ndi.binary_opening(rel > max(0.3, frac * h), iterations=2))
    if lab[py, px] == 0:
        return np.zeros(z.shape, bool), h
    yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]
    return ndi.binary_fill_holes(lab == lab[py, px]) & (np.hypot(yy - py, xx - px) < max_r), h


def summit_near(z, row, col, r=25):
    """Highest cell within r px of (row, col): snaps a published point to the summit."""
    y0, x0 = max(row - r, 0), max(col - r, 0)
    win = z[y0:row + r, x0:col + r]
    py, px = np.unravel_index(np.nanargmax(win), win.shape)
    return int(py + y0), int(px + x0)


def save_scene(path, z, inst, ignore, meta):
    np.savez_compressed(path.with_suffix(".npz"), z=z.astype("float32"), inst=inst.astype("uint16"), ignore=ignore)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def load_scene(path):
    d = np.load(path.with_suffix(".npz"))
    return d["z"], d["inst"], d["ignore"], json.loads(path.with_suffix(".json").read_text())


def linear_corridor(z, bg, inst, row, col, r_px, rng, res=1.0, kind=None):
    """A road/railroad corridor across the whole scene, through or beside the mound at
    (row, col): a cut (graded floor below ground, sloped sides, slicing through any
    mound) or a raised fill (embankment beside it). Mound pixels the cut removes
    leave the instance mask. kind: "cut" / "fill" / None (65% cut). A fill may end in a
    blunt raised block at (row, col), like a bridge abutment. Modifies z/inst in place;
    returns a description."""
    cut = (rng.random() < 0.65) if kind is None else kind == "cut"
    abut = not cut and kind == "fill" and rng.random() < 0.5
    ang = math.radians(rng.uniform(0, 180)); r_m = r_px * res
    off = rng.uniform(-0.5, 1.4) * r_m if cut else rng.uniform(1.1, 2.0) * r_m * rng.choice([-1, 1])
    w = float(rng.uniform(8, 16)); h = float(rng.uniform(1.5, 5.0))
    side = math.tan(math.radians(rng.uniform(30, 40) if cut else rng.uniform(25, 35)))
    H, W = z.shape; yy, xx = np.mgrid[:H, :W]
    d = np.abs((xx - col) * res * math.sin(ang) - (yy - row) * res * math.cos(ang) - off)
    ground = ndi.gaussian_filter(np.where(np.isfinite(bg), bg, np.nanmedian(bg)), 15 / res)
    old = z.copy()
    if cut:
        surf = ground - h + np.maximum(d - w / 2, 0) * side
        removed = (z - surf) > 0.5
        new = np.minimum(z, surf)
        inst[removed] = 0
    else:
        surf = ground + h - np.maximum(d - w / 2, 0) * side
        if abut:   # embankment stops at (row, col) with a steep end face
            along = (xx - col) * res * math.cos(ang) + (yy - row) * res * math.sin(ang)
            surf = np.minimum(surf, ground + h + along * math.tan(math.radians(40)))
        new = np.maximum(z, surf)
    z[:] = old + ndi.gaussian_filter(np.nan_to_num(new - old), 1.0 / res)   # rounded shoulders, not knife edges
    return dict(kind="cut" if cut else "fill", abutment=abut, width_m=round(w, 1), depth_m=round(h, 1), offset_m=round(off, 1),
                angle_deg=round(math.degrees(ang), 1), mound_row=row, mound_col=col)
