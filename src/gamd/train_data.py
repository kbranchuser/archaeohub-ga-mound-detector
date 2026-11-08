"""Training chips from 6.5 scenes, in Davis's input recipe.

A chip is 200 px at 1.524 m (305 m of ground; relief recipes add context, see bands.py): cut from a 1 m scene, resampled,
randomly rotated by 90° steps / flipped (before banding, so the hillshade sun
stays at 315° like real data), then turned into hillshade / slope / DEM bands.
Targets follow torchvision Mask R-CNN: boxes, labels, masks. An instance counts
only if at least `min_vis` of it is inside the chip; cut-off remnants become
ignore so they are neither mound nor background; so are real mounds whose outline
is unknown (`mask_quality == "point"`).
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from .dem import DAVIS_RES_M
from .bands import make_bands, PAD

CHIP = 200
MOUND = 2   # Davis class id for "Mound"


class Scenes:
    """All scenes of the given ids held in memory (float16 elevation offsets)."""
    def __init__(self, root, ids):
        self.items = []
        for i in ids:
            d = np.load(Path(root) / f"{i}.npz"); meta = json.loads((Path(root) / f"{i}.json").read_text())
            z = d["z"]; base = float(np.nanmedian(z))
            self.items.append(dict(z=(z - base).astype("float32"), base=base, inst=d["inst"], ignore=d["ignore"], meta=meta))


def _resample(a, out, order):
    f = out / a.shape[0]
    return ndi.zoom(a, f, order=order, mode="nearest", prefilter=False)[:out, :out]


def _window(a, y0, x0, n, fill_edge=True):
    """n×n window at (y0, x0) that may run past the array: edge-replicated outside."""
    H, W = a.shape
    ys, xs = slice(max(y0, 0), min(y0 + n, H)), slice(max(x0, 0), min(x0 + n, W))
    w = a[ys, xs]
    return np.pad(w, ((ys.start - y0, y0 + n - ys.stop), (xs.start - x0, x0 + n - xs.stop)), mode="edge" if fill_edge else "constant")


def sample_chip(sc, rng, focus_p=0.6, min_vis=0.5, dem_levels=(8.0, 20.0), res_src=1.0, recipe="davis"):
    """One (image tensor C×200×200 in 0-1, target dict, ignore mask) from scene dict sc.
    Relief recipes get PAD px of context per side, cut beyond the scene by edge replication."""
    pad = 0 if recipe == "davis" else PAD
    span = int(round(CHIP * DAVIS_RES_M / res_src)); ppad = int(round(pad * DAVIS_RES_M / res_src))
    H, W = sc["z"].shape; inst = sc["inst"]; insts = sc["meta"]["instances"]
    if insts and rng.random() < focus_p:                      # centre near a mound
        k = insts[int(rng.integers(len(insts)))]
        cy, cx = k["row"] + rng.integers(-90, 91), k["col"] + rng.integers(-90, 91)
    elif sc["meta"].get("focus") and rng.random() < focus_p:  # or near a look-alike
        f = sc["meta"]["focus"][int(rng.integers(len(sc["meta"]["focus"])))]
        cy, cx = f["row"] + rng.integers(-90, 91), f["col"] + rng.integers(-90, 91)
    else:
        cy, cx = rng.integers(span // 2, H - span // 2), rng.integers(span // 2, W - span // 2)
    y0 = int(np.clip(cy - span // 2, 0, H - span)) - ppad; x0 = int(np.clip(cx - span // 2, 0, W - span)) - ppad
    n = span + 2 * ppad
    z = _window(sc["z"], y0, x0, n); im = _window(inst, y0, x0, n, False); ig = _window(sc["ignore"], y0, x0, n, False)
    rot, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
    def aug(a):
        a = np.rot90(a, rot)
        return np.ascontiguousarray(a[:, ::-1] if flip else a)
    z, im, ig = aug(z), aug(im), aug(ig)
    big = CHIP + 2 * pad; c = slice(pad, pad + CHIP)
    zc = _resample(np.nan_to_num(z, nan=np.nanmedian(z)), big, 1); zc[_resample(np.isnan(z).astype("uint8"), big, 0) > 0] = np.nan
    imc = _resample(im, big, 0)[c, c]; igc = (_resample(ig.astype("uint8"), big, 0) > 0)[c, c]
    kw = dict(dem_levels_per_m=float(rng.uniform(*dem_levels))) if recipe == "davis" else {}
    bands = make_bands(zc, DAVIS_RES_M, recipe, pad=pad, **kw)
    boxes, masks = [], []
    for k in np.unique(imc):
        if k == 0: continue
        m = imc == k; full = (inst == k).sum() * (res_src / DAVIS_RES_M) ** 2
        if m.sum() < min_vis * full:
            igc |= m; continue
        if insts[k - 1].get("mask_quality") == "point":       # outline unknown: neither mound nor background
            igc |= ndi.binary_dilation(m, iterations=16); continue
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]); masks.append(m)
    t = dict(boxes=torch.tensor(np.array(boxes, "float32").reshape(-1, 4)),
             labels=torch.full((len(boxes),), MOUND, dtype=torch.int64),
             masks=torch.from_numpy(np.array(masks, "uint8").reshape(-1, CHIP, CHIP)))
    return torch.from_numpy(bands.astype("float32") / 255.0), t, igc


# ---------------------------------------------------------------- precomputed bands
# Relief bands are slow (openness/geomorphons ~0.3 s per padded chip), which left the
# GPU idle. scripts/precompute_bands.py computes relief6 once per scene on the 1.524 m
# grid; relief3 is its first 3 bands. Differences from on-the-fly chips: the grid is
# fixed (no sub-pixel offsets) and rotation/flip happen after banding, so the one
# directional band (anisotropic SVF) sees varied light directions in training.
PRE_BANDS = {"relief3": 3, "relief6": 6}


class PreScenes:
    """Precomputed scenes: bands (6×N×N uint8), inst, ignore on the 1.524 m grid."""
    def __init__(self, root, ids):
        self.items = []
        for i in ids:
            d = np.load(Path(root) / f"{i}.npz"); meta = json.loads((Path(root) / f"{i}.json").read_text())
            self.items.append(dict(bands=d["bands"], inst=d["inst"], ignore=d["ignore"], meta=meta))


def sample_chip_pre(sc, rng, focus_p=0.6, min_vis=0.5, recipe="relief6"):
    """Same contract as sample_chip, cut from precomputed bands."""
    nb = PRE_BANDS[recipe]; bands_all = sc["bands"]; inst = sc["inst"]; meta = sc["meta"]
    H, W = inst.shape; insts = meta["instances"]; f = meta["grid_scale"]   # source px -> grid px
    jit = int(90 * f)
    if insts and rng.random() < focus_p:
        k = insts[int(rng.integers(len(insts)))]
        cy, cx = int(k["row"] * f) + rng.integers(-jit, jit + 1), int(k["col"] * f) + rng.integers(-jit, jit + 1)
    elif meta.get("focus") and rng.random() < focus_p:
        fo = meta["focus"][int(rng.integers(len(meta["focus"])))]
        cy, cx = int(fo["row"] * f) + rng.integers(-jit, jit + 1), int(fo["col"] * f) + rng.integers(-jit, jit + 1)
    else:
        cy, cx = rng.integers(CHIP // 2, H - CHIP // 2), rng.integers(CHIP // 2, W - CHIP // 2)
    y0 = int(np.clip(cy - CHIP // 2, 0, H - CHIP)); x0 = int(np.clip(cx - CHIP // 2, 0, W - CHIP))
    b = bands_all[:nb, y0:y0 + CHIP, x0:x0 + CHIP]; im = inst[y0:y0 + CHIP, x0:x0 + CHIP]; igc = sc["ignore"][y0:y0 + CHIP, x0:x0 + CHIP]
    rot, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
    def aug(a, axes=(0, 1)):
        a = np.rot90(a, rot, axes=axes)
        return np.ascontiguousarray(np.flip(a, axes[1]) if flip else a)
    b, imc, igc = aug(b, (1, 2)), aug(im), aug(igc).copy()
    boxes, masks = [], []
    for k in np.unique(imc):
        if k == 0: continue
        m = imc == k
        if m.sum() < min_vis * (inst == k).sum():
            igc |= m; continue
        if insts[k - 1].get("mask_quality") == "point":
            igc |= ndi.binary_dilation(m, iterations=16); continue
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]); masks.append(m)
    t = dict(boxes=torch.tensor(np.array(boxes, "float32").reshape(-1, 4)),
             labels=torch.full((len(boxes),), MOUND, dtype=torch.int64),
             masks=torch.from_numpy(np.array(masks, "uint8").reshape(-1, CHIP, CHIP)))
    return torch.from_numpy(b.astype("float32") / 255.0), t, igc
