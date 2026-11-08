"""Model input recipes (step 5): how a DEM becomes image bands.

  davis    hillshade, slope, DEM (per-chip median stretch): Davis et al. 2021.
           The DEM band saturates where a chip spans a lot of relief (bluffs:
           Ocmulgee's tiles clip 3-27% of pixels), erasing mound shape.
  relief3  local relief (white top-hat), positive openness, slope.
  relief6  relief3 + negative openness (cuts, ditches), geomorphon landform
           index, anisotropic sky-view factor: the Archaeohub views archaeologists
           read by eye.

The relief bands describe height relative to nearby ground and the shape of the
surface, not absolute elevation, so they read the same on a floodplain or on a
bluff. They need context beyond the chip (the top-hat window is 141 m), so they
are computed on a padded window and cropped: callers pass `pad` px of extra
DEM on every side.
"""
import math
import numpy as np
from scipy import ndimage as ndi
from .dem import hillshade, davis_bands

RECIPES = {"davis": 3, "relief3": 3, "relief6": 6}
PAD = 75   # px at 1.524 m (114 m) of context needed on each side by relief recipes


def local_relief(z, res, window_m=141.0):
    """Height above local ground (m): white top-hat with a flat window wider than
    any mound; plateaus and bluffs wider than the window stay ground."""
    n = max(3, int(round(window_m / res)) | 1)
    return z - ndi.maximum_filter(ndi.minimum_filter(z, size=n, mode="nearest"), size=n, mode="nearest")


DIRS = ((-1, 0, 0), (-1, 1, 45), (0, 1, 90), (1, 1, 135), (1, 0, 180), (1, -1, 225), (0, -1, 270), (-1, -1, 315))  # dy, dx, azimuth


def horizons(z, res, radii_m):
    """Largest upward and downward elevation angles (degrees) per direction within
    each radius: {radius: (up[8,H,W], down[8,H,W])}. One pass serves openness,
    sky-view factor and geomorphons."""
    R = max(1, int(round(max(radii_m) / res))); H, W = z.shape
    zp = np.pad(z, R, mode="edge")
    out = {r: (np.full((8, H, W), -90.0), np.full((8, H, W), -90.0)) for r in radii_m}
    for d, (dy, dx, _) in enumerate(DIRS):
        step = res * math.hypot(dy, dx); up = np.full(z.shape, -90.0); dn = np.full(z.shape, -90.0)
        for k in range(1, R + 1):
            dist = k * step
            if dist > max(radii_m): break
            sh = zp[R + dy * k:R + dy * k + H, R + dx * k:R + dx * k + W]
            ang = np.degrees(np.arctan((sh - z) / dist))
            up = np.maximum(up, ang); dn = np.maximum(dn, -ang)
            for r in radii_m:
                if dist <= r < dist + step: out[r][0][d], out[r][1][d] = up, dn
        for r in radii_m:   # radius shorter than one step in this direction
            if r < step: out[r][0][d], out[r][1][d] = up, dn
    return out


def openness(z, res, radius_m=15.0, negative=False, hz=None):
    """Openness (degrees, ~60-100): mean over 8 directions of 90° minus the largest
    elevation angle within radius (Yokoyama et al. 2002). Convex forms are high.
    Negative openness looks downward: hollows/cuts are high."""
    up, dn = (hz or horizons(z, res, [radius_m]))[radius_m]
    return 90.0 - (dn if negative else up).mean(0)


def anisotropic_svf(z, res, radius_m=10.0, azimuth=315.0, hz=None):
    """Sky-view factor weighted toward `azimuth` (cf. RVT anisotropic SVF): visible
    sky fraction, with the sky on the lit side counting most."""
    up, _ = (hz or horizons(z, res, [radius_m]))[radius_m]
    svf_d = 1 - np.sin(np.radians(np.clip(up, 0, 90)))
    w = np.array([0.25 + (1 + math.cos(math.radians(a - azimuth))) / 2 for _, _, a in DIRS])
    return np.tensordot(w / w.sum(), svf_d, 1)


def geomorphon_index(z, res, radius_m=30.0, flat_deg=1.0, hz=None):
    """Geomorphon-style landform index (Jasiewicz & Stepinski 2013): per direction
    the ground is higher (+), lower (-) or level; index = #lower - #higher, from
    -8 (pit) through 0 (flat / straight slope) to +8 (peak); ridges ~+4..6."""
    up, dn = (hz or horizons(z, res, [radius_m]))[radius_m]
    return ((dn - up) > flat_deg).sum(0) - ((up - dn) > flat_deg).sum(0)


def _u8(a, lo, hi):
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255)


def make_bands(z, res, recipe, pad=0, **kw):
    """Bands (C×H×W uint8) for the centre of z after removing `pad` px per side."""
    crop = (slice(pad, z.shape[0] - pad), slice(pad, z.shape[1] - pad)) if pad else (slice(None), slice(None))
    if recipe == "davis":
        return davis_bands(z[crop], res, **kw)
    zf = np.where(np.isfinite(z), z, np.nanmedian(z)).astype("float64")
    zs = ndi.gaussian_filter(zf, 0.7)                      # tame 1-px LiDAR noise before derivatives
    _, slope = hillshade(zs, res)
    lrm = local_relief(zs, res)
    hz = horizons(zs, res, [10.0, 15.0])
    b = [np.sqrt(np.clip(lrm, 0, 16) / 16) * 255,           # sqrt: 1 m plowed mound ~64, 16 m platform 255
         _u8(openness(zs, res, hz=hz), 65, 100),
         _u8(slope, 0, 50)]
    if recipe == "relief6":
        b += [_u8(openness(zs, res, negative=True, hz=hz), 65, 100),
              _u8(geomorphon_index(ndi.gaussian_filter(zf, 1.5), res, flat_deg=1.5), -8, 8),   # smoother: no speckle on flats
              _u8(anisotropic_svf(zs, res, hz=hz), 0.75, 1.0)]
    return np.stack([x[crop] for x in b]).astype("uint8")
