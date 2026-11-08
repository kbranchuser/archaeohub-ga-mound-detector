"""Where can a synthetic mound go? Placement rules for training data (step 6.4).

A cell is suitable when it is
  - level:            slope < MAX_SLOPE over the whole would-be footprint,
  - not a channel:    not below its surroundings (stream beds, ditches),
  - real ground:      not an interpolated gap (buildings, water, bridges are
                      removed from bare-earth LiDAR and filled with flat TIN
                      facets, which have ~zero curvature texture),
  - not occupied:     no overlap with another placed mound.
  - not developed:    NLCD 2021 developed classes (21-24) and water are removed by
                      the caller (`landcover.modern_mask`); graded pads, parking lots
                      and roads otherwise pass every terrain test above.
Known mound sites are excluded upstream (sample areas are kept >= 3 km away).
"""
import numpy as np
from scipy import ndimage as ndi

MAX_SLOPE = 4.0  # degrees


def suitability(z, res=1.0):
    """Boolean raster of cells where a mound could plausibly have been built."""
    zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    zs = ndi.gaussian_filter(zf, 1.0)
    gy, gx = np.gradient(zs, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    level = ndi.maximum_filter(slope, size=int(15 / res)) < MAX_SLOPE        # level over ~15 m
    channel = (zs - ndi.uniform_filter(zs, size=int(61 / res))) < -0.4       # below surroundings
    lap = ndi.laplace(zf)
    tex = ndi.uniform_filter(lap ** 2, size=5)                               # natural ground is never perfectly smooth
    interpolated = tex < 1e-6   # TIN-filled gaps are ~1e-18; natural ground is >= ~1e-4 (measured on 12 GA areas)
    interpolated |= ~np.isfinite(z)
    interpolated = ndi.binary_dilation(interpolated, iterations=int(10 / res))
    return level & ~ndi.binary_dilation(channel, iterations=3) & ~interpolated


def choose_spots(ok, radii_px, rng, margin_px=20, tries=400, near=None, near_px=40, near_p=0.0):
    """Greedy random placement: the mound's footprint (radius r) must lie on suitable
    ground, and its 1.6 r spacing disc must not touch another placed mound.
    near / near_px / near_p: with probability near_p a mound must also sit within
    r + near_px of steep ground (`near` = px distance to it), i.e. on a bluff or
    terrace edge like Ocmulgee's mounds; falls back to anywhere if no such spot.
    radii_px: list of mound radii (px) to place, largest first. Returns [(row, col, r)]."""
    H, W = ok.shape; taken = np.zeros_like(ok); placed = []
    dist_ok = ndi.distance_transform_edt(ok)
    yy, xx = np.ogrid[:H, :W]
    for r in sorted(radii_px, reverse=True):
        edge = near is not None and rng.random() < near_p
        for t in range(tries * (2 if edge else 1)):
            y, x = rng.integers(margin_px + r, H - margin_px - r), rng.integers(margin_px + r, W - margin_px - r)
            if dist_ok[y, x] < r * 0.8:
                continue
            if edge and t < tries and near[y, x] > r + near_px:
                continue
            disc = (yy - y) ** 2 + (xx - x) ** 2 <= (1.6 * r) ** 2
            if taken[disc].any():
                continue
            taken |= disc; placed.append((int(y), int(x), int(r)))
            break
    return placed
