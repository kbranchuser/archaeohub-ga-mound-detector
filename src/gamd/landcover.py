"""NLCD 2021 land cover on the same UTM grid as `dem.fetch_dem`.

Used to keep synthetic mounds off modern developed ground (graded pads, lots,
roads), which bare-earth LiDAR shows as level, real-looking ground. NLCD is
30 m, so edges are coarse; callers dilate the mask a little.
"""
import math, urllib.parse
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from .dem import _get, utm_epsg

WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
COVERAGE = "mrlc_download__NLCD_2021_Land_Cover_L48"
DEVELOPED = (21, 22, 23, 24)   # open space, low, medium, high intensity
WATER = (11,)


def fetch_nlcd(lat, lon, size_m, res_m):
    """NLCD class codes (uint8) for the square `fetch_dem(lat, lon, size_m, res_m)` covers."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    h = size_m / 2
    px = int(round(size_m / res_m))
    # bbox of the UTM square in NLCD's Albers (EPSG:5070), padded one 30 m cell
    to_alb = Transformer.from_crs(epsg, 5070, always_xy=True)
    cx, cy = to_alb.transform([x - h, x + h, x - h, x + h], [y - h, y - h, y + h, y + h])
    x0, x1, y0, y1 = min(cx) - 60, max(cx) + 60, min(cy) - 60, max(cy) + 60
    q = urllib.parse.urlencode([
        ("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
        ("coverageId", COVERAGE), ("format", "image/tiff"),
        ("subset", f"X({x0},{x1})"), ("subset", f"Y({y0},{y1})")])
    with MemoryFile(_get(f"{WCS}?{q}")) as mf, mf.open() as ds:
        src = ds.read(1); src_tr, src_crs = ds.transform, ds.crs
    out = np.zeros((px, px), "uint8")
    reproject(src, out, src_transform=src_tr, src_crs=src_crs,
              dst_transform=from_origin(x - h, y + h, res_m, res_m), dst_crs=f"EPSG:{epsg}",
              resampling=Resampling.nearest)
    return out


def modern_mask(nlcd, res_m, buffer_m=15):
    """True where ground is developed (or open water), grown by `buffer_m`."""
    from scipy import ndimage as ndi
    m = np.isin(nlcd, DEVELOPED + WATER)
    return ndi.binary_dilation(m, iterations=max(1, int(math.ceil(buffer_m / res_m)))) if m.any() else m
