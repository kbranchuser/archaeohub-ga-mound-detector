"""Fetch USGS 3DEP bare-earth elevation and build Davis-style 3-band chips.

3DEP ImageServer serves the national 1 m (where available) bare-earth DEM as an
on-the-fly export, so no LAZ/PDAL processing is needed. We request it in UTM at
Davis's pixel size (5 ft = 1.524 m) so features have the same size in pixels as
in his training chips.
"""
import io, math, urllib.parse, urllib.request
import numpy as np
import rasterio
from rasterio.io import MemoryFile

EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
DAVIS_RES_M = 1.524  # 5 ft


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _get(url, tries=5):
    """GET with backoff: the 3DEP ImageServer returns occasional 502/503/504s."""
    import time, urllib.error
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if i == tries - 1 or (code is not None and code < 500):
                raise
            time.sleep(5 * 2 ** i)


def fetch_dem(lat, lon, size_m, res_m=DAVIS_RES_M):
    """Square DEM centred on lat/lon, `size_m` across, in UTM metres. Returns (z, transform, epsg)."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    h = size_m / 2
    px = int(round(size_m / res_m))
    q = urllib.parse.urlencode({
        "bbox": f"{x - h},{y - h},{x + h},{y + h}", "bboxSR": epsg, "imageSR": epsg,
        "size": f"{px},{px}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "noData": -9999, "f": "image"})
    data = _get(f"{EXPORT}?{q}")
    with MemoryFile(data) as mf, mf.open() as ds:
        z = ds.read(1).astype("float32")
        return np.where(z <= -9000, np.nan, z), ds.transform, epsg


def fetch_dem_tiled(lat, lon, size_m, res_m=DAVIS_RES_M, parts=2):
    """Like fetch_dem but as parts×parts smaller requests stitched on one UTM grid
    (large single exports time out on the 3DEP server)."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    n = int(round(size_m / res_m)); step = n // parts; out = np.full((n, n), np.nan, "float32")
    x0, y1 = x - size_m / 2, y + size_m / 2
    for i in range(parts):
        for j in range(parts):
            r0, c0 = i * step, j * step; r1 = n if i == parts - 1 else r0 + step; c1 = n if j == parts - 1 else c0 + step
            bx0, bx1 = x0 + c0 * res_m, x0 + c1 * res_m; by1, by0 = y1 - r0 * res_m, y1 - r1 * res_m
            q = urllib.parse.urlencode({
                "bbox": f"{bx0},{by0},{bx1},{by1}", "bboxSR": epsg, "imageSR": epsg, "size": f"{c1 - c0},{r1 - r0}",
                "format": "tiff", "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "noData": -9999, "f": "image"})
            with MemoryFile(_get(f"{EXPORT}?{q}")) as mf, mf.open() as ds:
                out[r0:r1, c0:c1] = ds.read(1)
    from rasterio.transform import from_origin
    return np.where(out <= -9000, np.nan, out), from_origin(x0, y1, res_m, res_m), epsg


def hillshade(z, res, az=315.0, alt=45.0):
    """ArcGIS-style hillshade (0-255), Horn's method."""
    zp = np.pad(np.nan_to_num(z, nan=np.nanmean(z)), 1, mode="edge")
    dzdx = ((zp[:-2, 2:] + 2 * zp[1:-1, 2:] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[1:-1, :-2] + zp[2:, :-2])) / (8 * res)
    dzdy = ((zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:])) / (8 * res)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    zen, azr = math.radians(90 - alt), math.radians(360 - az + 90)
    hs = 255 * (math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(azr - aspect))
    return np.clip(hs, 0, 255), np.degrees(slope)


def davis_bands(z, res=DAVIS_RES_M, slope_scale=255 / 90, dem_levels_per_m=10.0):
    """3-band uint8 in Davis's ACTUAL band order (verified visually): hillshade, slope, DEM.

    His README says DEM/hillshade/slope, but the chips are hillshade/slope/DEM.
    Slope: degrees * slope_scale. DEM: his was stretched over a whole county
    (a chip spans ~43 levels), so here each chip is centred on its median at a
    fixed levels-per-metre; both scales are calibration knobs, not known values.
    """
    hs, slope = hillshade(z, res)
    sl = np.clip(slope * slope_scale, 0, 255)
    med = np.nanmedian(z)
    dem = np.clip(40 + (np.nan_to_num(z, nan=med) - med) * dem_levels_per_m, 0, 255)
    return np.stack([hs, sl, dem]).astype("uint8")
