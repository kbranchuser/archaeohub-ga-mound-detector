"""Compute relief6 bands once per training scene (see train_data.PreScenes).

    python scripts/precompute_bands.py [workers]
Reads data/training/v1/scenes, writes data/training/v1/pre/*.npz|json (resumable).
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.bands import make_bands
from gamd.dem import DAVIS_RES_M

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "data/training/v1/scenes", ROOT / "data/training/v1/pre"


def one(sid):
    out = DST / f"{sid}.npz"
    if out.exists(): return f"{sid} exists"
    d = np.load(SRC / f"{sid}.npz"); meta = json.loads((SRC / f"{sid}.json").read_text())
    z = d["z"].astype("float64"); res = meta.get("res", 1.0)
    f = res / DAVIS_RES_M; n = int(round(z.shape[0] * f))
    nan = ~np.isfinite(z); zf = np.where(nan, np.nanmedian(z), z)
    zg = ndi.zoom(zf, n / z.shape[0], order=1, mode="nearest", prefilter=False)[:n, :n]
    bands = make_bands(zg, DAVIS_RES_M, "relief6")
    near = lambda a: ndi.zoom(a, n / a.shape[0], order=0, mode="nearest", prefilter=False)[:n, :n]
    np.savez_compressed(out, bands=bands, inst=near(d["inst"]), ignore=near(d["ignore"].astype("uint8")).astype(bool))
    meta["grid_scale"] = n / z.shape[0]; meta["grid_res"] = DAVIS_RES_M
    (DST / f"{sid}.json").write_text(json.dumps(meta))
    return sid


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in SRC.glob("*.json")); t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        for k, msg in enumerate(p.imap_unordered(one, ids), 1):
            if k % 50 == 0 or k == len(ids): print(f"[{k}/{len(ids)} {time.time() - t0:.0f}s] {msg}", flush=True)
