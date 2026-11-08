"""Scan areas on a cloud machine and save Archaeohub-ready readings (private).

    python scripts/area_scan.py AREAS_CSV OUT_DIR [CKPT ...]
AREAS_CSV: name,lat,lon,size_m. Writes OUT_DIR/<name>.json — the same payload the
Archaeohub Detections tab uses (flags, close-ups, relief views, exact corners).
For plain files instead, use scripts/detect.py.
"""
import sys, csv, json, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.reading import run_reading, default_ckpts
from gamd.scan import load

areas, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ckpts = sys.argv[3:] or default_ckpts()
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
models = [load(c, dev)[0] for c in ckpts]; names = [Path(c).parent.parent.name + "/" + Path(c).parent.name for c in ckpts]
for r in csv.DictReader(open(areas)):
    dst = out / f"{r['name'].replace(' ', '_').replace('/', '-')}.json"
    if dst.exists(): continue
    job = dict(name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]), size_m=float(r["size_m"]))
    t0 = time.time()
    try:
        res = run_reading(job, models, names, dev, lambda p, s: print(f"  {r['name']}: {p:.0f}% {s}", flush=True))
    except Exception as e:
        print(f"{r['name']}: FAILED {e}", flush=True); continue
    dst.write_text(json.dumps(dict(job=job, result=res)))
    print(f"{r['name']}: {len(res['flags'])} spots, best {max([f['score'] for f in res['flags']] or [0]):.2f}, {time.time() - t0:.0f}s", flush=True)
