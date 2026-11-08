"""Run the Davis model over a DEM and draw detections on the hillshade.
    python scripts/known_site_test.py data/etowah/dem.tif out.png [slope_scale] [dem_levels_per_m]"""
import sys, json
from pathlib import Path
import numpy as np, rasterio, torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.davis import build_model
from gamd.detect import detect
from gamd.dem import hillshade

dem, out = sys.argv[1], sys.argv[2]
kw = dict(slope_scale=float(sys.argv[3]) if len(sys.argv) > 3 else 255 / 90,
          dem_levels_per_m=float(sys.argv[4]) if len(sys.argv) > 4 else 10.0)
ds = rasterio.open(dem); z = ds.read(1); res = ds.res[0]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(Path(__file__).resolve().parents[1] / "models/davis2021/RCNN_2_Output.pth", dev)
dets = detect(model, z, res, device=dev, **kw)
hs, _ = hillshade(z, res); img = Image.fromarray(hs.astype("uint8")).convert("RGB"); d = ImageDraw.Draw(img)
col = {"Mound": (255, 60, 60), "Ring": (60, 200, 255), "Modern": (255, 200, 0)}
for i, t in enumerate(dets):
    b = t["box"]; d.rectangle(b, outline=col[t["cls"]], width=3); d.text((b[0] + 2, b[1] + 2), f"{i+1} {t['cls'][0]} {t['score']:.2f}", fill=col[t["cls"]])
img.save(out)
print(json.dumps(kw), f"{len(dets)} detections:", [(t['cls'], round(t['score'], 2), round(t['diam_m'])) for t in dets[:12]])
