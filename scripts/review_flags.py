"""Learn from mistakes, step 2: contact sheets of every flag for a person to judge.

    python scripts/review_flags.py MINING_DIR [min_score]

Writes MINING_DIR/review_NN.png: 24 numbered 200 m close-ups per sheet (hillshade
left, local-relief band right, flag circled). The reviewer then writes
MINING_DIR/decisions.json: {"negative": [ids...], "possible": [ids...]} where
"negative" = clearly modern or natural (becomes a "not a mound" lesson) and
"possible" = could be a real mound (kept private, never trained on).
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import hillshade, DAVIS_RES_M

d = Path(sys.argv[1]); thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
flags = [f for f in json.load(open(d / "flags.json")) if f.get("k") is not None and f["score"] >= thr]
for i, f in enumerate(flags): f["id"] = i
json.dump(flags, open(d / "flags_indexed.json", "w"), indent=1)
try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
except OSError: font = None
T = 200; cache = {}
for s in range(0, len(flags), 24):
    sheet = Image.new("RGB", (T * 2 * 4, T * 6), "white")
    for j, f in enumerate(flags[s:s + 24]):
        if f["area"] not in cache:
            a = np.load(d / f"{f['area']}.npz"); z = a["z"].astype("float64")
            cache = {f["area"]: (hillshade(np.where(np.isfinite(z), z, np.nanmedian(z)), 1.0)[0], a["bands"][0])}
        hs, lrm = cache[f["area"]]
        r1, c1 = int(f["row_grid"] * DAVIS_RES_M), int(f["col_grid"] * DAVIS_RES_M)       # 1 m hillshade px
        crop = lambda a, r, c, h: np.pad(a, h, mode="edge")[r:r + 2 * h, c:c + 2 * h]
        left = Image.fromarray(crop(hs, r1, c1, 100).astype("uint8")).convert("RGB")
        rg, cg = int(f["row_grid"]), int(f["col_grid"])
        right = Image.fromarray(crop(lrm, rg, cg, 66).astype("uint8")).convert("RGB").resize((T, T))
        for im in (left, right):
            ImageDraw.Draw(im).ellipse([T / 2 - 30, T / 2 - 30, T / 2 + 30, T / 2 + 30], outline=(255, 40, 40), width=2)
        x, y = (j % 4) * T * 2, (j // 4) * T
        sheet.paste(left, (x, y)); sheet.paste(right, (x + T, y))
        ImageDraw.Draw(sheet).text((x + 4, y + 2), f"#{f['id']} {f['score']:.2f} {f['region'][:5]}", fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    sheet.save(d / f"review_{s // 24:02d}.png")
print(len(flags), "flags on", (len(flags) + 23) // 24, "sheets")
