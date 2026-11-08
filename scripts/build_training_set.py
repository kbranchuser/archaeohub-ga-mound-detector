"""Step 6.5: build the training set as 600 m scenes (see src/gamd/dataset.py).

    python scripts/build_training_set.py synth N [workers]   # N synthetic scenes (resumable)
    python scripts/build_training_set.py synth-hard N [workers]  # bluff-edge + road/railroad corridor scenes
    python scripts/build_training_set.py real                # verified Georgia features
    python scripts/build_training_set.py oos                 # out-of-state park mounds (training only)
    python scripts/build_training_set.py mined MINING_DIR    # reviewed false flags as negatives
    python scripts/build_training_set.py manifest            # manifest.json + preview.png

Output: data/training/v1/scenes/*.npz|json, data/training/v1/manifest.json.

Synthetic: random Georgia ground (equal thirds mountains / Piedmont / coastal
plain) >= 3 km from any known site; 3-6 mounds on
suitable, undeveloped ground; 1-3 texture-only decoys; scene-wide forest
thinning on 30%. 20% of scenes are left empty (plain ground, developed or not).
Real: one scene per cluster of verified features. Mounds are positives with
measured outlines; reconstructions are positives flagged for lower weight;
possibles, effigies, earthworks and undetected sites are ignore discs. At
Etowah, Ocmulgee and Kolomoki other mounds have no coordinates, so within 1 km
of those sites the rest of the scene is ignore (they teach only positives). Splits: synthetic every
10th scene = val; real scenes carry their site as `group` so step 8 can hold
out whole sites (leave-one-site-out) for an honest score.
"""
import sys, csv, json, math, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gamd.dem import fetch_dem, hillshade, DAVIS_RES_M
from gamd import synth
from gamd.place import suitability, choose_spots
from gamd.landcover import fetch_nlcd, modern_mask
from gamd.dataset import insert_mound, insert_decoy, real_mound_mask, summit_near, save_scene, load_scene, linear_corridor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/training/v1"; SC = OUT / "scenes"
SIZE, RES = 600, 1.0
DAVIS_TILES = Path.home() / "Downloads/SC_Deep_Learning_Supplemental_Files/Training_Data"


def region_of(lat, lon):
    """Rough physiographic region: Blue Ridge + Ridge & Valley / Piedmont / Coastal Plain (fall line)."""
    if lat > 34.45 or (lat > 33.9 and lon < -84.6): return "mountains"
    return "coastal_plain" if lat < 32.4 + (lon + 85.0) * 1.1 / 3.1 else "piedmont"


def known_points():
    pts = []
    for f in ("ga_mounds_verified.csv", "ga_mound_candidates.csv"):
        for r in csv.DictReader(open(ROOT / "data" / f)):
            try: pts.append((float(r["lat"]), float(r["lon"])))
            except (ValueError, KeyError): pass
    return pts


def km(a, b):
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians(a[0]))) * 111.2


# ---------------------------------------------------------------- synthetic
def synth_scene(i, hard=False):
    """hard: Ocmulgee-like cases. 60% of mounds on a bluff/terrace edge (within 40 m of
    > 15° ground); half of the mounds get a road/railroad cut (max 2); 0-2 corridors away
    from mounds, mostly fills, half ending in a bridge-abutment block, as negatives."""
    path = SC / (f"synth_hard_{i:05d}" if hard else f"synth_{i:05d}")
    if path.with_suffix(".json").exists(): return f"{path.name} exists"
    from shapely.geometry import shape, Point
    ga = shape([f for f in json.load(open(ROOT / "data/us_states.json"))["features"] if f["properties"]["name"] == "Georgia"][0]["geometry"])
    known = known_points(); rng = np.random.default_rng((200_000 if hard else 100_000) + i)
    minx, miny, maxx, maxy = ga.bounds
    empty = rng.random() < 0.2
    want_region = ("mountains", "piedmont", "coastal_plain")[i % 3]   # equal thirds
    for attempt in range(60):
        for _ in range(10000):
            lat, lon = float(rng.uniform(miny, maxy)), float(rng.uniform(minx, maxx))
            if region_of(lat, lon) == want_region and ga.contains(Point(lon, lat)) and min(km((lat, lon), k) for k in known) >= 3: break
        try:
            z, tr, epsg = fetch_dem(lat, lon, SIZE, res_m=RES)
            nlcd = fetch_nlcd(lat, lon, SIZE, RES)
        except Exception as e:
            print(f"{path.name} fetch failed ({e}); resampling", flush=True); time.sleep(5); continue
        nan = ~np.isfinite(z)
        if nan.mean() > 0.2: continue
        z = z.astype("float64"); zf = np.where(nan, np.nanmedian(z), z)
        modern = modern_mask(nlcd, RES); ok = suitability(z, RES) & ~modern
        inst = np.zeros(z.shape, "uint16"); instances = []
        if empty: break                      # plain ground of any kind, hilly or developed
        if ok.mean() < 0.08: continue        # too steep/developed for a mound: draw again
        drawn = [synth.random_mound(260, RES, rng) for _ in range(int(rng.integers(3, 7)))]
        radii = []
        for dz, m, p in drawn:
            ys, xs = np.nonzero(m); radii.append(max(int(np.hypot(np.ptp(xs), np.ptp(ys)) / 2) + 2, 8))
        pool = {}
        for r, d in zip(radii, drawn): pool.setdefault(r, []).append(d)
        near = None
        if hard:
            gy, gx = np.gradient(ndi.gaussian_filter(zf, 1.0), RES)
            steep = np.degrees(np.arctan(np.hypot(gx, gy))) > 15
            near = ndi.distance_transform_edt(~steep) if steep.any() else None
        bg = zf.copy()
        for row, col, r in choose_spots(ok, radii, rng, near=near, near_px=int(40 / RES), near_p=0.6 if hard else 0.0):
            dz, m, p = pool[r].pop()
            k = len(instances) + 1
            fx = insert_mound(zf, inst, k, row, col, dz, m, p, zf, rng, RES)
            instances.append(dict(id=k, cls="mound", kind=p["kind"], row=row, col=col, height=round(p["height"], 2),
                                  damaged=bool(p.get("damaged") or "cut" in fx), effects=fx, mask_quality="exact",
                                  area_m2=int((inst == k).sum() * RES * RES), params={a: b for a, b in p.items() if a != "kind"}))
        if instances: break
    else:
        return f"{path.name} FAILED"
    corridors = []
    if hard:
        if empty: bg = zf.copy()
        for k in instances:   # at most 2 cuts through/beside mounds (cuts only: fills next to mounds taught "fill = mound")
            if len(corridors) < 2 and rng.random() < 0.5:
                corridors.append(linear_corridor(zf, bg, inst, k["row"], k["col"], max(int(math.sqrt(k["area_m2"] / math.pi) / RES), 8), rng, RES, kind="cut"))
        for _ in range(int(rng.random() < 0.8) + int(rng.random() < 0.4)):   # 0-2 away from mounds: mostly fills, half ending in an abutment
            row, col = (int(v) for v in rng.integers(60, z.shape[0] - 60, 2))
            if inst[max(row - 60, 0):row + 60, max(col - 60, 0):col + 60].any(): continue
            corridors.append(dict(linear_corridor(zf, bg, inst, row, col, 20, rng, RES, kind="fill" if rng.random() < 0.6 else "cut"), random=True))
        for k in instances:   # a cut may have removed part of a mound: refresh its area, drop it if mostly gone
            k["area_m2"] = int((inst == k["id"]).sum() * RES * RES)
        if corridors and any(c["kind"] == "cut" for c in corridors):
            for k in instances:
                if k["area_m2"] < 50: inst[inst == k["id"]] = 0; k["removed_by_cut"] = True
    decoys = []
    for _ in range(int(rng.integers(1, 4))):
        row, col = (int(v) for v in rng.integers(40, z.shape[0] - 40, 2))
        decoys.append(dict(row=row, col=col, effects=insert_decoy(zf, row, col, zf, rng, RES)))
    forest = bool(rng.random() < 0.3)
    if forest: zf = synth.forest_sparse(zf, keep_frac=float(rng.uniform(0.05, 0.25)), rng=rng)
    zf[nan] = np.nan
    meta = dict(id=path.name, type="synth", lat=lat, lon=lon, epsg=epsg, transform=list(tr)[:6], res=RES,
                region=region_of(lat, lon), group=f"synth_{region_of(lat, lon)}", split="val" if i % 10 == 0 else "train",
                developed_frac=round(float(np.isin(nlcd, (21, 22, 23, 24)).mean()), 3), suitable_frac=round(float(ok.mean()), 3),
                forest=forest, empty=empty or not instances, instances=instances, decoys=decoys, focus=[],
                hard=hard, corridors=corridors)
    save_scene(path, zf, inst, np.zeros(z.shape, bool), meta)
    return f"{path.name} {meta['region']:13s} {len(instances)} mounds  dev {meta['developed_frac']:.0%}  {'forest' if forest else ''}"


def _hard(i): return synth_scene(i, hard=True)


def run_synth(n, workers, hard=False):
    from multiprocessing import Pool
    SC.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with Pool(workers) as p:
        for k, msg in enumerate(p.imap_unordered(_hard if hard else synth_scene, range(n)), 1):
            print(f"[{k}/{n} {time.time() - t0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- real
CORE_IGNORE = {"Etowah", "Ocmulgee", "Kolomoki"}   # other mounds there have no coordinates


def run_real():
    SC.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data/ga_mounds_verified.csv")))
    for r in rows: r["pt"] = (float(r["lat"]), float(r["lon"])); r["group"] = r["site"].replace(" 5km", "").strip()
    meas = {m["feature"]: m for m in csv.DictReader(open(ROOT / "data/mound_measurements.csv"))}
    # greedy scenes: mounds first, each feature owned by the first scene whose inner
    # 480 m box holds it, so no mound is a positive in two scenes
    order = sorted((r for r in rows if r["label"] in ("mound", "reconstruction", "lookalike")),
                   key=lambda r: r["label"] == "lookalike")
    inside = lambda c, pt, half: abs(pt[0] - c[0]) * 111.2 < half and abs(pt[1] - c[1]) * 111.2 * math.cos(math.radians(c[0])) < half
    centres = []
    for r in order:
        c = next((c for c in centres if inside(c["pt"], r["pt"], 0.24)), None)
        if c: c["members"].append(r)
        else: centres.append(dict(pt=r["pt"], members=[r]))
    owner = {r["feature"]: n for n, c in enumerate(centres) for r in c["members"]}
    core_mounds = [r["pt"] for r in rows if r["label"] == "mound" and r["group"] in CORE_IGNORE]
    for n, c in enumerate(centres):
        lat, lon = c["pt"]
        group = c["members"][0]["group"]; path = SC / f"real_{n:03d}_{group.replace(' ', '_').lower()}"
        z, tr, epsg = fetch_dem(lat, lon, SIZE, res_m=RES); z = z.astype("float64")
        inv = ~tr
        def px(pt):
            from pyproj import Transformer
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(pt[1], pt[0])
            col, row = inv * (x, y); return int(row), int(col)
        H, W = z.shape; yy, xx = np.mgrid[:H, :W]
        inst = np.zeros(z.shape, "uint16"); ignore = np.zeros(z.shape, bool); instances, focus = [], []
        core = any(km((lat, lon), p) < 1.0 for p in core_mounds)
        for r in rows:
            row, col = px(r["pt"])
            if not (0 <= row < H and 0 <= col < W): continue
            if r["label"] in ("mound", "reconstruction") and owner.get(r["feature"]) != n:
                ignore |= np.hypot(yy - row, xx - col) < 60      # a positive in another scene
            elif r["label"] in ("mound", "reconstruction"):
                py, pxx = summit_near(z, row, col)
                good = meas.get(r["feature"], {}).get("measurement_quality", "good") == "good"
                m, h = real_mound_mask(z, py, pxx)
                if m.any():   # a mound outline is compact; a sprawling one has leaked into terrain
                    from scipy.spatial import ConvexHull
                    ys_, xs_ = np.nonzero(m); solidity = m.sum() / max(ConvexHull(np.c_[xs_, ys_]).volume, 1)
                    good = good and solidity > 0.8
                if not good or not m.any():
                    m = np.hypot(yy - py, xx - pxx) < 15; q = "point"   # outline not separable (bluff/railroad): centre only
                else: q = "measured"
                k = len(instances) + 1; inst[m & (inst == 0)] = k
                instances.append(dict(id=k, cls="mound", kind="conical" if "conical" in r["type"] else "platform",
                                      feature=r["feature"], row=py, col=pxx, height=round(h, 1), damaged="damaged" in r["condition"],
                                      reconstruction=r["label"] == "reconstruction", weight=0.5 if r["label"] == "reconstruction" else 1.0,
                                      mask_quality=q, area_m2=int(m.sum())))
            elif r["label"] == "lookalike":
                focus.append(dict(feature=r["feature"], kind=r["type"], row=row, col=col))
            else:   # possible, effigy, earthwork, not_detected: neither mound nor background
                ignore |= np.hypot(yy - row, xx - col) < 60
        if core:   # unlisted mounds nearby: supervise only labelled mounds and look-alikes
            keep = inst > 0
            for f in focus: keep |= np.hypot(yy - f["row"], xx - f["col"]) < 40
            ignore |= ~ndi.binary_dilation(keep, iterations=10)
        meta = dict(id=path.name, type="real", lat=lat, lon=lon, epsg=epsg, transform=list(tr)[:6], res=RES,
                    region=region_of(lat, lon), group=group, split="real", background_supervised=not core,
                    instances=instances, focus=focus, decoys=[], empty=not instances)
        save_scene(path, z, inst, ignore, meta)
        print(f"{path.name:34s} mounds {len(instances)} ({', '.join(i['feature'][:18] + '/' + i['mask_quality'] for i in instances)})  look-alikes {len(focus)}  ignore {ignore.mean():.0%}")


# ---------------------------------------------------------------- out-of-state parks
def run_oos():
    """Training-only positives from public mound parks outside Georgia
    (data/out_of_state_mounds.csv, from scripts/scout_sites.py + a visual check).
    Other mounds there are unlisted, so everything except the listed mounds is
    ignore; these scenes are never used for scoring (Georgia is the test)."""
    from pyproj import Transformer
    from scipy.spatial import ConvexHull
    SC.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data/out_of_state_mounds.csv")))
    for r in rows: r["pt"] = (float(r["lat"]), float(r["lon"]))
    inside = lambda c, pt, half: abs(pt[0] - c[0]) * 111.2 < half and abs(pt[1] - c[1]) * 111.2 * math.cos(math.radians(c[0])) < half
    centres = []
    for r in rows:
        c = next((c for c in centres if c["site"] == r["site"] and inside(c["pt"], r["pt"], 0.22)), None)
        if c: c["members"].append(r)
        else: centres.append(dict(site=r["site"], state=r["state"], pt=r["pt"], members=[r]))
    for n, c in enumerate(centres):
        lat, lon = c["pt"]; path = SC / f"oos_{n:03d}_{c['site'].replace(' ', '_').lower()}"
        if path.with_suffix(".json").exists(): print(path.name, "exists"); continue
        z, tr, epsg = fetch_dem(lat, lon, SIZE, res_m=RES); z = z.astype("float64")
        to = Transformer.from_crs(4326, epsg, always_xy=True); inv = ~tr
        inst = np.zeros(z.shape, "uint16"); instances = []
        for r in c["members"]:
            col, row = inv * to.transform(r["pt"][1], r["pt"][0]); row, col = int(row), int(col)
            py, px = summit_near(z, row, col, 15)
            win = int(r["ground_win_m"]); m, h = real_mound_mask(z, py, px, ground_win=win, max_r=max(110, win // 2 + 30))
            q = "measured"
            if m.any():
                ys, xs = np.nonzero(m); sol = m.sum() / max(ConvexHull(np.c_[xs, ys]).volume, 1)
                if sol < 0.9: q = "point"     # leaked into ravine/bluff (cleanly separated mounds score >= 0.94)
            else: q = "point"
            if q == "point": m = np.hypot(*np.mgrid[:z.shape[0], :z.shape[1]] - np.array([py, px])[:, None, None]) < 15
            k = len(instances) + 1; inst[m & (inst == 0)] = k
            instances.append(dict(id=k, cls="mound", kind="platform", feature=f"{c['site']} {r['feature']}", row=py, col=px,
                                  height=round(h, 1), mask_quality=q, area_m2=int(m.sum()), weight=1.0))
        ignore = ~ndi.binary_dilation(inst > 0, iterations=10)
        meta = dict(id=path.name, type="real", lat=lat, lon=lon, epsg=epsg, transform=list(tr)[:6], res=RES, region="out_of_state",
                    group=f"oos_{c['site']}", split="train_only", background_supervised=False, state=c["state"],
                    instances=instances, focus=[], decoys=[], empty=not instances)
        save_scene(path, z, inst, ignore, meta)
        print(f"{path.name:32s} {', '.join(i['feature'].split(' ', 1)[1][:16] + '/' + i['mask_quality'] + f'/{i["height"]}m' for i in instances)}")


# ---------------------------------------------------------------- mined negatives
def run_mined(mdir):
    """Learn from mistakes: random Georgia areas (>= 3 km from known mound sites and from
    every Archaeohub site) where a trained model flagged something a person judged
    clearly modern or natural (decisions.json "negative"). Those flags become look-alike
    focus points with supervised background; flags judged "possible" are ignore discs
    (kept private, never trained on)."""
    mdir = Path(mdir); flags = json.load(open(mdir / "flags_indexed.json")); dec = json.load(open(mdir / "decisions.json"))
    neg, pos = set(dec["negative"]), set(dec.get("possible", []))
    by_area = {}
    for f in flags: by_area.setdefault(f["area"], []).append(f)
    made = 0
    for aid, fs in sorted(by_area.items()):
        if not any(f["id"] in neg for f in fs): continue
        a = np.load(mdir / f"{aid}.npz"); z = a["z"].astype("float64"); H, W = z.shape; yy, xx = np.mgrid[:H, :W]
        ignore = np.zeros(z.shape, bool); focus = []
        for f in fs:
            row, col = int(f["row_grid"] * DAVIS_RES_M), int(f["col_grid"] * DAVIS_RES_M)
            if f["id"] in neg: focus.append(dict(feature=f"mined flag #{f['id']} ({f['score']:.2f})", kind="mined negative", row=row, col=col))
            else: ignore |= np.hypot(yy - row, xx - col) < 60      # possible or unjudged: neither mound nor background
        f0 = fs[0]; path = SC / f"mined_{aid}"
        meta = dict(id=path.name, type="real", lat=f0["lat"], lon=f0["lon"], epsg=f0["epsg"], transform=f0["transform"], res=RES,
                    region=f0["region"], group="mined_negatives", split="train_only", background_supervised=True,
                    instances=[], focus=focus, decoys=[], empty=True)
        save_scene(path, z, np.zeros(z.shape, "uint16"), ignore, meta); made += 1
    print(f"{made} mined scenes, {len(neg)} negatives, {len(pos)} possibles kept out")


# ---------------------------------------------------------------- manifest + preview
def run_manifest():
    from PIL import Image, ImageDraw
    metas = [json.loads(p.read_text()) for p in sorted(SC.glob("*.json"))]
    davis = {}
    if DAVIS_TILES.exists():
        davis = dict(path=str(DAVIS_TILES), images=len(list((DAVIS_TILES / "images").glob("*.tif"))),
                     labels={d.name: len(list(d.glob("*.tif"))) for d in sorted((DAVIS_TILES / "labels").iterdir()) if d.is_dir()},
                     note="Davis band recipe only (hillshade/slope/DEM uint8 at 1.524 m); no raw DEM")
    syn = [m for m in metas if m["type"] == "synth"]; real = [m for m in metas if m["type"] == "real"]
    kinds = {}
    for m in syn:
        for i in m["instances"]: kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    summary = dict(
        synthetic=dict(scenes=len(syn), train=sum(m["split"] == "train" for m in syn), val=sum(m["split"] == "val" for m in syn),
                       empty=sum(m["empty"] for m in syn), mounds=sum(len(m["instances"]) for m in syn), by_kind=kinds,
                       by_region={r: sum(m["region"] == r for m in syn) for r in ("mountains", "piedmont", "coastal_plain")},
                       forest=sum(m["forest"] for m in syn)),
        real=dict(scenes=len(real), groups=sorted({m["group"] for m in real}),
                  mounds=[i["feature"] for m in real for i in m["instances"] if not i.get("reconstruction")],
                  reconstructions=[i["feature"] for m in real for i in m["instances"] if i.get("reconstruction")],
                  lookalikes=sum(len(m["focus"]) for m in real)),
        davis_sc=davis)
    (OUT / "manifest.json").write_text(json.dumps(dict(summary=summary, scenes=[
        dict(id=m["id"], type=m["type"], group=m["group"], split=m["split"], region=m["region"], n=len(m["instances"])) for m in metas]), indent=1))
    print(json.dumps(summary, indent=1))
    # preview: 8 synthetic + all real, hillshade with red outlines, yellow = ignore, cyan + = look-alike
    pick = [m for m in syn if not m["empty"]][:8] + real
    T = 220; cols = 6; sheet = Image.new("RGB", (T * cols, T * math.ceil(len(pick) / cols)), "white")
    for k, m in enumerate(pick):
        z, inst, ign, _ = load_scene(SC / m["id"])
        hs = hillshade(np.where(np.isfinite(z), z, np.nanmedian(z)), RES)[0]
        rgb = np.stack([hs] * 3, -1)
        rgb[ign] = rgb[ign] * 0.6 + np.array([255, 220, 0]) * 0.4
        rgb[(inst > 0) & ~ndi.binary_erosion(inst > 0, iterations=2)] = (255, 30, 30)
        im = Image.fromarray(rgb.astype("uint8")); d = ImageDraw.Draw(im)
        for f in m["focus"]: d.line([(f["col"] - 12, f["row"]), (f["col"] + 12, f["row"])], fill=(0, 255, 255), width=4); d.line([(f["col"], f["row"] - 12), (f["col"], f["row"] + 12)], fill=(0, 255, 255), width=4)
        im = im.resize((T, T)); ImageDraw.Draw(im).text((3, 3), m["id"][:28], fill=(255, 255, 0))
        sheet.paste(im, ((k % cols) * T, (k // cols) * T))
    sheet.save(OUT / "preview.png"); print("wrote", OUT / "preview.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("synth", "synth-hard"): run_synth(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6, hard=cmd == "synth-hard")
    elif cmd == "real": run_real()
    elif cmd == "oos": run_oos()
    elif cmd == "mined": run_mined(sys.argv[2])
    elif cmd == "manifest": run_manifest()
