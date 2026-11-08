"""Step 8: fine-tune Davis's Mask R-CNN on Georgia training scenes.

    python scripts/train.py OUT_DIR [--iters N] [--init CKPT] [--fold K] [--real-frac F] [--recipe R] [--workers W]

Stage A (no --fold): synthetic scenes only, starting from Davis's weights, so the
real Georgia features stay a completely unseen test.
Final model (--fold -1): stage B with every Georgia site in training (no honest
score; eval numbers are on training sites).
Stage B (--fold 0-3): also real mounds and look-alikes, holding out one fold of
sites (FOLDS). Each fold is scored only on its own held-out sites, so the four
folds together give an honest score for every verified feature.
Every EVAL_EVERY iterations: checkpoint + real-site score + synthetic val recall.
"""
import os, sys, json, math, time, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamd.davis import build_model, adapt_channels
from gamd.bands import RECIPES
from gamd.train_data import Scenes, sample_chip, PreScenes, sample_chip_pre, PRE_BANDS
from gamd.ignore import enable_ignore
from eval_real import evaluate, summary

# each mound site is paired with look-alike-only sites so every group is held out once
FOLDS = [["Etowah", "Leake", "Long Swamp"], ["Ocmulgee", "Shoulderbone"], ["Kolomoki", "Hoojah", "Roods Landing"], ["Lamar", "Nacoochee"]]   # Roods with Kolomoki: both SW Georgia

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--init")
ap.add_argument("--fold", type=int); ap.add_argument("--real-frac", type=float, default=0.35)
ap.add_argument("--lr", type=float, default=0.005); ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--recipe", default="davis", choices=list(RECIPES))
ap.add_argument("--pre", action="store_true", help="use precomputed bands (scripts/precompute_bands.py); relief recipes only")
ap.add_argument("--workers", type=int, default=0, help="chip-making processes (Linux; on macOS they copy all scenes)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data/training/v1/scenes"
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
ITERS, BATCH, LR, WARMUP = args.iters, 4, args.lr, min(200, args.iters // 10)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
held = ([] if args.fold == -1 else FOLDS[args.fold]) if args.fold is not None else None

metas = {p.stem: json.loads(p.read_text()) for p in SC.glob("*.json")}
train_ids = sorted(i for i, m in metas.items() if m["type"] == "synth" and m["split"] == "train")
val_ids = sorted(i for i, m in metas.items() if m["type"] == "synth" and m["split"] == "val")
real_ids = sorted(i for i, m in metas.items() if m["type"] == "real" and held is not None and m["group"] not in held)
print(f"device {dev}; {len(train_ids)} synthetic train / {len(val_ids)} val; {len(real_ids)} real scenes"
      + (f" (held out: {', '.join(held) or 'none'})" if held is not None else ""), flush=True)
if args.pre:
    assert args.recipe in PRE_BANDS, "--pre needs a relief recipe"
    Load, SC_LOAD = PreScenes, ROOT / "data/training/v1/pre"
    chip_fn = lambda sc, r, **kw: sample_chip_pre(sc, r, recipe=args.recipe, **kw)
else:
    Load, SC_LOAD = Scenes, SC
    chip_fn = lambda sc, r, **kw: sample_chip(sc, r, recipe=args.recipe, **kw)
train, val = Load(SC_LOAD, train_ids), Load(SC_LOAD, val_ids)
real = Load(SC_LOAD, real_ids) if real_ids else None

model = adapt_channels(build_model(ROOT / "models/davis2021/RCNN_2_Output.pth", dev), RECIPES[args.recipe])
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=True)["model"])
model = enable_ignore(model).train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
lr_at = lambda it: LR * (it / WARMUP if it < WARMUP else 0.5 * (1 + math.cos(math.pi * (it - WARMUP) / max(ITERS - WARMUP, 1))))
SEED = 0 if args.fold is None else 10 + args.fold
_kind = lambda i: i.split("_")[0]   # real / oos / mined
buckets = [[k for k, i in enumerate(real_ids) if _kind(i) == kd] for kd in sorted({_kind(i) for i in real_ids})] if real_ids else []


class ChipStream(torch.utils.data.IterableDataset):
    """Endless random chips; each worker gets its own random stream."""
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(SEED * 100 + (wi.id + 1 if wi else 0))
        while True:
            if real and rng.random() < args.real_frac:   # equal share per kind: Georgia sites / out-of-state parks / mined negatives
                b = buckets[int(rng.integers(len(buckets)))]
                x, t, ig = chip_fn(real.items[b[int(rng.integers(len(b)))]], rng, focus_p=0.8)
            else:
                x, t, ig = chip_fn(train.items[int(rng.integers(len(train.items)))], rng)
            t["ignore"] = torch.from_numpy(ig)
            yield x, t


def collate(batch):
    return tuple(zip(*batch))


chips = iter(torch.utils.data.DataLoader(ChipStream(), batch_size=BATCH, num_workers=args.workers, collate_fn=collate,
                                         persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None))


def val_synth(model, n=200):
    """Recall of synthetic mounds and false detections per chip on held-out scenes."""
    model.eval(); r = np.random.default_rng(1); hit = tot = fp = 0
    with torch.no_grad():
        for _ in range(n):
            x, t, _ = chip_fn(val.items[int(r.integers(len(val.items)))], r)
            p = model([x.to(dev)])[0]; keep = p["scores"].cpu() >= 0.5
            pb = p["boxes"].cpu()[keep & (p["labels"].cpu() >= 2)]
            pc = torch.stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2], 1) if len(pb) else torch.zeros((0, 2))
            matched = set()
            for b in t["boxes"]:
                tot += 1; c = torch.tensor([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                d = (pc - c).norm(dim=1) if len(pc) else torch.tensor([])
                j = int(d.argmin()) if len(d) else -1
                if j >= 0 and d[j] * 1.524 < 20: hit += 1; matched.add(j)
            fp += len(pc) - len(matched)
    model.train()
    return dict(synth_recall=round(hit / max(tot, 1), 3), synth_fp_per_chip=round(fp / n, 3))


log = open(out / "log.jsonl", "a"); t0 = time.time()
for it in range(1, ITERS + 1):
    for g in opt.param_groups: g["lr"] = lr_at(it)
    xb, tb = next(chips)
    xs = [x.to(dev) for x in xb]; ts = [{k: v.to(dev) for k, v in t.items()} for t in tb]
    losses = model(xs, ts); loss = sum(losses.values())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step()
    if it % 50 == 0 or it <= 3:
        print(f"it {it} loss {loss.item():.3f} " + " ".join(f"{k[5:]}={v.item():.3f}" for k, v in losses.items()) + f"  {time.time() - t0:.0f}s", flush=True)
    if it % args.eval_every == 0 or it == ITERS:
        ck = dict(model=model.state_dict(), it=it, fold=args.fold, recipe=args.recipe)
        torch.save(ck, out / f"ckpt_{it}.pt")
        if it == ITERS: torch.save(ck, out / "ckpt_final.pt")
        model.eval()
        ks = (10.0, 20.0) if args.recipe == "davis" else (10.0,)     # DEM stretch only matters for davis
        per = {k: evaluate(model, dev, k, groups=held or None, recipe=args.recipe) for k in ks}
        model.train()
        rec = dict(it=it, fold=args.fold, held=held, recipe=args.recipe, **val_synth(model),
                   **{f"real_k{int(k)}": summary(v) for k, v in per.items()}, secs=round(time.time() - t0))
        print("EVAL", json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        json.dump(per[10.0], open(out / f"features_{it}.json", "w"), indent=1)
