# archaeohub-ga-mound-detector

Finding Native American earthen mounds in airborne LiDAR with deep learning, trained on
Georgia. This is the open-source release of the detector behind the private
"Detections" tab in [Archaeohub](https://archaeohub.org).

- **Just want to run it on your own area?** → [USING_THE_MODEL.md](USING_THE_MODEL.md)
- **What it's good at, its limits and its test record** → [MODEL_CARD.md](MODEL_CARD.md)
- **What changed in each version** → [RELEASE_NOTES.md](RELEASE_NOTES.md)

It starts from the model of **Dylan S. Davis, Gino Caspari, Carl P. Lipo and Matthew C. Sanger**
([Davis et al. 2021](https://doi.org/10.1016/j.jas.2021.105433); [model and data](https://doi.org/10.26207/ry6k-q463), CC BY 4.0),
a Mask R-CNN trained on South Carolina shell rings and mounds, converted here to run with open-source PyTorch (no ArcGIS). It then adds
relief inputs that carry over between regions, artificial training mounds placed in real
Georgia terrain, look-alike training (road fills, bluffs, abutments, house pads),
out-of-state park mounds, hard-negative mining, and honest leave-one-site-out testing.

On 18 known Georgia mounds, each scored by a version that never saw that site, it finds
**14 of 18 (78%)**; the original model found 5 of 18.

> **Site-protection rule.** This project publishes models and methods, **never the
> locations of possible sites**. Please don't post coordinates of anything the detector
> flags; report real finds to your state's archaeological site files. `data/`,
> `detections/` and `outputs/` are git-ignored on purpose (except the public test list).

### What's in this public release

- All code: data preparation, synthetic mounds, training, evaluation, scanning
  (`scripts/detect.py`), and AWS scripts (bring your own AWS account; see `aws/launch.sh`).
- **Public test list** (`data/ga_mounds_verified.csv`): the test mounds and look-alikes on
  **public land only** (Etowah, Ocmulgee and its Lamar unit, Kolomoki, Nacoochee, Rock
  Eagle, Rock Hawk). The 8 Rood's Landing mounds and sites on private land are withheld,
  so this list covers 10 of the 18 test mounds (the model finds 8 of those 10).
- **Not included:** the training scenes and mined negatives (they contain locations of
  ordinary private land), and the trained weights, which are published as release assets.

Updates: the author keeps improving the model privately (reviews from real use feed each
retraining round) and publishes new versions here as releases.

## Results

| Model | Known Georgia mounds found | Look-alikes wrongly flagged |
|---|---|---|
| Davis et al. (2021), unmodified | 5 of 18 (28%) | 2 of 18 |
| **This model (v1.0)** | **14 of 18 (78%)** | 2 of 18 |

Each site was scored by a version of the model trained without it (leave-one-site-out).
Details, limits and the public test subset are in [MODEL_CARD.md](MODEL_CARD.md); what
changed in each version is in [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the two model files in place (see USING_THE_MODEL.md, step 3), then:
.venv/bin/python scripts/detect.py --point 34.1250,-84.8075 --size 800 --out results/etowah
```

Full setup, outputs and troubleshooting: [USING_THE_MODEL.md](USING_THE_MODEL.md).

## Credit and license

The base model and training data are by **Dylan S. Davis, Gino Caspari, Carl P. Lipo and
Matthew C. Sanger**, released under **CC BY 4.0**:

- Davis, D. S., Caspari, G., Lipo, C. P., & Sanger, M. C. (2021). Deep learning reveals extent
  of Archaic Native American shell-ring building practices. *Journal of Archaeological
  Science* 132: 105433. https://doi.org/10.1016/j.jas.2021.105433
- Data and model: https://doi.org/10.26207/ry6k-q463

**This repository's code:** MIT License, © 2026 Kelob Branch (see [LICENSE](LICENSE)).
**Trained model weights** (release assets): CC BY 4.0, as a derivative of the Davis et al.
model; credit both Davis et al. (2021) and this project.

Data sources used at run time: USGS 3D Elevation Program (public domain), NLCD 2021 land
cover (MRLC, public domain), Microsoft Global ML Building Footprints (ODbL).
