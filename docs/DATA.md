# Data

Nothing in this file's list is committed to git. It's large, third-party, or can reveal site locations.

## Davis et al. 2021 model + training tiles (CC BY 4.0)

Download from https://doi.org/10.26207/ry6k-q463. Penn State's ScholarSphere uses a
browser check, so download in a browser. You only need:

- `Deep_Learning_Files.zip` → unzips to `SC_Deep_Learning_Supplemental_Files/`
  - `SC_Arch_RCNN.dlpk` is a **7-Zip** archive (not a zip). Extract `RCNN_2_Output.pth`
    and `RCNN_2_Output.emd` into `models/davis2021/`.
  - `Training_Data/images/*.tif`: 200×200 px chips, 4 bands uint8, in the order **hillshade, slope, DEM, alpha**.
    (His README says DEM/hillshade/slope; visual inspection of the chips shows otherwise.)
    The DEM band was stretched over the whole county, so one chip spans only ~43 levels, **5 ft (1.524 m) pixels**, EPSG:3656 (NAD83 / South Carolina ft).
  - `Training_Data/labels/{Modern,Mound,Ring}/*.tif`: per-class masks.
- `README.md`

Not needed: `BUCH_comp.tif`, `*_LIDAR.zip` (his SC input mosaics), `SC_RF_*.tif.zip`
(Sentinel radar/optical composite for his random-forest cross-check).

### Model facts (verified)
- torchvision `maskrcnn_resnet50_fpn`, 4 classes (background, Modern, Mound, Ring), 494 tensors.
- Legacy (pre-1.6) torch pickle `{'model', 'opt'}`. It only references OrderedDict, numpy
  scalar/dtype, `codecs.encode` and torch tensor rebuilders (checked with `pickletools`, no
  execution). Loaded with `weights_only=True` plus that explicit allowlist (`src/gamd/davis.py`).
- **Input scaling:** bands divided by 255, then torchvision's built-in ImageNet
  normalisation. Raw 0-255 with normalisation disabled finds nothing.
- `min_size=max_size=200` so torchvision doesn't upscale the chips.

## Georgia LiDAR
USGS 3DEP (e.g. *GA Statewide B2 2018*). Exported from Archaeohub renders or pulled
directly. Kept under `data/` (ignored).

## Known-site coordinates (published, not DINAA)
- Etowah Mound A: 34.1250, -84.8075 (Etowah Indian Mounds State Historic Site). DINAA coordinates are deliberately coarsened, so don't use them.

## Training set v1 (step 6.5)
`python scripts/build_training_set.py real|synth N|manifest` writes `data/training/v1/`:
- `scenes/*.npz`: `z` (float32 elevation, m, 1 m pixels, synthetic mounds inserted),
  `inst` (uint16 instance id per pixel), `ignore` (bool: neither mound nor background).
- `scenes/*.json`: location, region, `group`, `split`, instances (kind, height, damage,
  realism effects, `mask_quality` = exact / measured / point), look-alike `focus` points.
- `manifest.json`: per-scene index + summary; `preview.png`: contact sheet.

Rules that matter at training time:
- Real scenes at Etowah, Ocmulgee and Kolomoki (within 1 km) have other, unlisted mounds,
  so everything except the labelled mounds and look-alikes is `ignore`.
- `mask_quality == "point"` (Ocmulgee's three, Earth Lodge): outline not separable from the
  bluff/railroad. Use for detection scoring and box/centre loss only, not mask loss.
- Reconstructions carry `weight: 0.5`. Effigies, possibles and undetected sites are ignore discs.
- Hold out by `group` (site), never by scene, so the score reflects unseen sites.

## Out-of-state park mounds (training only)
`data/out_of_state_mounds.csv` (30 mounds, 9 public parks: Moundville AL, Winterville MS, Toltec AR,
Parkin AR, Pinson TN, Lake Jackson FL, Letchworth-Love FL, Kincaid IL, Angel IN). Found with
`scripts/scout_sites.py` (compact rises above a 141 m top-hat), each checked by eye on a zoomed
hillshade (`scripts/scout_zoom.py`); ravine knobs, riverbank features, museum pads and anything
uncertain were left out. Dropped: Shiloh (outlines run into the dissected bluff), Chucalissa (no
mound at the coordinate used), Emerald (platform is a reshaped natural hill; edge ambiguous).
`build_training_set.py oos` makes 16 scenes; outlines with solidity < 0.9 become centre-only
(9 of 30). Everything else in those scenes is ignore. Never used for scoring: Georgia is the test.
