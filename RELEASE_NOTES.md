# Release notes

## v1.0.1 (November 2026): fewer house-pad false alarms

**The model**
- Retrained with 3,012 real house pads as "not a mound" examples. Bare-earth LiDAR removes
  the house but keeps its graded pad, which was the commonest false alarm in v1.0. The
  pads come from 240 scenes centred on rural and suburban Georgia buildings (Microsoft
  building footprints), kept at least 3 km from every known mound site.
- In real use on 15 private scan squares, spots within 30 m of a building fell from 13 to 5.
  Spots away from buildings also changed (4 dropped, 6 new, none field-checked yet), so
  expect a v1.0.1 scan of the same square to differ somewhat from a v1.0 scan.
- Everything else (architecture, relief inputs, scan settings) is unchanged.

**Testing**
- The test list grew from 18 to 21 known Georgia mounds: Kolomoki Mound E, Kolomoki
  Mound F and Ocmulgee's McDougal Mound, all on public land and included in
  `data/ga_mounds_verified.csv`.
- v1.0.1 finds **16 of 21 (76%)** in leave-one-site-out testing and wrongly flags 2 of 18
  look-alikes. The original Davis et al. model finds 7 of 21.
- On the original 18 mounds the score is the same as v1.0: 14 of 18. Of the three new
  mounds it finds Kolomoki E and F and misses McDougal.

**New scripts**
- `scripts/build_house_negatives.py` builds the house-pad training scenes;
  `aws/jobs_houses.txt` is the training job that uses them.
- `scripts/check_candidates.py` makes a tight LiDAR view of each published mound location
  in a CSV, for checking by eye before adding a mound to a test list.

**Upgrading**
- Download the new `ckpt_final.pt` and replace the old file at
  `runs/final/all/ckpt_final.pt`. No other setup changes.

**Download:** `ckpt_final.pt` from the
[v1.0.1 release](https://github.com/kbranchuser/archaeohub-ga-mound-detector/releases/tag/v1.0.1).

## v1.0 (November 2026): first public release

The first open-source version of the Georgia LiDAR mound detector.

**The model**
- Mask R-CNN built on the Davis et al. (2021) South Carolina model, converted to run in
  open-source PyTorch (no ArcGIS needed).
- Reads six relief layers computed from USGS 3DEP 1 m elevation instead of raw elevation,
  so it carries over between regions: local relief, positive and negative openness, slope,
  a landform index and directional sky view.
- Trained on synthetic mounds placed in real Georgia terrain, real Georgia and Southeastern
  park mounds, common look-alikes (road and railroad fills, bluff edges, bridge abutments,
  graded pads) and reviewed false alarms from random Georgia ground.
- Finds 14 of 18 known Georgia mounds (78%) in leave-one-site-out testing, against 5 of 18
  for the original model. Wrongly flags 2 of 18 look-alike places.

**Scanning**
- `scripts/detect.py` scans any square in the lower 48 states and saves a spot list (CSV),
  map points (GeoJSON), a close-up of each spot and 11 relief views.
- Each scan downloads a 120 m border around the square so relief layers are correct right
  up to the edges, then drops any spot in that border.
- Each spot is tagged with its land cover (NLCD 2021) and the distance to the nearest
  building (Microsoft footprints), to help rule out house pads.

**Also included**
- Setup guide ([USING_THE_MODEL.md](USING_THE_MODEL.md)), model card
  ([MODEL_CARD.md](MODEL_CARD.md)) and a public test list of mounds and look-alikes on
  public land.
- Training, evaluation and optional AWS scripts (bring your own AWS account).

**Known limits**
- Graded ground (house pads, road fills, dams) is still the commonest false alarm.
- Low, plowed-down or railroad-cut mounds are often missed.
- Untested outside the Southeastern US.

**Download:** `ckpt_final.pt` from the
[v1.0 release](https://github.com/kbranchuser/archaeohub-ga-mound-detector/releases/tag/v1.0).
