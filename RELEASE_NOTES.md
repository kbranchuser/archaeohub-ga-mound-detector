# Release notes

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
