# Model card: archaeohub-ga-mound-detector

## Summary

Finds possible earthen mounds (platform and conical) in bare-earth airborne LiDAR.
Given a square of USGS 3DEP 1 m elevation, it returns flagged spots with a confidence
score, an estimated height and width, the land cover, and the distance to the nearest
building.

| | |
|---|---|
| **Architecture** | torchvision Mask R-CNN (ResNet-50 FPN), first layer widened to 6 input bands |
| **Starting point** | Davis et al. (2021) South Carolina mound and shell-ring model (CC BY 4.0) |
| **Input** | 200 × 200 px chips at 1.524 m (5 ft) per pixel, scanned with 50% overlap |
| **Input bands ("relief6")** | local relief (141 m top-hat), positive openness, slope, negative openness, geomorphon landform index, anisotropic sky-view factor |
| **Output** | Boxes, masks and scores for Mound, Ring and Modern classes; spots with score ≥ 0.5 are reported |
| **Weights license** | CC BY 4.0 (credit Davis et al. 2021 and this project) |
| **Code license** | MIT |

## Intended use

- A **first pass** over LiDAR to point an archaeologist, survey crew or researcher to places
  worth a closer look, on the map and then in the field with permission.
- Research and teaching about LiDAR feature detection.

**Not intended for:** confirming that a site exists, publishing site locations,
treasure-hunting or digging, or any decision about land development without professional
archaeological review.

## Training data

- **Synthetic mounds:** 1,500 scenes of real Georgia terrain (mountains, Piedmont, coastal
  plain), with mounds whose sizes and shapes were measured from real Georgia mounds. A third
  are "hard" scenes with bluff edges, road and railroad cuts and fills, and abutments.
- **Real Georgia mounds and look-alikes** from published locations (state and national
  parks, NRHP records), with leave-one-site-out grouping.
- **Out-of-state park mounds:** 30 mounds from public parks in AL, MS, AR, TN, FL, IL and
  IN, used for training only.
- **Hard negatives:** about 277 false flags from 400 random Georgia areas, each reviewed by a
  person.
- Developed land was masked out of synthetic placement using NLCD 2021.

Site-file (DINAA) coordinates were not used. The training scenes are not published, because
the random areas are ordinary private land.

## Evaluation

**Protocol:** a version of the model trained without a site is scored on that site
(leave-one-site-out, 4 folds). A hit is a Mound or Ring detection with score ≥ 0.5 within
40 m of the feature.

| Model | Mounds found (of 18) | Look-alikes flagged (of 18) |
|---|---|---|
| Davis et al. (2021), unmodified | 5 (28%) | 2 |
| **This model** | **14 (78%)** | **2** |

- **Found:** Etowah A, B, C; Ocmulgee Great Temple; Kolomoki A and a conical mound;
  6 of 8 Rood's Landing mounds (a site never seen in training); Lamar A and B.
- **Missed:** Ocmulgee Lesser Temple and Funeral Mound (both cut by a railroad), and two
  small Rood's Landing platforms.
- **False alarms:** an I-16 bridge abutment (0.82) and a rocky knoll in Helen (0.57).
- **Public test list:** the published `data/ga_mounds_verified.csv` keeps only public-land
  sites: 10 of the 18 mounds, of which the model finds 8.

18 test mounds is a small number; treat the percentages as rough.

## Known limitations

- **Graded ground looks like platforms.** House pads, lots, road and railroad fills, dam
  embankments and spoil piles are the commonest false alarms. Spots are tagged with land
  cover and nearest-building distance to help, but they are not hidden.
- **Low or damaged mounds** (plowed down, cut by roads or railroads, under 1 m) are often
  missed.
- **Region:** trained on the Southeastern US. Elsewhere (western mountains, glaciated
  terrain) results are untested and likely worse.
- **Height estimates** read high on slopes.
- **Data:** depends on USGS 3DEP 1 m coverage and the quality of its ground classification.
  Dense vegetation or old LiDAR collections reduce detail.

## Ethical considerations

Unrecorded mounds and burials are vulnerable to looting and disturbance, and many are
sacred to Native nations today. Flagged locations should not be shared publicly. Real finds
should be reported to the state's archaeological site files or State Historic Preservation
Office, and descendant communities consulted where appropriate. Disturbing archaeological
sites or human remains is illegal in most states.

## Citation

Branch, K. (2026). *archaeohub-ga-mound-detector: LiDAR mound detection for Georgia.*
https://github.com/kbranchuser/archaeohub-ga-mound-detector

Davis, D. S., Caspari, G., Lipo, C. P., & Sanger, M. C. (2021). Deep learning reveals extent
of Archaic Native American shell-ring building practices. *Journal of Archaeological
Science* 132: 105433. https://doi.org/10.1016/j.jas.2021.105433
