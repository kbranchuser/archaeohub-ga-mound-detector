# Using the mound detector

This guide is for anyone who wants to **run** the Georgia mound detector on their own
area. You don't need to train anything. (To train or evaluate, see `README.md`.)

You give it a spot on the map. It downloads free USGS LiDAR elevation for a square
around that spot, turns it into the relief views archaeologists read by eye, and flags
compact rises that look like human-made mounds.

> **A flag is a lead for fieldwork, not a discovery.** Please read
> [Responsible use](#responsible-use) before sharing any results.

---

## What it's good at (and not)

- **Tested on:** 18 known mounds in Georgia, each scored by a version of the model that
  never saw that site. It found **14 of 18 (78%)** and wrongly flagged **2 of 18**
  look-alike places (a bridge abutment and a rocky knoll).
- **Works best on:** platform and conical mounds, roughly 1 m high or more, in the
  Southeastern US.
- **Common false alarms:** graded house pads, road and railroad fills, dam embankments,
  spoil piles, and the ends of natural ridges. Each spot is tagged with its land cover
  and distance to the nearest building to help you spot these.
- **Weaker on:** very low or plowed-down mounds, mounds cut by railroads or roads, and
  terrain unlike the Southeast (mountains in the West, glaciated New England).
- **Coverage:** anywhere in the lower 48 states with USGS 3DEP 1 m LiDAR (most of the
  country). It needs an internet connection to download the elevation data.

---

## 1. What you need

- **A computer** running macOS, Linux or Windows. A Mac with Apple silicon or a computer
  with an NVIDIA graphics card is faster, but a normal CPU works.
- **Python 3.10 or newer** (tested on 3.14).
- **About 2 GB of disk space** for the software and the two model files.

## 2. Install

```bash
git clone https://github.com/kbranchuser/archaeohub-ga-mound-detector.git
cd archaeohub-ga-mound-detector
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\python` in place of `.venv/bin/...`.

## 3. Get the two model files

The detector needs two files:

**a) The base model (Davis et al. 2021), free under CC BY 4.0.** The detector is built
on top of it.

1. In a web browser, open https://doi.org/10.26207/ry6k-q463 and download
   `Deep_Learning_Files.zip`. (The site blocks scripted downloads, so use a browser.)
2. Unzip it. Inside `SC_Deep_Learning_Supplemental_Files/` is `SC_Arch_RCNN.dlpk`. That
   is a **7-Zip** archive, not a normal zip. Open it with 7-Zip (Windows),
   The Unarchiver (Mac) or `7z x SC_Arch_RCNN.dlpk` (Linux).
3. Copy `RCNN_2_Output.pth` from it to:
   ```
   models/davis2021/RCNN_2_Output.pth
   ```

**b) The trained Georgia model** (`ckpt_final.pt`, about 175 MB). Download it from the
[Releases page](https://github.com/kbranchuser/archaeohub-ga-mound-detector/releases)
(the newest release has the newest model) and put it at:

```
runs/final/all/ckpt_final.pt
```

## 4. Run it

**One spot.** Give the centre of the square as `latitude,longitude` and its size in
metres (500 to 3,000 works well):

```bash
.venv/bin/python scripts/detect.py --point 34.1250,-84.8075 --size 800 --name etowah --out results/etowah
```

That example is Etowah Indian Mounds State Historic Site, a public park, and a good
first test. You should get 4 spots, including the three large mounds.

**Many spots.** Make a CSV file with `name,lat,lon` columns (and optionally `size_m`),
then:

```bash
.venv/bin/python scripts/detect.py --csv my_places.csv --out results
```

Each place gets its own folder inside `results/`.

**How long it takes:** roughly 10 seconds to 2 minutes per square kilometre. Most of
that is downloading elevation from USGS, which can be slow or briefly unavailable. If
a download fails, just run the command again.

## 5. Read the results

Each run makes a folder like this:

| File | What it is |
|---|---|
| `spots.csv` | One row per flagged spot (open it in Excel or Google Sheets) |
| `spots.geojson` | The same spots as map points, for QGIS, ArcGIS or geojson.io |
| `spots/spot_N.png` | A close-up of each spot (raised ground warm, cuts cool, 50 m scale bar) |
| `views/*.png` | 11 relief views of the whole square |
| `run.json` | Which model ran, the square's exact corners, run time |

Columns in `spots.csv`:

| Column | Meaning |
|---|---|
| `score` | Model confidence, 0.5 to 1.0. Above 0.8 is strong, 0.65 to 0.8 medium, below that weak. |
| `height_m` | How far the rise stands above the surrounding ground (reads high on slopes) |
| `width_m` | Rough width across |
| `landcover` | National land-cover class at the spot (NLCD 2021), such as "pasture / hay" or "developed, low intensity" |
| `building_m` | Distance to the nearest building, in metres. Under about 30 m often means a house pad. |
| `models_agree` | When you pass several `--ckpt` files: how many of the models flagged this spot |

**Checking a spot:** open its close-up and the views. A mound usually reads as a compact,
round or squared rise with even sides that ignores the slope of the land. Then look at
the same place on satellite imagery to rule out houses, roads and ponds.

## 6. Troubleshooting

- **"No model found":** check the file is at `runs/final/all/ckpt_final.pt` (step 3b).
- **Error loading `RCNN_2_Output.pth`:** check it's in `models/davis2021/` (step 3a).
- **"No 1 m LiDAR elevation covers most of this square":** USGS has no 1 m data there yet.
  Try a nearby spot or check coverage at https://apps.nationalmap.gov/lidar-explorer/.
- **Timeouts or HTTP 5xx errors:** USGS's elevation service is busy. Wait a minute and
  run again.
- **Very slow:** use a smaller `--size`. A Mac with Apple silicon or an NVIDIA card is
  picked up automatically.

---

## Responsible use

Unrecorded archaeological sites, and especially burials, are easily damaged by looting
and curiosity. Please:

- **Never post the locations of flagged spots publicly**, including on social media,
  shared maps or in screenshots with coordinates.
- **Don't dig or collect.** In most states, disturbing archaeological sites or human
  remains is illegal, and on private land you need the owner's permission even to walk it.
- **Report real finds to your state's archaeological site files.** In Georgia, that's
  the Georgia Archaeological Site File at the University of Georgia. Contact the State
  Historic Preservation Office (SHPO) or a professional archaeologist.
- **Respect descendant communities.** Many of these places are sacred to Native nations
  today.

## Credit

Built on the model and training data of **Dylan S. Davis, Gino Caspari, Carl P. Lipo and
Matthew C. Sanger** (CC BY 4.0): Davis et al. (2021), *Deep learning reveals extent of
Archaic Native American shell-ring building practices*, Journal of Archaeological Science
132: 105433, https://doi.org/10.1016/j.jas.2021.105433. Data:
https://doi.org/10.26207/ry6k-q463.

Elevation: USGS 3D Elevation Program. Land cover: NLCD 2021 (MRLC). Building outlines:
Microsoft Global ML Building Footprints (ODbL).

Detector and Georgia training by Kelob Branch.
