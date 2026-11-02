# Progress: potato in place of sugarcane, better cotton and soybean

Status on 2 Nov 2026. Training was stopped early on request, so **the app and the live site still run the previous
models** (cotton, rice, soybean, sugarcane, tomato). The training code for the new version is in `main`; the app
changes that go with it are waiting for the new models (see [What is not in `main` yet](#what-is-not-in-main-yet)).

## Why

The previous models were tested on 119 photos from the web that none of the models had trained on
(`web_test_photos/`, kept out of training). Validation scores were 95–97%, but on these photos:

| Model (Oct 2026) | Disease right | Crop right | Right disease in top 3 |
|---|---|---|---|
| Single model, as deployed then | 52% | 97% | 85% |
| Single model, retrained | 57% | 95% | 78% |
| Router + crop specialists (live now) | 54% | 99% | 89% |

Disease right, per crop, for the router + specialists:

| Crop | Disease right |
|---|---|
| Cotton | 5 / 8 |
| Rice | 14 / 26 |
| Soybean | 7 / 22 |
| Sugarcane | 6 / 17 |
| Tomato | 22 / 46 |

The models were much weaker on real photos than validation suggested. Soybean diseases were often called healthy,
cotton was confused with soybean, and sugarcane mosaic was taken for rice. Most of the cotton and sugarcane training
photos were lab photos on a plain background, with few or no field photos.

## What changed

### Crops and conditions
- **Sugarcane removed, potato added.** Potato has healthy, early blight and late blight. These are common, look
  clearly different and have many field photos available. Sugarcane photos are still used, as part of the "other"
  (not a supported crop) class.
- **Cotton:** leaf curl virus added, and alternaria leaf spot and verticillium wilt removed. The two removed
  conditions had no field photos at all.
- Crops are now: cotton, soybean, potato, rice, tomato.

### Training data (`training/prepare.py`, `training/download_extra.py`)
New sources:

| Source | Used for |
|---|---|
| PLDD-UP potato set (Mendeley `3j4nfkvp2n`) | potato early blight, late blight, healthy |
| Indonesian potato set (Mendeley `ptz377bwb8`) | potato late blight, healthy |
| Cotton disease set (Mendeley `74jsdxtmx2`, Cotton-Original) | cotton healthy, bacterial blight, fusarium wilt, leaf curl |
| Multi-class soybean set (Mendeley `6fhphxg297`) | soybean bacterial blight, rust |
| Bugwood images (up to 80 per subject, credits in `data/raw/bugwood/_credits.json`) | soybean frogeye, bacterial blight, rust; cotton bacterial blight |
| PlantVillage / PlantDoc potato classes | potato |

- `remote_zip.py` reads remote zip files with HTTP range requests, so only the sampled photos are downloaded, not
  the whole archive. `download_extra.py` fetches 6 files in parallel.
- Bugwood microscope, spore and culture images are filtered out, so only leaf and plant photos are kept.
- **Holdout:** any training photo within a dHash distance of 8 of a `web_test_photos` image is dropped, so the web
  test stays honest.

The rebuilt dataset (`prepare.py --per-class 2000`) has 27,401 photos in 25 classes. Field photos are in brackets:

| Class | Photos (field) |
|---|---|
| cotton bacterial blight | 584 (375) |
| cotton fusarium wilt | 371 (47) |
| cotton healthy | 1,289 (956) |
| cotton leaf curl | 453 (453) |
| other | 1,089 (537) |
| potato early blight | 1,015 (815) |
| potato healthy | 931 (901) |
| potato late blight | 1,352 (1,152) |
| rice (5 classes) | 2,000 each |
| soybean bacterial blight | 692 (378) |
| soybean frogeye leaf spot | 518 (203) |
| soybean healthy | 2,000 (1,034) |
| soybean rust | 1,180 (871) |
| tomato (8 classes) | unchanged |

Leaf masks for background swapping (`masks.py`) cover 6,054 of 6,393 lab photos.

### Training (`training/train.py`, `training/train_all.sh`)
- New `--balance` option that weights each class by `1 / count ^ balance`. The crop specialists use `--balance 1`,
  so every condition is drawn equally often and a big healthy class does not swamp the diseases.
- The single 28-class comparison model is no longer trained.
- `export.py` deletes models of crops the app no longer covers, such as `sugarcane.onnx`.

### Testing (`training/webtest.py`)
This scores the router + specialists on `web_test_photos/` exactly as the app decides. It prints overall, per-crop
and Bugwood-only results, lists every mistake, and writes `runs/web_test.json`. Some test photos are also from
Bugwood, so they are reported separately.

### App (`app/src/data/crops.ts`, `app/src/components/art/Plant.tsx`, `app/src/data/ui.ts`)
Potato has full English and Marathi text (symptoms, treatment, simulator values) and a drawn potato plant in place
of sugarcane. Cotton leaf curl is added, and the removed cotton conditions are dropped. These changes type-check
(`npx tsc --noEmit`).

## Training progress when stopped

`retrain.sh` ran: prepare (done), masks (done), then the router. The router was stopped after 8 of its 10 epochs:

| Epoch | Loss | Validation | Field photos only |
|---|---|---|---|
| 1 | 1.303 | 95.4% | 95.0% |
| 2 | 0.671 | 97.8% | 97.2% |
| 3 | 0.599 | 98.3% | 97.8% |
| 4 | 0.563 | 98.5% | 98.0% |
| 5 | 0.539 | 98.7% | 98.2% |
| 6 | 0.513 | 98.8% | 98.4% |
| 7 | 0.498 | 98.9% | 98.5% |
| 8 | 0.480 | 98.9% | 98.4% |

These are scores for picking the crop (6 classes: cotton, other, potato, rice, soybean, tomato), from 23,300
training and 4,101 validation photos (2,516 field). The crop specialists had not started. One epoch takes about 22
minutes on the laptop RTX 2050, and the whole chain takes about 10 hours.

## What is not in `main` yet
- **App changes (potato in, sugarcane out).** The models now in `app/public/models/` still have sugarcane and no
  potato. These changes will be pushed together with the new models, so the live site never shows a crop it cannot
  diagnose.
- **Model checkpoints** (`training/runs/`, `*.pt`). These are git-ignored and kept on the training laptop. The
  previous version's checkpoints are in `training/runs_v2/`.
- **`web_test_photos/`.** Most of these photos are from other websites, so they are not published in this public
  repository.

## Next steps
1. Run the full rebuild and retrain (about 10 h). This runs prepare, masks, the router, the five specialists,
   `evaluate.py`, `export.py` and `webtest.py`:
   ```bash
   cd training && sh retrain.sh
   ```
   To skip the data steps and only train: `sh train_all.sh`, then `python webtest.py`.
2. Compare `logs/webtest.log` with the 54% baseline above, then iterate on the classes that are still weak.
3. Build the app, check that potato shows and the models load, update the README accuracy tables and data credits,
   then push. GitHub Pages redeploys on every push to `main`.

## Known weak spots
- **Cotton fusarium wilt** has only 47 field photos.
- **Soybean frogeye leaf spot** has only 203 field photos.
- The ASDID soybean dataset could not be downloaded: Zenodo returned 403, and Dryad's API needs a login.
- Tomato is unchanged in this round.
