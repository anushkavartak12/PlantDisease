# 🌱 Plant Doctor · पीक डॉक्टर

A cozy, bilingual (English / मराठी) farm helper for five of the most common crops of Maharashtra.
Snap a leaf to find its disease, or grow a crop in a little weather-and-disease game to learn **why** diseases appear.

It is an installable web app (PWA). The AI model runs **on the device**, so it works offline, costs nothing to run,
and photos never leave the phone.

**Crops:** 🎋 Sugarcane (ऊस) · ☁️ Cotton (कापूस) · 🫘 Soybean (सोयाबीन) · 🌾 Rice (भात) · 🍅 Tomato (टोमॅटो)

<p align="center">
  <img src="docs/screenshots/home.jpg" width="240" alt="Home screen with Shetkari Dada">
  <img src="docs/screenshots/scan-result.jpg" width="240" alt="Leaf scan result: tomato late blight">
  <img src="docs/screenshots/game.jpg" width="240" alt="Grow and learn game with a disease outbreak">
</p>

---

## Features

### 📷 Check a leaf
Take a photo with the camera, or upload one. The model:
1. **identifies the leaf type** (which of the 5 crops, or "not a crop I know"),
2. **diagnoses the condition** (27 conditions, including "healthy"), with a confidence score,
3. **shows the closest matching photos from its reference database**, so you can compare by eye,
4. explains **symptoms and what to do**, and reads it aloud with 🔊 in English or Marathi.

If confidence is low it asks for a better photo instead of guessing.

### 🌱 Grow & learn (the game)
Pick a crop and a season (Kharif monsoon, Rabi winter or summer), then play with **temperature, humidity, rain and
nitrogen fertiliser**. Each disease has its own weather "recipe" based on real plant pathology: a temperature window,
a humidity threshold, how much wet leaves matter, insect carriers for viruses (whitefly, aphids, leafhoppers) and
nitrogen effects. When conditions suit a disease, its bar fills up and it breaks out on the plant with its own look
(rings, rust pustules, streaks, mottling, wilting, curling).

Save the harvest with farm work: 💧 water, 🕳️ drain, 🧴 fungicide, 🧪 copper spray, 🌿 neem + traps, ✂️ remove sick
leaves. Shetkari Dada explains what happened and why. Every sound is generated in code with the Web Audio API.

### 📔 Disease diary
Every disease you meet in the game or find with the camera is collected, with its explanation and treatment.

<p align="center">
  <img src="docs/screenshots/game-setup.jpg" width="240" alt="Choosing a crop and season">
  <img src="docs/screenshots/diary.jpg" width="240" alt="Disease diary">
  <img src="docs/screenshots/home-marathi.jpg" width="240" alt="Home screen in Marathi">
  <img src="docs/screenshots/scan-result-marathi.jpg" width="240" alt="Rice blast result in Marathi">
</p>

---

## Quick start

### Just use it
Open the GitHub Pages link for this repo on any phone or computer and tap **Install / Add to Home screen**.

> First time: in the repo go to **Settings → Pages → Source: GitHub Actions**. After that every push to `main`
> deploys automatically (`.github/workflows/deploy.yml`).

### Run it locally (one click)
Needs [Node.js 20+](https://nodejs.org).

| OS | Do this |
|---|---|
| Windows | double-click `setup.bat` |
| Mac / Linux | `./setup.sh` |

Or by hand:
```bash
cd app
npm install
npm run dev
```
Open http://localhost:5173.

**Camera on your phone:** browsers only allow the camera over HTTPS. Run `setup.bat https` (or `npm run dev:https`),
open the `https://<your-pc-ip>:5173` address it prints on your phone (same Wi-Fi), and accept the certificate warning.

---

## How it works

```
 phone camera ──► 3 views (centre crop, mirrored, whole photo)
                        │
                        ▼
        router: MobileNetV3, 224 px ──► which crop? (5 crops or "other")
                        │
                        ▼  top crop (and the runner-up when the router gives it ≥ 20%)
        crop specialist: MobileNetV3, 288 px ──► healthy or which disease
                        │
          ┌─────────────┴──────────────────┐
     condition + confidence      embedding ──► closest of that crop's reference photos
          │                                │
          └──► result card (EN / मराठी, TTS) ◄┘
```

The app runs in two steps, like a general doctor who sends you on to a specialist. A **router** model only decides
which crop the leaf is from. That crop's **specialist** model then picks between that crop's own conditions, so it
only has to learn the fine differences that matter for one plant. The confidence shown is
P(crop) × P(condition | crop). All models run in the browser with ONNX Runtime Web (WebGPU, falling back to WASM).

- **Models:** six MobileNetV3-Large networks (ImageNet-pretrained via `timm`): the router at 224 px and one
  specialist per crop at 288 px, each starting from the router's weights. Trained with augmentation, label smoothing,
  EMA weights and background swapping (below). Each is exported to ONNX with float16 weights (about 8.4 MB, 50 MB in
  all) and two outputs: `logits` and a normalised `embedding`. `app/public/models/manifest.json` lists them.
  Specialists load the first time their crop comes up.
- **Reference database:** the 6 most typical validation photos per condition (162 in all), stored as thumbnails with
  int8-quantised embeddings from that crop's specialist (`app/public/models/refs.json`), searched in the browser.
- **Inference:** about 0.15 s per photo on a laptop once the models are loaded (about 1.5 s the first time a crop
  comes up); it runs fully offline once the app is cached.
- **Speech:** Web Speech API (`en-IN`, `mr-IN`, falling back to a Hindi voice for Devanagari when no Marathi voice is
  installed).

### Accuracy
Held-out validation: 3,874 photos split at random from the training data, the same photos for every model, scored
the way the app runs (`training/evaluate.py`):

| | Overall | Field photos only (2,056) |
|---|---|---|
| Disease / condition | **97.0%** | **95.3%** |
| Leaf type (which crop, or "other") | **99.5%** | **99.4%** |

"Field photos" are phone-camera shots with real backgrounds (leaf still on the plant, soil, grass, hands), which is
what the app really sees.

For comparison, on the same photos a single 28-class model trained the same way scores 96.2% disease (94.1% on field
photos), so the two-step design removes about a fifth of its field-photo mistakes. The earlier single model shipped
in the app scored 89.6% on field photos (on an older, smaller validation set). Lab photos have their plain backdrop
replaced by real field backgrounds during training (**background swapping**), so the model does not learn "plain
background = this disease". All six models train in about 10 hours on a laptop RTX 2050 (`training/train_all.sh`).

<details>
<summary>Per-class validation accuracy</summary>

| Crop | Condition | Accuracy |
|---|---|---|
| Cotton | alternaria leaf spot | 100.0% |
| Cotton | bacterial blight | 95.6% |
| Cotton | fusarium wilt | 95.9% |
| Cotton | healthy | 100.0% |
| Cotton | verticillium wilt | 95.5% |
| Other | unsupported plant | 91.8% |
| Rice | bacterial leaf blight | 99.0% |
| Rice | blast | 99.7% |
| Rice | brown spot | 99.0% |
| Rice | healthy | 98.3% |
| Rice | tungro | 98.0% |
| Soybean | bacterial blight | 100.0% |
| Soybean | frogeye leaf spot | 83.3% |
| Soybean | healthy | 99.0% |
| Soybean | rust | 98.0% |
| Sugarcane | healthy | 100.0% |
| Sugarcane | mosaic | 97.9% |
| Sugarcane | red rot | 100.0% |
| Sugarcane | rust | 99.0% |
| Sugarcane | yellow leaf | 100.0% |
| Tomato | bacterial spot | 83.3% |
| Tomato | early blight | 91.7% |
| Tomato | healthy | 98.2% |
| Tomato | late blight | 94.9% |
| Tomato | leaf curl | 98.3% |
| Tomato | leaf mold | 87.0% |
| Tomato | mosaic virus | 78.9% |
| Tomato | septoria leaf spot | 92.3% |
</details>

> ⚠️ **Honest limits.** Validation images are split at random from the same datasets as training, so near-duplicates
> can inflate the score, and 95% on field photos is an optimistic estimate of how it does on your own phone photos.
> Cotton (fusarium, verticillium, alternaria) and sugarcane (red rot, mosaic) have **no field photos at all** and rely
> on background swapping. Tomato mosaic virus, tomato bacterial spot and soybean frogeye leaf spot are the weakest
> conditions (79–83%). The "other" class has no non-leaf examples (hands, soil), so fill the frame with one leaf. The
> app shows its confidence and asks for a retake when unsure. It is a learning and first-aid tool, **not** a lab
> diagnosis: confirm with your local Krishi Vigyan Kendra before spraying.

---

## Retrain the model

**Easiest: Google Colab (free GPU).** Open
[`training/colab_train.ipynb` in Colab](https://colab.research.google.com/github/AadidevRaizada/PlantDisease/blob/main/training/colab_train.ipynb),
choose *Runtime → T4 GPU*, *Run all*, then unzip the downloaded `plant_doctor_model.zip` into `app/public/`.

Locally: `download.sh` → `prepare.py` → `masks.py` → `train_all.sh` (trains the router and the five specialists,
scores them with `evaluate.py` and exports them with `export.py`; safe to re-run after a crash). See
[training/README.md](training/README.md). Don't rebuild `data/images` while training runs: the models share one
validation split.

To add a crop or disease: add it to `app/src/data/crops.ts` (id `<crop>___<condition>`, English + Marathi text, and
simulator settings), map a dataset label to that id in `training/prepare.py`, and retrain.

---

## Project structure

```
app/                         React + Vite + TypeScript PWA
  src/data/crops.ts          crops, diseases, treatments, simulator settings (EN + मराठी)
  src/data/ui.ts             all interface text (EN + मराठी)
  src/lib/classifier.ts      ONNX inference (router → crop specialist), nearest reference photos
  src/lib/tts.ts             text-to-speech
  src/lib/sound.ts           Web Audio synth (effects, rain, birds)
  src/game/sim.ts            weather → disease simulator
  src/game/Game.tsx          game screen
  src/components/            scanner, result, home, diary, SVG art (scene, farmer, plants)
  public/models/             manifest.json, router.onnx + one <crop>.onnx per crop, refs.json, metrics.json
  public/refs/               reference thumbnails
training/                    download.sh, prepare.py, masks.py, train.py, train_all.sh, evaluate.py, export.py,
                             colab_train.ipynb
docs/screenshots/            images used in this README
setup.bat / setup.sh         one-click local setup
.github/workflows/deploy.yml GitHub Pages deployment
```

## Tech stack
React 18 · Vite 5 · TypeScript · ONNX Runtime Web · vite-plugin-pwa (Workbox) · Web Speech API · Web Audio API ·
hand-drawn SVG · Baloo 2 font (Latin + Devanagari) · PyTorch · timm · Google Colab · GitHub Actions + Pages.

---

## Sources

### Datasets (Hugging Face)
| Used for | Dataset |
|---|---|
| Sugarcane (healthy, red rot, rust, mosaic, yellow leaf) | [YaswanthReddy23/Sugarcane_leaf](https://huggingface.co/datasets/YaswanthReddy23/Sugarcane_leaf) |
| Cotton (healthy, bacterial blight, alternaria, fusarium wilt, verticillium wilt) | [Project-AgML/cotton_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/cotton_leaf_disease_classification) |
| Soybean (healthy, rust, bacterial blight, frogeye) | [anandvermagmailcom/soybean-leaf-diseases](https://huggingface.co/datasets/anandvermagmailcom/soybean-leaf-diseases) |
| Rice (healthy, blast, bacterial leaf blight, brown spot, tungro) | [sharmin3/Rice-Leaf-Disease](https://huggingface.co/datasets/sharmin3/Rice-Leaf-Disease) |
| Field photos (tomato, soybean healthy, other plants) | [Project-AgML/plant_doc_classification](https://huggingface.co/datasets/Project-AgML/plant_doc_classification) (PlantDoc, CC BY-SA 4.0) |
| Field photos: rice | [Project-AgML/rice_leaf_disease_classification_bd](https://huggingface.co/datasets/Project-AgML/rice_leaf_disease_classification_bd) (CC BY 4.0) |
| Field photos: rice (more) | [Project-AgML/paddy_disease_classification](https://huggingface.co/datasets/Project-AgML/paddy_disease_classification) (Paddy Doctor, **no licence stated**: training only, not in the shipped reference photos), [rice_leaf_disease_classification_2](https://huggingface.co/datasets/Project-AgML/rice_leaf_disease_classification_2) and [rice_disease_classification_bangladesh](https://huggingface.co/datasets/Project-AgML/rice_disease_classification_bangladesh) (CC BY 4.0) |
| Field photos: soybean | [Project-AgML/soybean_leaf_disease_classification_brazil](https://huggingface.co/datasets/Project-AgML/soybean_leaf_disease_classification_brazil) (healthy leaves, CC BY 4.0) |
| Field photos: tomato | [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) (Gehlot et al., *Multimedia Systems* 2023, [doi:10.1007/s00530-023-01158-y](https://doi.org/10.1007/s00530-023-01158-y); **no licence declared**: training only, not in the shipped reference photos) |
| Field photos: cotton | [Project-AgML/cotton_leaf_disease_classification_bangladesh_2](https://huggingface.co/datasets/Project-AgML/cotton_leaf_disease_classification_bangladesh_2) (CC BY 4.0) |
| Field photos: sugarcane | [Project-AgML/sugarcane_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/sugarcane_leaf_disease_classification) (CC BY 4.0) |
| Tomato conditions, healthy soybean, "other plant" class | [BrandonFors/Plant-Diseases-PlantVillage-Dataset](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) (PlantVillage) |

PlantVillage reference: Hughes & Salathé, *An open access repository of images on plant health to enable the
development of mobile disease diagnostics* (2015), [arXiv:1511.08060](https://arxiv.org/abs/1511.08060).
Check each dataset's licence before commercial use. Paddy Doctor and Tomato-Village state none: ask their authors before publishing the model commercially.

### Model and libraries
- MobileNetV3: Howard et al., *Searching for MobileNetV3* (2019), [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) · [PyTorch](https://pytorch.org)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) · [Export a PyTorch model to ONNX](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [vite-plugin-pwa](https://vite-pwa-org.netlify.app/) · [Web Speech API](https://developer.mozilla.org/docs/Web/API/Web_Speech_API) · [Web Audio API](https://developer.mozilla.org/docs/Web/API/Web_Audio_API)
- [Baloo 2 font](https://fonts.google.com/specimen/Baloo+2) via [Fontsource](https://fontsource.org/fonts/baloo-2)

### Disease information
Symptoms, treatments and the simulator's weather thresholds are simplified from general extension guidance on
these diseases. They are meant for learning and have not been reviewed by an agronomist. The Marathi text should be
reviewed by a local agronomist or KVK before any real-world use.

---

<sub>Built with ❤️ for Maharashtra's farmers. Illustrations are original SVG, inspired by flat Indian village art.</sub>
