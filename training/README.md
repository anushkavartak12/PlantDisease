# Training the leaf model

**Easiest: Google Colab (free GPU).** Open [colab_train.ipynb](https://colab.research.google.com/github/AadidevRaizada/PlantDisease/blob/main/training/colab_train.ipynb), pick *Runtime → T4 GPU*, *Run all*, then unzip the downloaded `plant_doctor_model.zip` into `app/public/`.

## Running locally

Produces `app/public/models/model.onnx`, `labels.json`, `refs.json` and the reference thumbnails in `app/public/refs/`.
Labels are `<crop>___<condition>` and must match `app/src/data/crops.ts`.

```bash
cd training
uv venv -p 3.11 .venv
# CPU build (small download). If you have an NVIDIA GPU and a fast connection use
# --index-url https://download.pytorch.org/whl/cu124 instead.
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv timm onnx pillow tqdm

sh download.sh                                     # ~5 GB of raw data (slow links: it takes a while; files resume)
.venv/Scripts/python prepare.py --per-class 700
.venv/Scripts/python train.py --epochs 10
.venv/Scripts/python export.py
```
(On Mac/Linux use `.venv/bin/python`.)

## Data (all from Hugging Face, no login needed)
| Crop | Classes | Source |
|---|---|---|
| Sugarcane | healthy, red rot, rust, mosaic, yellow leaf | `YaswanthReddy23/Sugarcane_leaf` |
| Cotton | healthy, bacterial blight, alternaria, fusarium wilt, verticillium wilt | `Project-AgML/cotton_leaf_disease_classification` |
| Soybean | healthy, rust, bacterial blight, frogeye | `anandvermagmailcom/soybean-leaf-diseases` (+ PlantVillage healthy) |
| Rice | healthy, blast, bacterial leaf blight, brown spot, tungro | `sharmin3/Rice-Leaf-Disease` |
| Tomato | healthy, early blight, late blight, leaf curl, septoria, bacterial spot, mosaic, leaf mould | PlantVillage (`BrandonFors/Plant-Diseases-PlantVillage-Dataset`, test split) |
| Other | leaves of unsupported plants | PlantVillage + PlantDoc (apple, grape, corn, potato, …) |
| Field photos | tomato, soybean healthy, other | PlantDoc (`Project-AgML/plant_doc_classification`) |
| Field photos | rice (+ Paddy Doctor, 2 more), cotton, sugarcane, soybean healthy (Brazil), tomato (Tomato-Village, via git sparse checkout) | `Project-AgML/rice_leaf_disease_classification_bd`, `…/cotton_leaf_disease_classification_bangladesh_2`, `…/sugarcane_leaf_disease_classification` |

Check each dataset's licence before commercial use.

## Model
MobileNetV3-Large (ImageNet-pretrained, `timm`) fine-tuned at 224 px with strong augmentation and label smoothing.
The export has two outputs: `logits` for the diagnosis and an L2-normalised `embedding` used to find the most similar
reference photos in the app. Per-class validation accuracy is saved to `runs/metrics.json`.

**Background swap.** Lab photos (PlantVillage, white-sheet rice) have their plain backdrop detected and replaced by real field backgrounds while training (`--swap-p`), and field photos are oversampled (`--field-repeat`). `train.py` reports accuracy on field photos separately, plus leaf-type (crop) accuracy, and picks the best epoch using it. On a small machine keep `--workers 2` (each worker costs ~500 MB).

**Caveat:** some source photos are close-ups on plain or field backgrounds. Accuracy on real phone photos will be
lower than the validation number, so the app shows its confidence and asks for a retake when unsure.
