"""Build a balanced, resized training set at data/images/<class_id>/*.jpg
from the raw dataset files in data/raw/ (fetch them with download.sh).

Class ids must match app/src/data/crops.ts.
"""

import argparse
import io
import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "images"
SIZE = 256  # stored short side; training crops to 224
OTHER = "other___unsupported"

PLANTVILLAGE_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

PV_MAP = {
    "Tomato___healthy": "tomato___healthy",
    "Tomato___Early_blight": "tomato___early_blight",
    "Tomato___Late_blight": "tomato___late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato___leaf_curl",
    "Tomato___Septoria_leaf_spot": "tomato___septoria_leaf_spot",
    "Tomato___Bacterial_spot": "tomato___bacterial_spot",
    "Tomato___Tomato_mosaic_virus": "tomato___mosaic_virus",
    "Tomato___Leaf_Mold": "tomato___leaf_mold",
    "Soybean___healthy": "soybean___healthy",
}


def pv_class(name: str) -> str | None:
    if name in PV_MAP:
        return PV_MAP[name]
    if name.startswith("Tomato"):
        return None  # tomato conditions we don't cover (spider mites, target spot)
    return OTHER


PLANTDOC_NAMES = [
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf", "Bell_pepper leaf spot", "Blueberry leaf",
    "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf", "Peach leaf", "Potato leaf early blight",
    "Potato leaf late blight", "Raspberry leaf", "Soyabean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
    "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf", "Tomato leaf bacterial spot",
    "Tomato leaf late blight", "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato mold leaf",
    "Tomato two spotted spider mites leaf", "grape leaf", "grape leaf black rot",
]
PD_MAP = {
    "Tomato Early blight leaf": "tomato___early_blight", "Tomato Septoria leaf spot": "tomato___septoria_leaf_spot",
    "Tomato leaf": "tomato___healthy", "Tomato leaf bacterial spot": "tomato___bacterial_spot",
    "Tomato leaf late blight": "tomato___late_blight", "Tomato leaf mosaic virus": "tomato___mosaic_virus",
    "Tomato leaf yellow virus": "tomato___leaf_curl", "Tomato mold leaf": "tomato___leaf_mold",
    "Soyabean leaf": "soybean___healthy",
}


def pd_class(name: str) -> str | None:
    if name in PD_MAP:
        return PD_MAP[name]
    return None if name.startswith("Tomato") else OTHER  # spider mites: not covered


# Real photos with natural backgrounds (phone cameras, fields). Taken first and prefixed "field_" so that
# train.py can oversample them, build a background-swap pool from them and report accuracy on them separately.
FIELD = {
    "plantdoc_0.parquet": (PLANTDOC_NAMES, pd_class),
    "plantdoc_1.parquet": (PLANTDOC_NAMES, pd_class),
    "rice_bd.parquet": (["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_blast", "leaf_scald",
                         "narrow_brown_spot"], {
        "bacterial_leaf_blight": "rice___bacterial_leaf_blight", "brown_spot": "rice___brown_spot",
        "healthy": "rice___healthy", "leaf_blast": "rice___blast"}.get),
    "cotton_bd.parquet": (["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage",
                           "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"], {
        "Bacterial Blight": "cotton___bacterial_blight", "Healthy Leaf": "cotton___healthy"}.get),
    "sugarcane_2_0.parquet": (SUGARCANE2 := ["Banded Chlorosis", "Brown Spot", "BrownRust", "Dried", "Grassy shoot",
                                             "Healthy", "Pokkah Boeng", "Sett Rot", "Smut", "Viral Disease",
                                             "Yellow Leaf"], {
        "Healthy": "sugarcane___healthy", "BrownRust": "sugarcane___rust",
        "Yellow Leaf": "sugarcane___yellow_leaf"}.get),
    "sugarcane_2_1.parquet": (SUGARCANE2, {
        "Healthy": "sugarcane___healthy", "BrownRust": "sugarcane___rust",
        "Yellow Leaf": "sugarcane___yellow_leaf"}.get),
}

# Round 2. names=None: read the class names from the parquet's own Hugging Face metadata.
def _m(d):
    return d.get


FIELD.update({
    **{f"paddy_{k}.parquet": (None, _m({
        "bacterial_leaf_blight": "rice___bacterial_leaf_blight", "blast": "rice___blast",
        "brown_spot": "rice___brown_spot", "normal": "rice___healthy", "tungro": "rice___tungro"})) for k in (0, 1)},
    **{f"rice2_{k}.parquet": (None, _m({
        "Bacterial leaf blight": "rice___bacterial_leaf_blight", "Brown spot": "rice___brown_spot",
        "Leaf Blast": "rice___blast"})) for k in (0, 1)},
    **{f"ricebd2_{k}.parquet": (None, _m({
        "Healthy": "rice___healthy", "Rice Blast": "rice___blast", "Rice Tungro": "rice___tungro"})) for k in (0, 1)},
    **{f"dhan_{k}.parquet": (None, _m({
        "Brown Spot": "rice___brown_spot", "Browon Spot": "rice___brown_spot", "Rice Blast": "rice___blast",
        "Rice Tungro": "rice___tungro", "Rice Turgro": "rice___tungro"})) for k in (0, 1)},
    **{f"cotton2_{k}.parquet": (None, _m({
        "Alternaria": "cotton___alternaria_leaf_spot", "Bacterial Blight": "cotton___bacterial_blight",
        "Healthy": "cotton___healthy"})) for k in (0, 1, 2)},
    **{f"soybean_br_{k}.parquet": (None, _m({"Healthy": "soybean___healthy"})) for k in (0, 1)},
    **{f"soybean2_{k}.parquet": (None, _m({
        "Bacterial leaf Blight": "soybean___bacterial_blight", "Healthy": "soybean___healthy"})) for k in range(6)},
})

# Sources with no stated licence. Used for training only: export.py keeps them out of the shipped reference photos.
NOLIC = {"paddy_0.parquet", "paddy_1.parquet"}
TOMATO_VILLAGE = {"Early_blight": "tomato___early_blight", "Healthy": "tomato___healthy",
                  "Late_blight": "tomato___late_blight"}


def hf_names(path: Path) -> list[str]:
    """Class names stored in a Hugging Face parquet file's metadata."""
    meta = pq.read_schema(path).metadata or {}
    feats = json.loads(meta[b"huggingface"])["info"]["features"]
    return feats["label"]["names"]


# parquet file -> (label names, label -> class id)
PARQUETS = {
    **FIELD,
    "sugarcane.parquet": (["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"], {
        "Healthy": "sugarcane___healthy", "RedRot": "sugarcane___red_rot", "Rust": "sugarcane___rust",
        "Mosaic": "sugarcane___mosaic", "Yellow": "sugarcane___yellow_leaf"}.get),
    "cotton.parquet": (["Alternaria_Leaf", "Bacterial_Blight", "Fusarium_Wilt", "Healthy_Leaf", "Verticillium_Wilt"], {
        "Healthy_Leaf": "cotton___healthy", "Bacterial_Blight": "cotton___bacterial_blight",
        "Alternaria_Leaf": "cotton___alternaria_leaf_spot", "Fusarium_Wilt": "cotton___fusarium_wilt",
        "Verticillium_Wilt": "cotton___verticillium_wilt"}.get),
    "soybean.parquet": (["Bacterial_blight", "Frogeye", "Healthy", "Soyabean_rust"], {
        "Healthy": "soybean___healthy", "Soyabean_rust": "soybean___rust",
        "Bacterial_blight": "soybean___bacterial_blight", "Frogeye": "soybean___frogeye_leaf_spot"}.get),
    "plantvillage.parquet": (PLANTVILLAGE_NAMES, pv_class),
}

RICE_FOLDERS = {
    "bacterialblight": "rice___bacterial_leaf_blight", "blast": "rice___blast", "brownspot": "rice___brown_spot",
    "healthy": "rice___healthy", "tungro": "rice___tungro",
}


def save(data: bytes, dest: Path) -> bool:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return False
    w, h = im.size
    s = SIZE / min(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=88)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--field-share", type=float, default=0.7, help="max fraction of a class taken from field photos")
    ap.add_argument("--other-per-label", type=int, default=20)
    ap.add_argument("--raw", type=Path, default=RAW)
    args = ap.parse_args()
    random.seed(0)
    shutil.rmtree(OUT, ignore_errors=True)
    counts: dict[str, int] = defaultdict(int)
    other_counts: dict[str, int] = defaultdict(int)

    field_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[tuple[str, str], int] = defaultdict(int)

    def take(cls: str | None, src_label: str, data: bytes, name: str, group: str = "") -> None:
        if cls is None or counts[cls] >= args.per_class:
            return
        is_field = name.startswith("field_")
        if is_field:
            cap = args.per_class * args.field_share
            # no single dataset may fill more than half the field quota of a class
            if field_counts[cls] >= cap or source_counts[cls, group] >= cap / 2:
                return
        if cls == OTHER:
            if other_counts[src_label] >= args.other_per_label:
                return
            other_counts[src_label] += 1
        if save(data, OUT / cls / name):
            counts[cls] += 1
            field_counts[cls] += is_field
            source_counts[cls, group] += 1

    for fname, (names, mapper) in PARQUETS.items():
        path = args.raw / fname
        if not path.exists():
            print("missing", path)
            continue
        try:
            if names is None:
                names = hf_names(path)
                print(f"{fname}: classes {names}")
            table = pq.read_table(path)
        except Exception as e:  # e.g. a download that is still in progress
            print(f"skipping {fname}: {type(e).__name__}")
            continue
        label_col = "label" if "label" in table.column_names else table.column_names[-1]
        images, labels = table.column("image"), table.column(label_col)
        order = list(range(table.num_rows))
        random.shuffle(order)
        for i in order:
            lab = labels[i].as_py()
            src = names[lab] if isinstance(lab, int) else str(lab)
            prefix = ("field_nolic_" if fname in NOLIC else "field_") if fname in FIELD else ""
            take(mapper(src), src, images[i].as_py()["bytes"], f"{prefix}{path.stem}_{i}.jpg",
                 re.sub(r"_\d+$", "", path.stem))
        print(f"read {fname}: {table.num_rows} rows")

    village = args.raw / "tomato_village"
    if village.exists():
        files = sorted(f for f in village.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                       and TOMATO_VILLAGE.get(f.parent.name))
        random.shuffle(files)
        for k, f in enumerate(files):
            take(TOMATO_VILLAGE[f.parent.name], "", f.read_bytes(), f"field_nolic_tomato_village_{k}.jpg",
                 "tomato_village")
        print(f"read tomato_village: {len(files)} images")

    rice = args.raw / "rice.zip"
    if rice.exists():
        with zipfile.ZipFile(rice) as z:
            members = [m for m in z.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(members)
            for k, m in enumerate(members):
                parts = [re.sub(r"[^a-z]", "", p.lower()) for p in Path(m).parts[:-1]]
                cls = next((RICE_FOLDERS[p] for p in reversed(parts) if p in RICE_FOLDERS), None)
                take(cls, "", z.read(m), f"rice_{k}.jpg")
        print(f"read rice.zip: {len(members)} images")

    print("\nImages per class:")
    for c in sorted(counts):
        print(f"  {c:40s} {counts[c]:4d}   ({field_counts[c]} field)")
    print(f"total {sum(counts.values())} images in {len(counts)} classes")


if __name__ == "__main__":
    main()
