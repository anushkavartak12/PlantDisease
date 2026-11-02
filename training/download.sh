#!/usr/bin/env sh
# Downloads the raw datasets (~565 MB) into data/raw. Safe to re-run: curl resumes partial files.
set -e
cd "$(dirname "$0")"
mkdir -p data/raw
H=https://huggingface.co/datasets
get() { echo "-> $1"; curl -L -C - --retry 5 -o "data/raw/$1" "$2"; }
get soybean.parquet      "$H/anandvermagmailcom/soybean-leaf-diseases/resolve/main/data/train-00000-of-00001.parquet"
get sugarcane.parquet    "$H/YaswanthReddy23/Sugarcane_leaf/resolve/main/data/train-00000-of-00001.parquet"
get cotton.parquet       "$H/Project-AgML/cotton_leaf_disease_classification/resolve/main/raw/train-00000-of-00001.parquet"
get rice.zip             "$H/sharmin3/Rice-Leaf-Disease/resolve/main/Rice%20Leaf%20Disease-20241115T062818Z-001.zip"
get plantvillage.parquet "$H/BrandonFors/Plant-Diseases-PlantVillage-Dataset/resolve/main/data/test-00000-of-00001.parquet"

# --- Field-photo datasets (real backgrounds, phone cameras): fixes "only works on plain backgrounds" ---
A=$H/Project-AgML
get plantdoc_0.parquet   "$A/plant_doc_classification/resolve/main/data/train-00000-of-00002.parquet"
get plantdoc_1.parquet   "$A/plant_doc_classification/resolve/main/data/train-00001-of-00002.parquet"
get rice_bd.parquet      "$A/rice_leaf_disease_classification_bd/resolve/main/data/train-00000-of-00001.parquet"
get cotton_bd.parquet    "$A/cotton_leaf_disease_classification_bangladesh_2/resolve/main/raw/train-00000-of-00001.parquet"
get sugarcane_2_0.parquet "$A/sugarcane_leaf_disease_classification/resolve/main/data/train-00000-of-00002.parquet"
get sugarcane_2_1.parquet "$A/sugarcane_leaf_disease_classification/resolve/main/data/train-00001-of-00002.parquet"

# --- More field photos (round 2), in priority order. The link is slow (~0.6 MB/s), so parallel jobs do not help. ---
# Licences: CC BY 4.0, except paddy_* (Paddy Doctor: none stated on the HF page) and tomato_village (GitHub:
# none declared). Check before commercial use; export.py keeps those two out of the shipped reference photos.

# Tomato-Village: real tomato field photos (Rajasthan). Only the 3 classes we cover, via sparse checkout (~170 MB).
if [ ! -d data/raw/tomato_village ]; then
  git -c http.sslBackend=schannel clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/mamta-joshi-gehlot/Tomato-Village.git data/raw/tomato_village
  git -C data/raw/tomato_village sparse-checkout set --no-cone     "/Variant-a(Multiclass Classification)/*/Early_blight/*" "/Variant-a(Multiclass Classification)/*/Healthy/*"     "/Variant-a(Multiclass Classification)/*/Late_blight/*"
  git -C data/raw/tomato_village -c http.sslBackend=schannel checkout -q
fi
get paddy_0.parquet       "$A/paddy_disease_classification/resolve/main/data/train-00000-of-00002.parquet"
get paddy_1.parquet       "$A/paddy_disease_classification/resolve/main/data/train-00001-of-00002.parquet"
get soybean_br_0.parquet  "$A/soybean_leaf_disease_classification_brazil/resolve/main/data/train-00000-of-00002.parquet"
get soybean_br_1.parquet  "$A/soybean_leaf_disease_classification_brazil/resolve/main/data/train-00001-of-00002.parquet"
get rice2_0.parquet       "$A/rice_leaf_disease_classification_2/resolve/main/data/train-00000-of-00002.parquet"
get rice2_1.parquet       "$A/rice_leaf_disease_classification_2/resolve/main/data/train-00001-of-00002.parquet"
get ricebd2_0.parquet     "$A/rice_disease_classification_bangladesh/resolve/main/raw/train-00000-of-00002.parquet"
get ricebd2_1.parquet     "$A/rice_disease_classification_bangladesh/resolve/main/raw/train-00001-of-00002.parquet"
# soybean field photos (bacterial blight + healthy, 2.9 GB): the only field photos of a soybean disease
for k in 0 1 2 3 4 5; do
  get soybean2_$k.parquet "$A/soybean_leaf_disease_classification/resolve/main/data/train-0000$k-of-00006.parquet"
done

# Optional, low value per GB (prepare.py already knows them; they are skipped if missing):
#   dhan_0/1.parquet     Dhan-Shomadhan rice, 669 MB:   .../Dhan-Shomadhan_rice_leaf_disease_classification/.../data/train-0000{0,1}-of-00002.parquet
#   cotton2_0..2.parquet cotton field set, 1.3 GB for 132 images (partly synthetic): .../cotton_disease_classification/.../data/train-0000{0,1,2}-of-00003.parquet
