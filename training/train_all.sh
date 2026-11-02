#!/usr/bin/env sh
# Train all six models, score the pipeline and export it to the app. Safe to re-run after a crash or restart:
# finished models are skipped and an interrupted one resumes from its last epoch. Progress: logs/<task>.log
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" OPENBLAS_NUM_THREADS=1  # pretrained weights are cached after the first run
PY="${PY:-.venv/Scripts/python}"
[ -x "$PY" ] || PY=.venv/bin/python
mkdir -p logs

train() {  # task size epochs batch lr [extra args]
  [ -f "runs/$1/done" ] && return 0
  $PY -X faulthandler -u train.py --task "$1" --size "$2" --epochs "$3" --batch "$4" --lr "$5" ${6:+$6} >> "logs/$1.log" 2>&1
}

train router 224 10 48 6e-4
for crop in cotton soybean sugarcane tomato rice; do
  train "$crop" 288 14 32 5e-4 "--init runs/router/best.pt"
done
train all 224 10 48 6e-4  # the old single 28-class model, only as a baseline for evaluate.py
$PY -u evaluate.py > logs/evaluate.log 2>&1
$PY -u export.py > logs/export.log 2>&1
echo "ALL_DONE $(date)" >> logs/evaluate.log
