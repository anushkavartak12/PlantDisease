#!/usr/bin/env sh
# Full rebuild after a change to the crop list or the data: dataset -> leaf masks -> 6 models -> scores -> app export.
# Progress: logs/retrain.log and logs/<task>.log. Re-run train_all.sh alone to resume an interrupted training.
set -e
cd "$(dirname "$0")"
PY="${PY:-.venv/Scripts/python}"
[ -x "$PY" ] || PY=.venv/bin/python
mkdir -p logs
{
  echo "start $(date)"
  $PY -u prepare.py --per-class 2000
  echo "prepared $(date)"
  $PY -u masks.py
  echo "masks $(date)"
  sh train_all.sh
  $PY -u webtest.py > logs/webtest.log 2>&1
  echo "RETRAIN_DONE $(date)"
} >> logs/retrain.log 2>&1
