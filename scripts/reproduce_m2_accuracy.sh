#!/usr/bin/env bash
# M2 accuracy baselines: full COCO val2017 (5000) / ADE20K val (2000), batch 1.
# One row per (model, backend) appended to results/m2_accuracy.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
for model in yolox_s rtdetr_r18 segformer_b0; do
  for backend in eager_fp32 eager_amp compile_fp32; do
    echo ">> $model $backend"
    .venv/bin/python -m infer_toolkit.evaluate.accuracy --model "$model" --backend "$backend"
  done
done
