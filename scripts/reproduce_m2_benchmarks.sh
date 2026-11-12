#!/usr/bin/env bash
# M2 latency / throughput / memory baselines (methodology §3-§5, amendments A1-A8).
# Machine state required: AC power, Armoury Crate Turbo, GPU mode Standard (MSHybrid),
# GPU-heavy Windows apps closed (preflight flags >1024 MiB in use before start).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-yolox_s rtdetr_r18 segformer_b0}"
for m in $MODELS; do
  echo ">> latency $m";    .venv/bin/python -m infer_toolkit.benchmark.latency --model "$m" --mode latency
done
for m in $MODELS; do
  echo ">> throughput $m"; .venv/bin/python -m infer_toolkit.benchmark.latency --model "$m" --mode throughput
done
for m in $MODELS; do
  echo ">> memory $m";     .venv/bin/python -m infer_toolkit.benchmark.memory --model "$m"
done
