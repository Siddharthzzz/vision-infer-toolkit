#!/usr/bin/env bash
# M2 latency / throughput / memory baselines (methodology §3-§5, amendments A1-A8).
# Machine state required: AC power, Armoury Crate Turbo, GPU mode Standard (MSHybrid),
# GPU-heavy Windows apps closed (preflight flags >1024 MiB in use before start).
# Resumable: a (model, phase) already present in its results CSV is skipped, because rows are
# written per model only when that model's phase completes. FORCE=1 re-runs everything.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-yolox_s rtdetr_r18 segformer_b0}"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1  # progress lines reach the log as they happen, not at process exit

done_already() {  # $1 = csv basename, $2 = model
  [[ "${FORCE:-0}" == 1 ]] && return 1
  [[ -f "results/$1.csv" ]] || return 1
  $PY -c "import csv,sys; sys.exit(0 if any(r['model']=='$2' for r in csv.DictReader(open('results/$1.csv'))) else 1)"
}

for m in $MODELS; do
  if done_already m2_latency "$m"; then echo ">> skip latency $m (done)"; continue; fi
  echo ">> latency $m";    $PY -m infer_toolkit.benchmark.latency --model "$m" --mode latency
done
for m in $MODELS; do
  if done_already m2_throughput "$m"; then echo ">> skip throughput $m (done)"; continue; fi
  echo ">> throughput $m"; $PY -m infer_toolkit.benchmark.latency --model "$m" --mode throughput
done
for m in $MODELS; do
  if done_already m2_memory "$m"; then echo ">> skip memory $m (done)"; continue; fi
  echo ">> memory $m";     $PY -m infer_toolkit.benchmark.memory --model "$m"
done
