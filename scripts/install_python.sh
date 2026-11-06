#!/usr/bin/env bash
# Create/refresh .venv with the pinned Python stack. Usage: bash scripts/install_python.sh [--lock]
#   default : install from requirements.lock if present, else from top-level pins
#   --lock  : resolve from top-level pins and rewrite requirements.lock
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d .venv ]] || python3 -m venv .venv
PIP=".venv/bin/pip"
$PIP install -q --upgrade pip

if [[ "${1:-}" != "--lock" && -f requirements.lock ]]; then
  $PIP install -r requirements.lock
else
  $PIP install -r requirements.txt -r requirements-dev.txt
fi
$PIP install --no-deps -r requirements-trt.txt   # see comment in requirements-trt.txt
$PIP install --no-deps -e .

# libcudart.so.13 must come from the new-name package.
$PIP show nvidia-cuda-runtime >/dev/null || { echo "nvidia-cuda-runtime missing" >&2; exit 1; }

if [[ "${1:-}" == "--lock" ]]; then
  { echo "--extra-index-url https://download.pytorch.org/whl/cu130"
    $PIP freeze --exclude-editable | grep -vE '^tensorrt'; } > requirements.lock
  echo "wrote requirements.lock"
fi
