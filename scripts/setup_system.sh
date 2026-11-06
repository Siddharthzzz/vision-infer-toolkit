#!/usr/bin/env bash
# System-level setup for WSL2 Ubuntu 24.04. Run once:  bash scripts/setup_system.sh
# Installs: build tools, CUDA toolkit 13.0 (NO driver - WSL uses the Windows driver),
# TensorRT 10.13.3.9 C++ dev packages (+ trtexec). Nsight tools come with the toolkit.
#
# Versions match this machine's existing Isaac ROS apt pins
# (/etc/apt/preferences.d/isaac-ros-*.pref): CUDA 13.0, TensorRT 10.13.3.9+cuda13.0.
# This script does not modify those pins.
set -euo pipefail

CUDA_PKG=cuda-toolkit-13-0
TRT_VER=10.13.3.9-1+cuda13.0
KEYRING=/usr/share/keyrings/cuda-archive-keyring.gpg
NV=https://developer.download.nvidia.com/compute/cuda/repos

if ! grep -qi microsoft /proc/version; then
  echo "This script is for WSL2 only." >&2; exit 1
fi

echo ">> Base build tools"
sudo apt-get update
sudo apt-get install -y build-essential ninja-build pkg-config clang-format \
  libgtest-dev libopencv-dev wget

echo ">> NVIDIA apt keyring (WSL-Ubuntu repo)"
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  tmp=$(mktemp -d)
  wget -q -O "$tmp/cuda-keyring.deb" "$NV/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "$tmp/cuda-keyring.deb"
  rm -rf "$tmp"
fi

# TensorRT packages live only in the ubuntu2404 repo; add it with the same key.
echo "deb [signed-by=$KEYRING] $NV/ubuntu2404/x86_64/ /" |
  sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-trt.list >/dev/null

# Never let a Linux GPU driver into WSL. (TensorRT is already pinned by Isaac ROS.)
sudo tee /etc/apt/preferences.d/vision-infer-toolkit >/dev/null <<EOF
Package: cuda-drivers* nvidia-driver* nvidia-dkms* cuda-compat* nvidia-open*
Pin: release a=*
Pin-Priority: -1
EOF

sudo apt-get update

# Fail early with a clear message if apt would pick something else.
cand=$(apt-cache policy libnvinfer-dev | awk '/Candidate:/{print $2}')
if [[ "$cand" != "$TRT_VER" ]]; then
  echo "apt candidate for libnvinfer-dev is '$cand', expected '$TRT_VER'. Check /etc/apt/preferences.d/." >&2
  exit 1
fi

echo ">> CUDA toolkit 13.0 (bundles Nsight Systems + Nsight Compute)"
sudo apt-get install -y "$CUDA_PKG"

echo ">> TensorRT $TRT_VER (C++ headers/libs, ONNX parser, plugins, trtexec)"
sudo apt-get install -y \
  "libnvinfer-dev=$TRT_VER" "libnvinfer-plugin-dev=$TRT_VER" \
  "libnvonnxparsers-dev=$TRT_VER" "libnvinfer-bin=$TRT_VER"
sudo apt-mark hold libnvinfer10 libnvinfer-dev libnvinfer-plugin10 libnvinfer-plugin-dev \
  libnvonnxparsers10 libnvonnxparsers-dev libnvinfer-bin

echo ">> PATH (add to ~/.bashrc if missing)"
line='export PATH=/usr/local/cuda-13.0/bin:/usr/src/tensorrt/bin:$PATH'
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc

echo ">> Done. Open a new shell, then: nvcc --version && trtexec --help | head -1"
