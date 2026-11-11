// M0 toolchain smoke test: proves nvcc, the CUDA runtime and linking work.
#pragma once
#include <cuda_runtime.h>

namespace itk {

// Writes out[i] = i for i in [0, n).
cudaError_t launch_iota(int* d_out, int n, cudaStream_t stream);

}  // namespace itk
