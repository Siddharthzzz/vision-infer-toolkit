#include "smoke.cuh"

namespace itk {
namespace {
__global__ void iota_kernel(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}
}  // namespace

cudaError_t launch_iota(int* d_out, int n, cudaStream_t stream) {
  constexpr int kBlock = 256;
  int grid = (n + kBlock - 1) / kBlock;
  iota_kernel<<<grid, kBlock, 0, stream>>>(d_out, n);
  return cudaGetLastError();
}

}  // namespace itk
