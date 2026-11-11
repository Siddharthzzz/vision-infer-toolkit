#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "smoke.cuh"

TEST(Toolchain, CudaKernelRuns) {
  constexpr int n = 1000;
  int* d = nullptr;
  ASSERT_EQ(cudaMalloc(&d, n * sizeof(int)), cudaSuccess);
  ASSERT_EQ(itk::launch_iota(d, n, nullptr), cudaSuccess);
  std::vector<int> h(n, -1);
  ASSERT_EQ(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(d);
  for (int i = 0; i < n; ++i) ASSERT_EQ(h[i], i);
}

TEST(Toolchain, TensorRTHeaderMatchesLibrary) {
  // Catches header/library version skew (e.g. pip wheel vs apt package).
  EXPECT_EQ(getInferLibVersion(), NV_TENSORRT_VERSION);
  EXPECT_EQ(NV_TENSORRT_MAJOR, 10);
}
