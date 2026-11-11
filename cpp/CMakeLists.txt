cmake_minimum_required(VERSION 3.24)
project(infer_toolkit_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 40xx (Ada)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# TensorRT (apt packages install into the multiarch include/lib dirs)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES x86_64-linux-gnu REQUIRED)
find_library(NVINFER_LIB nvinfer REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser REQUIRED)
add_library(tensorrt INTERFACE)
target_include_directories(tensorrt INTERFACE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(tensorrt INTERFACE ${NVINFER_LIB} ${NVONNXPARSER_LIB} CUDA::cudart)

# --- kernels -----------------------------------------------------------------
add_library(itk_kernels STATIC kernels/smoke.cu)
target_include_directories(itk_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_link_libraries(itk_kernels PUBLIC CUDA::cudart)
target_compile_options(itk_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# --- tests -------------------------------------------------------------------
enable_testing()
find_package(GTest REQUIRED)
add_executable(test_smoke ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE itk_kernels tensorrt GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_smoke)
