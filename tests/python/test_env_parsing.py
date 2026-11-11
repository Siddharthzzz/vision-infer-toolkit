"""CPU-only unit tests for environment parsing (run in CI)."""

import pytest

from infer_toolkit.env import (
    GPU_QUERY_FIELDS,
    parse_nvcc_version,
    parse_nvidia_smi_csv,
    parse_trt_header_version,
)


def test_parse_nvidia_smi_csv():
    line = (
        "NVIDIA GeForce RTX 4070 Laptop GPU, 616.92, 8188, 8.9, 42.00, 55.00, 115.00, P5,"
        " 3105, 8001, 51\n"
    )
    out = parse_nvidia_smi_csv(line)
    assert out["gpu.name"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
    assert out["gpu.compute_cap"] == "8.9"
    assert out["gpu.enforced.power.limit"] == "42.00"
    assert len(out) == len(GPU_QUERY_FIELDS)


def test_parse_nvidia_smi_csv_wrong_field_count():
    with pytest.raises(ValueError):
        parse_nvidia_smi_csv("a, b")


def test_parse_nvcc_version():
    text = "Cuda compilation tools, release 13.0, V13.0.88\nBuild cuda_13.0.r13.0/compiler"
    assert parse_nvcc_version(text) == "13.0"
    assert parse_nvcc_version("garbage") is None


@pytest.mark.parametrize(
    "header",
    [
        "#define TRT_MAJOR_ENTERPRISE 10\n#define TRT_MINOR_ENTERPRISE 13\n"
        "#define TRT_PATCH_ENTERPRISE 3\n#define TRT_BUILD_ENTERPRISE 9\n",
        "#define NV_TENSORRT_MAJOR 10\n#define NV_TENSORRT_MINOR 13\n"
        "#define NV_TENSORRT_PATCH 3\n#define NV_TENSORRT_BUILD 9\n",
    ],
)
def test_parse_trt_header_version(header):
    assert parse_trt_header_version(header) == "10.13.3.9"
