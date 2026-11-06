"""M0 acceptance: the pinned GPU stack is installed and actually works.

Run locally with `pytest -m gpu`. Skipped in CI (no GPU).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

EXPECTED_TRT = "10.13.3.9"


def test_torch_sees_ada_gpu():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (8, 9)
    assert torch.version.cuda.startswith("13.")
    x = torch.ones(1024, device="cuda")
    assert float(x.sum()) == 1024.0


def test_tensorrt_version_and_calibrator_api():
    trt = pytest.importorskip("tensorrt")
    assert trt.__version__ == EXPECTED_TRT
    # M4 depends on the implicit-quantization calibrator interfaces.
    for name in ("IInt8EntropyCalibrator2", "IInt8MinMaxCalibrator", "IInt8LegacyCalibrator"):
        assert hasattr(trt, name), f"tensorrt.{name} missing"


def test_tensorrt_builds_tiny_engine():
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    inp = net.add_input("x", trt.float32, (1, 4))
    out = net.add_activation(inp, trt.ActivationType.RELU).get_output(0)
    net.mark_output(out)
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26)
    assert builder.build_serialized_network(net, cfg) is not None


def test_onnxruntime_runs_on_cuda_ep():
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    g = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    ort.preload_dlls()  # load CUDA/cuDNN from the pip nvidia-* wheels
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider"
    x = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(sess.run(None, {"x": x})[0], np.maximum(x, 0))
    assert onnx.__version__.startswith("1.21.")


def test_modelopt_quantization_importable():
    pytest.importorskip("modelopt")
    import modelopt.torch.quantization as mtq

    assert hasattr(mtq, "quantize")
