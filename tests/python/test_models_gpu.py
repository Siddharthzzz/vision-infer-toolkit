"""GPU smoke tests: each adapter builds, preprocesses a synthetic image, runs on all three
backends' eager paths, and post-processes to the right result shapes/coordinates."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def jpg(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (375, 500, 3), dtype=np.uint8)  # non-square on purpose
    p = tmp_path_factory.mktemp("img") / "x.jpg"
    cv2.imwrite(str(p), img)
    return str(p)


@pytest.mark.parametrize("name", ["yolox_s", "rtdetr_r18", "segformer_b0"])
@pytest.mark.parametrize("backend", ["eager_fp32", "eager_amp"])
def test_adapter_end_to_end(name, backend, jpg):
    from infer_toolkit.backends import make_runner
    from infer_toolkit.models import get_adapter

    ad = get_adapter(name)
    model = ad.build("cuda")
    x, meta = ad.preprocess(jpg)
    H, W = ad.input_hw
    assert x.shape == (3, H, W) and x.dtype == np.float32
    assert meta["orig_hw"] == (375, 500)

    fn, info = make_runner(model, backend)
    outs = fn(torch.from_numpy(x)[None].cuda())
    (res,) = ad.postprocess(outs, [meta])
    if ad.task == "det":
        assert res["boxes_xyxy"].ndim == 2 and res["boxes_xyxy"].shape[1] == 4
        assert len(res["scores"]) == len(res["boxes_xyxy"])
    else:
        assert res.shape == (375, 500) and res.dtype == np.uint8 and res.max() < 150
