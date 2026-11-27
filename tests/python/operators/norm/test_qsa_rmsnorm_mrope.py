import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(32, 4, 128)]


def generate_data(shape, mode):
    tokens, _, width = shape
    x = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn(width, dtype=x.dtype)
    cache = torch.randn(tokens + 7, 64)
    positions = torch.randint(-7, tokens + 7, (3, tokens))
    if mode == "raw":
        return x, weight, cache, 1e-6, [11, 11, 10], 64, positions
    trig = cache[positions] if mode == "three-axis" else cache[:tokens]
    return x, weight, trig.bfloat16(), 1e-6, [11, 11, 10], 64, None


@pytest.mark.parametrize("shape", [(1, 1, 128), (32, 4, 128), (0, 1, 128)])
@pytest.mark.parametrize("mode", ["text", "three-axis", "raw"])
def test_qsa_rmsnorm_mrope(shape, mode):
    from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

    executor = check_accuracy(qsa_rmsnorm_mrope, args=generate_data(shape, mode))
    check_performance(executor, enable_perf=shape in perf_shape and mode == "raw")


def test_qsa_rmsnorm_mrope_invalid_input():
    from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

    x, w, trig, eps, sections, dim, pos = generate_data((2, 4, 128), "text")
    with pytest.raises(ValueError, match="unit last stride"):
        qsa_rmsnorm_mrope(
            x, torch.zeros(256, dtype=w.dtype)[::2], trig, eps, sections, dim, pos
        )
