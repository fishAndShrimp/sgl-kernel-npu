import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(32, 4)]


def generate_data(shape):
    keys = torch.randn(64, 1, 128, dtype=torch.bfloat16)
    positions = torch.arange(64 * 3).reshape(3, 64).T
    groups = torch.randint(0, 64, shape, dtype=torch.int32)
    return keys, positions, groups


@pytest.mark.parametrize("shape", [(0, 4), (1, 4), (32, 4)])
def test_qsa_compress_group_slots(shape):
    from sgl_kernel_npu.operators.indexer import qsa_compress_group_slots

    executor = check_accuracy(qsa_compress_group_slots, args=generate_data(shape))
    check_performance(executor, enable_perf=shape in perf_shape)


def test_qsa_compress_group_slots_invalid_input():
    from sgl_kernel_npu.operators.indexer import qsa_compress_group_slots

    keys, positions, groups = generate_data((2, 3))
    with pytest.raises(ValueError, match="rows, 4"):
        qsa_compress_group_slots(keys, positions, groups)
