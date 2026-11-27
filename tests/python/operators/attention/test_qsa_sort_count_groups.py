import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(128, 512)]


def generate_data(shape):
    rows, width = shape
    indices = torch.arange(width - 1, -1, -1, dtype=torch.int32).repeat(rows, 1)
    indices[::2] = -1
    indices[1::3, 2::7] = -1
    return indices


@pytest.mark.parametrize("shape", [(0, 512), (9, 512), (128, 512)])
def test_qsa_sort_count_groups(shape):
    from sgl_kernel_npu.operators.attention import qsa_sort_count_groups

    executor = check_accuracy(qsa_sort_count_groups, args=(generate_data(shape),))
    check_performance(executor, enable_perf=shape in perf_shape)


def test_qsa_sort_count_groups_invalid_input():
    from sgl_kernel_npu.operators.attention import qsa_sort_count_groups

    with pytest.raises(ValueError, match="512"):
        qsa_sort_count_groups(torch.zeros(2, 511, dtype=torch.int32))
