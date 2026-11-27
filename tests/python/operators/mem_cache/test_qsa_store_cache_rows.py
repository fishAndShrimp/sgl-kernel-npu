import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(32, 128)]


def generate_data(shape, dtype):
    count, width = shape
    cache = torch.zeros(20, 4, 1, width, dtype=dtype)[::2]
    rows = (
        torch.arange(count * width, dtype=torch.int32)
        .reshape(count, 1, width)
        .to(dtype)
    )
    slots = torch.arange(count + 1, dtype=torch.int64)
    slots[::5] = -1
    slots[1::5] = 40
    return dict(cache=cache, slot_mapping=slots, rows=rows)


@pytest.mark.parametrize(
    "shape,dtype",
    [((0, 3), torch.int64), ((7, 3), torch.int64), ((32, 128), torch.bfloat16)],
)
def test_qsa_store_cache_rows(shape, dtype):
    from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows

    executor = check_accuracy(
        qsa_store_cache_rows,
        kwargs=generate_data(shape, dtype),
        mutable_kwargs=("cache",),
        standard={"torch.bfloat16": AccuracyStandard(0, 0)},
    )
    check_performance(executor, enable_perf=shape in perf_shape)


def test_qsa_store_cache_rows_invalid_input():
    from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows

    data = generate_data((7, 3), torch.int64)
    data["rows"] = data["rows"].float()
    with pytest.raises(ValueError, match="cache dtype"):
        qsa_store_cache_rows(**data)
