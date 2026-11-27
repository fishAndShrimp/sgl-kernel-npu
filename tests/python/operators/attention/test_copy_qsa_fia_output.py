import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(32, 4, 256)]


def generate_data(shape, strided):
    rows, heads, dim = shape
    result = torch.randn(shape, dtype=torch.bfloat16)
    selected = torch.arange(rows) % 3 != 0
    if strided:
        output = torch.full((rows, heads * 2, dim), 7, dtype=result.dtype)[:, ::2]
    else:
        output = torch.full_like(result, 7)
    return dict(result=result, selected=selected, output=output)


@pytest.mark.parametrize("shape", [(0, 1, 256), (3, 1, 256), (32, 4, 256)])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_copy_qsa_fia_output(shape, strided):
    from sgl_kernel_npu.operators.attention import copy_qsa_fia_output

    executor = check_accuracy(
        copy_qsa_fia_output,
        kwargs=generate_data(shape, strided),
        mutable_kwargs=("output",),
        standard={"torch.bfloat16": AccuracyStandard(0, 0)},
    )
    check_performance(executor, enable_perf=shape in perf_shape)


def test_copy_qsa_fia_output_invalid_input():
    from sgl_kernel_npu.operators.attention import copy_qsa_fia_output

    data = generate_data((2, 2, 256), False)
    with pytest.raises(ValueError, match="selected"):
        copy_qsa_fia_output(data["result"], data["selected"][:1], data["output"])
