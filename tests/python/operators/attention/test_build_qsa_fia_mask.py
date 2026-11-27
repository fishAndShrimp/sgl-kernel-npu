import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

perf_shape = [(32, 2051)]


def generate_data(shape):
    from sgl_kernel_npu.operators.attention import QSAFIAMaskWorkspace

    rows, width = shape
    ids = torch.arange(512, dtype=torch.int32).repeat(rows, 1)
    ids[1::3, ::2] = -1
    lens = torch.full((rows,), width, dtype=torch.int32)
    lens[: min(rows, 4)] = torch.arange(min(rows, 4), dtype=torch.int32)
    if rows > 5:
        ids[5].fill_(-1)
        lens[5] = width // 4 * 4
    workspace = QSAFIAMaskWorkspace(
        ids.device, rows, rows * ((width + 31) // 32 * 32), 520
    )
    return (ids, lens, width, width, workspace)


@pytest.mark.parametrize("shape", [(0, 17), (7, 33), (32, 2051)])
def test_build_qsa_fia_mask(shape):
    from sgl_kernel_npu.operators.attention import build_qsa_fia_mask

    args = generate_data(shape)
    executor = make_executor(build_qsa_fia_mask, args=args)
    if not get_skip_acc_check():
        impl, impl_ref = get_test_impls()
        require_distinct_impls(build_qsa_fia_mask, impl, impl_ref)
        # Both outputs alias scratch space. Snapshot them before the next call.
        expected = tuple(t.clone() for t in build_qsa_fia_mask(*args, impl=impl_ref))
        args[-1].words.fill_(0)
        args[-1].selected.fill_(False)
        assert_accuracy(executor(), expected)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)


def test_build_qsa_fia_mask_invalid_input():
    from sgl_kernel_npu.operators.attention import build_qsa_fia_mask

    ids, lens, length, width, workspace = generate_data((2, 2051))
    with pytest.raises(ValueError, match="workspace"):
        build_qsa_fia_mask(ids, lens, length, width + 32, workspace)
