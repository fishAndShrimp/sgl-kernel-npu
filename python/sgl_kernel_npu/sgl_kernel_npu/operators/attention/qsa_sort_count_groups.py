from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "attention.qsa_sort_count_groups"

_triton_fn = None
_torch_native_fn = None


@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(indices):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.qsa_sort_count_groups import (
            qsa_sort_count_groups,
        )

        _triton_fn = qsa_sort_count_groups
    return _triton_fn(indices)


@register_impl(OP, "torch_native")
def _torch_native_impl(indices):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.qsa_sort_count_groups import (
            qsa_sort_count_groups,
        )

        _torch_native_fn = qsa_sort_count_groups
    return _torch_native_fn(indices)


@register_operator(OP)
def qsa_sort_count_groups(indices, *, impl="auto"):
    """Sort each [rows,512] row and return [1,rows,512] block IDs and [1,rows] valid counts.

    The default implementation is Triton.
    """
    return get_impl_by_name(OP, impl)(indices)
