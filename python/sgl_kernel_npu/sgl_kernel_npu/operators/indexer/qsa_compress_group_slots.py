from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "indexer.qsa_compress_group_slots"

_triton_fn = None
_torch_native_fn = None


@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(keys, positions, groups):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.qsa_compress_group_slots import (
            qsa_compress_group_slots,
        )

        _triton_fn = qsa_compress_group_slots
    return _triton_fn(keys, positions, groups)


@register_impl(OP, "torch_native")
def _torch_native_impl(keys, positions, groups):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.qsa_compress_group_slots import (
            qsa_compress_group_slots,
        )

        _torch_native_fn = qsa_compress_group_slots
    return _torch_native_fn(keys, positions, groups)


@register_operator(OP)
def qsa_compress_group_slots(keys, positions, groups, *, impl="auto"):
    """Average four BF16 key rows per group and return the first slot’s three position IDs.

    The default implementation is Triton.
    """
    return get_impl_by_name(OP, impl)(keys, positions, groups)
