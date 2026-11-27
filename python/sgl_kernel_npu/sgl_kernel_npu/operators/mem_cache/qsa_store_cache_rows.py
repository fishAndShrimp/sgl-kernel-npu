from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "mem_cache.qsa_store_cache_rows"

_triton_fn = None
_torch_native_fn = None


@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(cache, slot_mapping, rows):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.qsa_store_cache_rows import (
            qsa_store_cache_rows,
        )

        _triton_fn = qsa_store_cache_rows
    return _triton_fn(cache, slot_mapping, rows)


@register_impl(OP, "torch_native")
def _torch_native_impl(cache, slot_mapping, rows):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.qsa_store_cache_rows import (
            qsa_store_cache_rows,
        )

        _torch_native_fn = qsa_store_cache_rows
    return _torch_native_fn(cache, slot_mapping, rows)


@register_operator(OP)
def qsa_store_cache_rows(cache, slot_mapping, rows, *, impl="auto"):
    """Write rows to valid slots of a [pages,page_size,1,width] cache in place.

    The default implementation is Triton.
    """
    return get_impl_by_name(OP, impl)(cache, slot_mapping, rows)
