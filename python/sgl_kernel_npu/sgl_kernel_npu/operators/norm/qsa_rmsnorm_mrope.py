from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "norm.qsa_rmsnorm_mrope"

_triton_fn = None
_torch_native_fn = None


@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.qsa_rmsnorm_mrope import qsa_rmsnorm_mrope

        _triton_fn = qsa_rmsnorm_mrope
    return _triton_fn(tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions)


@register_impl(OP, "torch_native")
def _torch_native_impl(
    tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.qsa_rmsnorm_mrope import (
            qsa_rmsnorm_mrope,
        )

        _torch_native_fn = qsa_rmsnorm_mrope
    return _torch_native_fn(
        tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions
    )


@register_operator(OP)
def qsa_rmsnorm_mrope(
    tensor,
    weight,
    cos_sin,
    eps,
    mrope_section,
    rope_dim,
    positions=None,
    *,
    impl="auto"
):
    """Normalize BF16 [T,H,D] with (1 + weight) and rotate the leading rope_dim features.

    The default implementation is Triton.
    """
    return get_impl_by_name(OP, impl)(
        tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions
    )
