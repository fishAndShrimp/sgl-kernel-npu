from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "attention.copy_qsa_fia_output"

_triton_fn = None
_torch_native_fn = None


@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(result, selected, output):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.copy_qsa_fia_output import (
            copy_qsa_fia_output,
        )

        _triton_fn = copy_qsa_fia_output
    return _triton_fn(result, selected, output)


@register_impl(OP, "torch_native")
def _torch_native_impl(result, selected, output):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.copy_qsa_fia_output import (
            copy_qsa_fia_output,
        )

        _torch_native_fn = copy_qsa_fia_output
    return _torch_native_fn(result, selected, output)


@register_operator(OP)
def copy_qsa_fia_output(result, selected, output, *, impl="auto"):
    """Copy [rows,heads,dim] into output in place, zeroing rows where selected is false.

    The default implementation is Triton.
    """
    return get_impl_by_name(OP, impl)(result, selected, output)
