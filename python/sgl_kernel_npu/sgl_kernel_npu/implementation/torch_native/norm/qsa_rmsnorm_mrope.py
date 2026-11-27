import torch


def qsa_rmsnorm_mrope(
    tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions=None
):
    x = tensor.float()
    normalized = (
        x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * (1 + weight.float())
    ).bfloat16()
    if positions is not None:
        cos_sin = cos_sin[positions.long()].bfloat16()
    cos, sin = cos_sin.float().chunk(2, dim=-1)
    if cos.ndim == 3:
        axes = torch.zeros(rope_dim // 2, device=x.device, dtype=torch.long)
        axes[1 : 3 * mrope_section[1] : 3] = 1
        axes[2 : 3 * mrope_section[2] : 3] = 2
        columns = torch.arange(rope_dim // 2, device=x.device)
        cos = cos.permute(1, 0, 2)[:, axes, columns]
        sin = sin.permute(1, 0, 2)[:, axes, columns]
    cos, sin = cos[:, None], sin[:, None]
    first, second = normalized[..., :rope_dim].float().chunk(2, dim=-1)
    left = (first * cos).bfloat16() - (second * sin).bfloat16()
    right = (second * cos).bfloat16() + (first * sin).bfloat16()
    return torch.cat((left, right, normalized[..., rope_dim:]), dim=-1)
