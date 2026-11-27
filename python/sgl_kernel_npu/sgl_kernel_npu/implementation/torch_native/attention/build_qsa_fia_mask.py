import torch


def build_qsa_fia_mask(compact_indices, visible_lens, kv_length, mask_width, workspace):
    rows, groups = compact_indices.shape
    word_count = (mask_width + 31) // 32 * 8
    _, words, _, _, selected = workspace.views(rows, word_count, (groups + 8) // 8 * 8)
    visible = visible_lens.clamp(0, kv_length).long()[:, None]
    ids = compact_indices.long()
    full = visible // 4
    # Put invalid entries in a spare column outside the returned mask.
    valid = (ids >= 0) & (ids < full)
    block_mask = torch.ones(
        (rows, (mask_width + 3) // 4 + 1), dtype=torch.bool, device=ids.device
    )
    block_mask.scatter_(1, torch.where(valid, ids, block_mask.shape[1] - 1), False)
    token = torch.arange(mask_width, device=ids.device)
    allowed = ~block_mask[:, token // 4] | ((token >= full * 4) & (token < visible))
    selected.copy_(allowed.any(dim=1))
    allowed[:, 0] |= ~selected
    mask = words.view(torch.bool)[:, :mask_width]
    mask.copy_(~allowed)
    return mask, selected
