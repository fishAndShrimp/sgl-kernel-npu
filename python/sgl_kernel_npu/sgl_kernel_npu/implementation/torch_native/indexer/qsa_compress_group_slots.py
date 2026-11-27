def qsa_compress_group_slots(keys, positions, groups):
    slots = groups.long()
    keys = keys.reshape(-1, 128)
    pooled = keys[slots].float().mean(dim=1).to(keys.dtype)
    return pooled.unsqueeze(1), positions[slots[:, 0]]
