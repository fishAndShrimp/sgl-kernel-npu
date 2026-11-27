import torch


def qsa_sort_count_groups(indices):
    valid = indices >= 0
    sorted_ids = indices.masked_fill(~valid, 16777215).sort(dim=-1).values
    return sorted_ids.unsqueeze(0), valid.sum(1, dtype=torch.int32).unsqueeze(0)
