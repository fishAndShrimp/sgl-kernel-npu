import torch


def copy_qsa_fia_output(result, selected, output):
    output.copy_(torch.where(selected[:, None, None], result, 0))
