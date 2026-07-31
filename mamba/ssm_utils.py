"""SSM parameter extraction for Mamba2 prefill."""

import torch


def extract_ssm_parameters(
    hidden_B_C: torch.Tensor,
    intermediate_size: int,
    n_groups: int,
    ssm_state_size: int,
    num_heads: int,
) -> tuple:
    """Split combined post-conv tensor into hidden, B, C and expand groups to heads.

    Args:
        hidden_B_C: [batch, seq, intermediate_size + 2*n_groups*ssm_state_size]
    Returns:
        (hidden_inner [B,S,I], B [B,S,H,N], C [B,S,H,N])
    """
    batch_size, seq_len = hidden_B_C.shape[:2]
    hidden_inner, B, C = torch.split(
        hidden_B_C,
        [intermediate_size, n_groups * ssm_state_size, n_groups * ssm_state_size],
        dim=-1,
    )
    group_repeat = num_heads // n_groups
    B = B.reshape(batch_size, seq_len, n_groups, ssm_state_size).repeat_interleave(group_repeat, dim=2)
    C = C.reshape(batch_size, seq_len, n_groups, ssm_state_size).repeat_interleave(group_repeat, dim=2)
    return hidden_inner, B, C
