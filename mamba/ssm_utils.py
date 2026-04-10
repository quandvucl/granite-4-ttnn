"""SSM parameter extraction and preprocessing utilities."""

import torch


def extract_ssm_parameters(
    hidden_B_C: torch.Tensor,
    intermediate_size: int,
    n_groups: int,
    ssm_state_size: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract hidden, B, C from combined tensor after conv processing.

    Args:
        hidden_B_C: Combined tensor [batch, seq, intermediate_size + 2*n_groups*state_size]
        intermediate_size: Size of hidden states
        n_groups: Number of groups for grouped query
        ssm_state_size: SSM state dimension
        num_heads: Total number of heads

    Returns:
        Tuple of (hidden_inner, B, C)
            hidden_inner: [batch, seq, intermediate_size]
            B: [batch, seq, num_heads, ssm_state_size]
            C: [batch, seq, num_heads, ssm_state_size]
    """
    batch_size, seq_len = hidden_B_C.shape[:2]

    # Split into components
    hidden_inner, B, C = torch.split(
        hidden_B_C,
        [
            intermediate_size,
            n_groups * ssm_state_size,
            n_groups * ssm_state_size,
        ],
        dim=-1,
    )

    # Reshape and expand B, C from n_groups to num_heads
    group_repeat_factor = num_heads // n_groups

    B = B.reshape(batch_size, seq_len, n_groups, ssm_state_size)
    C = C.reshape(batch_size, seq_len, n_groups, ssm_state_size)

    # Expand: [B, S, n_groups, state] -> [B, S, num_heads, state]
    B = B.repeat_interleave(group_repeat_factor, dim=2)
    C = C.repeat_interleave(group_repeat_factor, dim=2)

    return hidden_inner, B, C


def prepare_dt_parameter(
    dt: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Prepare dt parameter for SSM computation.

    Args:
        dt: Time step [batch, seq, num_heads]
        num_heads: Number of heads
        head_dim: Dimension per head

    Returns:
        dt expanded: [batch, seq, num_heads, head_dim]
    """
    batch_size, seq_len = dt.shape[:2]

    # Expand to match hidden dimension
    # [B, S, H] -> [B, S, H, D]
    dt = dt.unsqueeze(-1).expand(batch_size, seq_len, num_heads, head_dim)

    return dt


def reshape_hidden_for_ssm(
    hidden_inner: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Reshape hidden states for SSM computation.

    Args:
        hidden_inner: [batch, seq, intermediate_size]
        num_heads: Number of heads
        head_dim: Dimension per head

    Returns:
        Reshaped: [batch, seq, num_heads, head_dim]
    """
    batch_size, seq_len = hidden_inner.shape[:2]
    return hidden_inner.reshape(batch_size, seq_len, num_heads, head_dim)


def apply_gated_norm(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    norm_fn,
) -> torch.Tensor:
    """
    Apply gated RMS normalization.

    Args:
        scan_output: SSM output [batch, seq, intermediate_size]
        gate: Gate values [batch, seq, intermediate_size]
        norm_fn: Normalization function

    Returns:
        Normalized output [batch, seq, intermediate_size]
    """
    return norm_fn(scan_output, gate)
