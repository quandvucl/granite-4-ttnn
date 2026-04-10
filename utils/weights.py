"""Weight conversion utilities for Attention1D integration."""

import torch


def reverse_permute(tensor, n_heads, dim1, dim2):
    """
    Convert HuggingFace Q/K weights to Meta format for RoPE compatibility.

    This transformation is required because HuggingFace stores Q/K weights with a
    different head layout that is incompatible with TTNN's RoPE implementation.

    Args:
        tensor: Weight tensor to convert [dim1, dim2]
        n_heads: Number of attention heads
        dim1: First dimension (usually n_heads * head_dim)
        dim2: Second dimension (usually hidden_size)

    Returns:
        Converted tensor in Meta format

    Example:
        >>> q_weight_hf = model.q_proj.weight  # [1536, 1536]
        >>> q_weight_meta = reverse_permute(q_weight_hf, 12, 1536, 1536)
    """
    return (
        tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def convert_qkv_weights_to_meta(
    q_weight, k_weight, v_weight, num_heads, num_kv_heads, head_dim, hidden_size
):
    """
    Convert Q/K/V weights from HuggingFace format to Meta format and combine them.

    This is the main entry point for preparing attention weights for Attention1D.

    Args:
        q_weight: Query projection weight [num_heads * head_dim, hidden_size]
        k_weight: Key projection weight [num_kv_heads * head_dim, hidden_size]
        v_weight: Value projection weight [num_kv_heads * head_dim, hidden_size]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head
        hidden_size: Model hidden dimension

    Returns:
        Combined QKV weight tensor [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]

    Example:
        >>> qkv_weight = convert_qkv_weights_to_meta(
        ...     hf_attn.q_proj.weight,
        ...     hf_attn.k_proj.weight,
        ...     hf_attn.v_proj.weight,
        ...     num_heads=12,
        ...     num_kv_heads=12,
        ...     head_dim=128,
        ...     hidden_size=1536
        ... )
    """
    dim = num_heads * head_dim
    dim_kv = num_kv_heads * head_dim

    # Convert Q and K to Meta format (V stays as-is)
    q_weight_meta = reverse_permute(q_weight, num_heads, dim, hidden_size)
    k_weight_meta = reverse_permute(k_weight, num_kv_heads, dim_kv, hidden_size)

    # Combine: [Q, K, V] along dim 0
    # Result shape: [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]
    qkv_weight = torch.cat([q_weight_meta, k_weight_meta, v_weight], dim=0)

    return qkv_weight
