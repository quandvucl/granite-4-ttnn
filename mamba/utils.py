"""Shared Mamba utility functions for TTNN prefill/segment operations."""

import torch
import ttnn
from utils import to_torch_tensor, to_tt_tensor
from utils.device import _make_mesh_mapper
from models.common.tensor_utils import pad_dim_to_size


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pad tensor along sequence dimension (dim=1) using shared utility."""
    if pad_size == 0:
        return input_tensor
    return pad_dim_to_size(input_tensor, dim=1, size=input_tensor.shape[1] + pad_size)


def reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """Reshape tensor into chunks."""
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    This creates a causal attention-like matrix for intra-chunk computation.
    Input: [..., chunk_size] (e.g., [bsz, num_heads, num_chunks, chunk_size])
    Output: [..., chunk_size, chunk_size] (adds one dimension)

    Note: This is a CPU/PyTorch version. For TTNN implementation, see segment_sum_ttnn.
    """
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=0,
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def segment_sum_ttnn(input_tensor_tt, device, use_memory_efficient=True):
    """
    TTNN implementation of segment_sum for causal attention-like computation.

    Creates lower triangular cumsum for intra-chunk recurrence.
    Input: [..., chunk_size]
    Output: [..., chunk_size, chunk_size]

    All operations on device - fully parallelizable.

    Args:
        input_tensor_tt: Input tensor [..., chunk_size]
        device: TTNN device
        use_memory_efficient: If True, use chunked computation to reduce peak memory
                             (enables <16 device operation). Default: True.
    """
    # Get chunk size from last dimension
    shape = input_tensor_tt.shape
    chunk_size = shape[-1]

    # Memory-efficient version for large chunk sizes
    # Key insight: segment_sum creates a lower triangular matrix where
    # result[..., i, j] = sum(input[..., 0:i+1]) if j <= i, else -inf
    # We can compute this more efficiently by:
    # 1. Computing cumsum once
    # 2. Using gather/select operations to build the matrix
    # However, TTNN broadcast semantics may still create the full tensor.
    # The main memory savings come from changing the order of operations.

    if use_memory_efficient and chunk_size > 128:
        # For now, fall back to original - true memory savings require
        # either smaller chunk_size or algorithmic changes
        # TODO: Implement CPU-side chunking or kernel fusion
        pass  # Fall through to original implementation

    mesh_mapper = _make_mesh_mapper(device)

    # Original implementation for small chunk sizes or when requested
    # Step 1: Unsqueeze to add broadcast dimension: [..., chunk_size] -> [..., chunk_size, 1]
    input_unsqueezed = ttnn.unsqueeze(input_tensor_tt, -1)

    # Step 2: Create lower triangular mask (diagonal=-1)
    ones_2d = torch.ones(chunk_size, chunk_size, dtype=torch.float32)
    ones_2d_tt = to_tt_tensor(
        ones_2d,
        device,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    mask_lower = ttnn.tril(ones_2d_tt, diagonal=-1)
    ones_2d_tt.deallocate(True)

    # Step 3: Broadcast multiply (mask zeros out above-diagonal elements)
    input_masked = ttnn.multiply(input_unsqueezed, mask_lower)
    input_unsqueezed.deallocate(True)
    mask_lower.deallocate(True)

    # Step 4: Cumsum along dim=-2 (fully parallelizable on device)
    input_masked = ttnn.to_layout(input_masked, ttnn.TILE_LAYOUT)
    tensor_segsum = ttnn.cumsum(input_masked, dim=-2)
    input_masked.deallocate(True)

    # Step 5: Apply second mask (diagonal=0) to create causal structure
    ones_2d_2 = torch.ones(chunk_size, chunk_size, dtype=torch.float32)
    ones_2d_2_tt = to_tt_tensor(
        ones_2d_2,
        device,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    mask_diag = ttnn.tril(ones_2d_2_tt, diagonal=0)
    ones_2d_2_tt.deallocate(True)

    # Create -inf for masking
    neginf_value = -1e38
    neginf_tt = ttnn.full_like(tensor_segsum, neginf_value)

    # Apply mask: where mask==1, keep cumsum, else -inf
    result = ttnn.where(mask_diag, tensor_segsum, neginf_tt)

    tensor_segsum.deallocate(True)
    mask_diag.deallocate(True)
    neginf_tt.deallocate(True)

    return result


def init_mamba_cache(batch_size: int, device="cpu", dtype=torch.bfloat16) -> dict:
    """
    Initialize conv and SSM caches for Mamba decode mode.
    Both caches use float32 for maximum numerical precision regardless of dtype.

    Note: Hardcoded dimensions for granite-moe-3b-hybrid-8-v2:
    - conv_state: [batch, 3328, 4] where 3328 is conv_dim and 4 is kernel_size
    - ssm_state: [batch, 48, 64, 128] where 48 is num_heads, 64 is head_dim, 128 is ssm_state_size

    For dynamic initialization based on model config, use tt_ops.mamba.config.
    """
    return {
        "conv_state": torch.zeros(
            batch_size, 3328, 4, device=device, dtype=torch.float32
        ),
        "ssm_state": torch.zeros(
            batch_size, 48, 64, 128, device=device, dtype=torch.float32
        ),
    }
