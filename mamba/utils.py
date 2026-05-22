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


def make_segment_sum_masks(chunk_size: int, device):
    """
    Pre-compute the two triangular masks used by segment_sum_ttnn.

    Returns (mask_lower_tt, mask_diag_tt) — both [chunk_size, chunk_size] TTNN tensors.
    Call once at model init and pass to segment_sum_ttnn via the masks= argument.
    """
    mesh_mapper = _make_mesh_mapper(device)
    ones = torch.ones(chunk_size, chunk_size, dtype=torch.float32)
    ones_tt = to_tt_tensor(ones, device, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_mapper)
    mask_lower = ttnn.tril(ones_tt, diagonal=-1)
    # reuse ones_tt for second mask — same values, different tril
    mask_diag = ttnn.tril(ones_tt, diagonal=0)
    ones_tt.deallocate(True)
    return mask_lower, mask_diag


def segment_sum_ttnn(input_tensor_tt, device, use_memory_efficient=True,
                     masks=None):
    """
    TTNN implementation of segment_sum for causal attention-like computation.

    Creates lower triangular cumsum for intra-chunk recurrence.
    Input: [..., chunk_size]
    Output: [..., chunk_size, chunk_size]

    Args:
        input_tensor_tt: Input tensor [..., chunk_size]
        device: TTNN device
        use_memory_efficient: Ignored (kept for API compatibility).
        masks: Optional (mask_lower_tt, mask_diag_tt) pre-computed by
               make_segment_sum_masks().  When provided, avoids 2 PCIe
               uploads per call — pass for performance-critical paths.
    """
    chunk_size = input_tensor_tt.shape[-1]

    if masks is not None:
        mask_lower, mask_diag = masks
    else:
        mask_lower, mask_diag = make_segment_sum_masks(chunk_size, device)

    # Step 1: [..., chunk_size] → [..., chunk_size, 1]
    input_unsqueezed = ttnn.unsqueeze(input_tensor_tt, -1)

    # Step 2: zero above-diagonal elements
    input_masked = ttnn.multiply(input_unsqueezed, mask_lower)
    input_unsqueezed.deallocate(True)

    # Step 3: causal cumsum
    input_masked = ttnn.to_layout(input_masked, ttnn.TILE_LAYOUT)
    tensor_segsum = ttnn.cumsum(input_masked, dim=-2)
    input_masked.deallocate(True)

    # Step 4: mask to −inf above diagonal
    neginf_tt = ttnn.full_like(tensor_segsum, -1e38)
    result = ttnn.where(mask_diag, tensor_segsum, neginf_tt)
    tensor_segsum.deallocate(True)
    neginf_tt.deallocate(True)

    if masks is None:
        mask_lower.deallocate(True)
        mask_diag.deallocate(True)

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
