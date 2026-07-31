"""TTNN utilities for Mamba2 prefill chunk-scan."""

import torch
import ttnn
from utils import to_tt_tensor
from utils.device import _make_mesh_mapper


def make_segment_sum_masks(chunk_size: int, device):
    """Pre-compute the two triangular masks used by segment_sum_ttnn.

    Returns (mask_lower_tt, mask_diag_tt) — [chunk_size, chunk_size] TTNN tensors.
    Call once at model init and pass via masks= to avoid repeated PCIe uploads.
    """
    mesh_mapper = _make_mesh_mapper(device)
    ones = torch.ones(chunk_size, chunk_size, dtype=torch.float32)
    ones_tt = to_tt_tensor(ones, device, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_mapper)
    mask_lower = ttnn.tril(ones_tt, diagonal=-1)
    mask_diag  = ttnn.tril(ones_tt, diagonal=0)
    ones_tt.deallocate(True)
    return mask_lower, mask_diag


def segment_sum_ttnn(input_tensor_tt, device, use_memory_efficient=True, masks=None):
    """Lower-triangular cumulative sum for intra-chunk Mamba2 recurrence.

    Input:  [..., chunk_size]
    Output: [..., chunk_size, chunk_size]

    Args:
        masks: Optional (mask_lower_tt, mask_diag_tt) from make_segment_sum_masks().
               Pass pre-computed masks to avoid 2 PCIe uploads per call.
    """
    if masks is not None:
        mask_lower, mask_diag = masks
    else:
        mask_lower, mask_diag = make_segment_sum_masks(input_tensor_tt.shape[-1], device)

    input_unsqueezed = ttnn.unsqueeze(input_tensor_tt, -1)
    input_masked = ttnn.multiply(input_unsqueezed, mask_lower)
    input_unsqueezed.deallocate(True)

    input_masked = ttnn.to_layout(input_masked, ttnn.TILE_LAYOUT)
    tensor_segsum = ttnn.cumsum(input_masked, dim=-2)
    input_masked.deallocate(True)

    neginf_tt = ttnn.full_like(tensor_segsum, -1e38)
    result = ttnn.where(mask_diag, tensor_segsum, neginf_tt)
    tensor_segsum.deallocate(True)
    neginf_tt.deallocate(True)

    if masks is None:
        mask_lower.deallocate(True)
        mask_diag.deallocate(True)

    return result
