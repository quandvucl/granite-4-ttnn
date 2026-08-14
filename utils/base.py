"""Base utilities for TTNN tensor conversions."""

from typing import Optional, Tuple

import torch
import ttnn


def to_tt_tensor(
    torch_tensor: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=None,
) -> ttnn.Tensor:
    """
    Convert PyTorch tensor to TTNN tensor.

    Args:
        torch_tensor: Input PyTorch tensor
        device: TTNN device
        dtype: Target dtype (default: bfloat16)
        layout: Target layout (default: ROW_MAJOR_LAYOUT for transfers)
        mesh_mapper: Optional mesh mapper for sharding/replication across devices

    Returns:
        TTNN tensor

    Note:
        Always use ROW_MAJOR_LAYOUT for CPU->TT transfers to prevent
        tile padding corruption. Convert to TILE_LAYOUT for computation.
        For mesh tensors:
        - Use mesh_mapper for explicit sharding/replication strategy
        - If no mesh_mapper provided on mesh device, auto-replicate
    """
    if torch_tensor is None:
        return None

    # Ensure tensor is contiguous
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()

    # Check if device is a mesh
    is_mesh = hasattr(device, "get_num_devices") and device.get_num_devices() > 1

    if mesh_mapper is not None:
        # Explicit mesh mapper provided: use from_torch with mesh_mapper
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Explicitly use DRAM for weights
            mesh_mapper=mesh_mapper,
        )
    elif is_mesh:
        # Mesh device with no explicit mapper: auto-replicate for safety
        # (Most operations need replicated inputs when weights are sharded)
        auto_mapper = ttnn.ReplicateTensorToMesh(device)
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Explicitly use DRAM for weights
            mesh_mapper=auto_mapper,
        )
    else:
        # Standard single-device tensor
        tt_tensor = ttnn.Tensor(
            torch_tensor, data_type=dtype, device=device, layout=layout
        )

    return tt_tensor


def to_torch_tensor(
    tt_tensor: ttnn.Tensor, target_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """
    Convert TTNN tensor back to PyTorch tensor.

    Args:
        tt_tensor: Input TTNN tensor
        target_shape: Optional target shape to reshape to

    Returns:
        PyTorch tensor

    Note:
        TILE_LAYOUT may add padding, so always force reshape to
        expected shape after conversion.
        For mesh tensors, properly gathers sharded tensors.
    """
    if tt_tensor is None:
        return None

    # Check if this is a mesh tensor
    device = tt_tensor.device()
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        # Use auto_compose to properly gather sharded tensors
        from models.common.auto_compose import to_torch_auto_compose
        torch_tensor = to_torch_auto_compose(tt_tensor, device=device)
    else:
        torch_tensor = tt_tensor.cpu().to_torch()

    # Force reshape to expected shape to undo padding
    if target_shape is not None:
        torch_tensor = torch_tensor.view(*target_shape)

    return torch_tensor
