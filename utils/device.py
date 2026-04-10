"""TTNN device and mesh utilities."""

from typing import Optional, Tuple, Callable
import torch
import ttnn
from utils.base import to_tt_tensor, to_torch_tensor


def _is_mesh_device(device) -> bool:
    """Check if device is a MeshDevice (multi-card)."""
    return hasattr(device, "get_num_devices") and device.get_num_devices() > 1


def _make_mesh_mapper(device):
    """Return ReplicateTensorToMesh mapper for MeshDevice, else None."""
    if _is_mesh_device(device):
        return ttnn.ReplicateTensorToMesh(device)
    return None


def _to_tt(
    torch_tensor: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=None,
) -> ttnn.Tensor:
    """
    Convenience wrapper for to_tt_tensor with auto mesh mapper.

    Args:
        torch_tensor: Input PyTorch tensor
        device: TTNN device
        dtype: Target dtype
        layout: Target layout
        mesh_mapper: Optional explicit mesh mapper (auto-replicate if None on mesh)

    Returns:
        TTNN tensor
    """
    if mesh_mapper is None:
        mesh_mapper = _make_mesh_mapper(device)

    return to_tt_tensor(torch_tensor, device, dtype, layout, mesh_mapper)


def _to_torch(
    tt_tensor: ttnn.Tensor,
    device,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Convenience wrapper for to_torch_tensor.

    Args:
        tt_tensor: Input TTNN tensor
        device: TTNN device (used for mesh detection)
        target_shape: Optional target shape to reshape to

    Returns:
        PyTorch tensor
    """
    return to_torch_tensor(tt_tensor, target_shape)


def tt_operation_with_cleanup(
    input_tensor: torch.Tensor,
    device,
    dtype: ttnn.DataType,
    operation: Callable[[ttnn.Tensor], ttnn.Tensor],
    target_shape: Optional[Tuple[int, ...]] = None,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    mesh_mapper=None,
) -> torch.Tensor:
    """
    Execute a TTNN operation with automatic tensor conversion and cleanup.

    Pattern:
        torch → ttnn → operation → ttnn → torch (with auto-deallocation)

    Args:
        input_tensor: Input PyTorch tensor
        device: TTNN device
        dtype: TTNN data type
        operation: Function that takes ttnn.Tensor and returns ttnn.Tensor
        target_shape: Optional shape for output
        layout: TTNN layout for input conversion
        mesh_mapper: Optional mesh mapper

    Returns:
        PyTorch tensor result

    Example:
        result = tt_operation_with_cleanup(
            x, device, dtype,
            lambda t: ttnn.exp(t),
            target_shape=x.shape
        )
    """
    if mesh_mapper is None:
        mesh_mapper = _make_mesh_mapper(device)

    # Convert to TTNN
    tt_input = to_tt_tensor(input_tensor, device, dtype, layout, mesh_mapper)

    # Execute operation
    tt_output = operation(tt_input)

    # Cleanup input
    tt_input.deallocate(True)

    # Convert back to torch
    result = to_torch_tensor(tt_output, target_shape)

    # Cleanup output
    tt_output.deallocate(True)

    return result


def softplus_and_clamp_tt(
    input_tt: ttnn.Tensor,
    min_val: float,
    max_val: float,
    deallocate_input: bool = True,
) -> ttnn.Tensor:
    """
    Apply softplus followed by clamp on TTNN tensor.

    Computes: clamp(softplus(input), min_val, max_val)
    where softplus(x) = log(1 + exp(x))

    Args:
        input_tt: Input TTNN tensor
        min_val: Minimum clamp value
        max_val: Maximum clamp value
        deallocate_input: Whether to deallocate input tensor (default: True)

    Returns:
        Clamped TTNN tensor

    Note:
        Intermediate tensors are automatically deallocated.
    """
    # softplus(x) = log(1 + exp(x))
    exp_tt = ttnn.exp(input_tt)
    if deallocate_input:
        input_tt.deallocate(True)

    one_tt = ttnn.full_like(exp_tt, 1.0)
    exp_plus_one_tt = ttnn.add(exp_tt, one_tt)
    exp_tt.deallocate(True)
    one_tt.deallocate(True)

    softplus_tt = ttnn.log(exp_plus_one_tt)
    exp_plus_one_tt.deallocate(True)

    # Clamp
    clamped_tt = ttnn.clip(softplus_tt, min_val, max_val)
    softplus_tt.deallocate(True)

    return clamped_tt


def softplus_and_clamp_torch_via_tt(
    input_tensor: torch.Tensor,
    min_val: float,
    max_val: float,
    device,
    dtype: ttnn.DataType,
    target_shape: Optional[Tuple[int, ...]] = None,
    mesh_mapper=None,
) -> torch.Tensor:
    """
    Apply softplus+clamp via TTNN with full torch→ttnn→torch conversion.

    Args:
        input_tensor: Input PyTorch tensor
        min_val: Minimum clamp value
        max_val: Maximum clamp value
        device: TTNN device
        dtype: TTNN data type
        target_shape: Optional output shape
        mesh_mapper: Optional mesh mapper

    Returns:
        PyTorch tensor with softplus+clamp applied
    """
    if mesh_mapper is None:
        mesh_mapper = _make_mesh_mapper(device)

    # Convert to TTNN
    input_tt = to_tt_tensor(
        input_tensor, device, dtype, ttnn.TILE_LAYOUT, mesh_mapper
    )

    # Apply softplus+clamp
    result_tt = softplus_and_clamp_tt(input_tt, min_val, max_val, deallocate_input=True)

    # Convert back
    result = to_torch_tensor(result_tt, target_shape)
    result_tt.deallocate(True)

    return result
