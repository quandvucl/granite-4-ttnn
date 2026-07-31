"""TTNN device and mesh utilities."""

from typing import Optional, Tuple
import torch
import ttnn
from utils.base import to_tt_tensor


def _is_mesh_device(device) -> bool:
    """Return True if device is a multi-card MeshDevice."""
    return hasattr(device, "get_num_devices") and device.get_num_devices() > 1


def _make_mesh_mapper(device):
    """Return ReplicateTensorToMesh for a MeshDevice, else None."""
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
    """Upload a torch tensor to device, auto-replicating on mesh if no mapper given."""
    if mesh_mapper is None:
        mesh_mapper = _make_mesh_mapper(device)
    return to_tt_tensor(torch_tensor, device, dtype, layout, mesh_mapper)


def softplus_and_clamp_tt(
    input_tt: ttnn.Tensor,
    min_val: float,
    max_val: float,
    deallocate_input: bool = True,
) -> ttnn.Tensor:
    """clamp(softplus(input), min_val, max_val) on a TTNN tensor."""
    exp_tt = ttnn.exp(input_tt)
    if deallocate_input:
        input_tt.deallocate(True)
    one_tt = ttnn.full_like(exp_tt, 1.0)
    exp_plus_one = ttnn.add(exp_tt, one_tt)
    exp_tt.deallocate(True)
    one_tt.deallocate(True)
    softplus_tt = ttnn.log(exp_plus_one)
    exp_plus_one.deallocate(True)
    clamped = ttnn.clip(softplus_tt, min_val, max_val)
    softplus_tt.deallocate(True)
    return clamped


def softplus_and_clamp_torch_via_tt(
    input_tensor: torch.Tensor,
    min_val: float,
    max_val: float,
    device,
    dtype: ttnn.DataType,
    target_shape: Optional[Tuple[int, ...]] = None,
    mesh_mapper=None,
) -> torch.Tensor:
    """clamp(softplus(input_tensor), min_val, max_val) via TTNN, returns torch tensor."""
    from utils.base import to_torch_tensor
    if mesh_mapper is None:
        mesh_mapper = _make_mesh_mapper(device)
    input_tt = to_tt_tensor(input_tensor, device, dtype, ttnn.TILE_LAYOUT, mesh_mapper)
    result_tt = softplus_and_clamp_tt(input_tt, min_val, max_val, deallocate_input=True)
    result = to_torch_tensor(result_tt, target_shape)
    result_tt.deallocate(True)
    return result
