"""
Utilities for handling TTNN mesh devices and mesh tensors.
"""
import ttnn
import torch


def is_mesh_device(device) -> bool:
    """Check if a device is a mesh device."""
    return isinstance(device, ttnn.MeshDevice)


def to_torch_from_mesh(tensor: ttnn.Tensor, device) -> torch.Tensor:
    """
    Convert a TTNN tensor to PyTorch tensor, handling both single and mesh devices.

    For mesh tensors, extracts data from the first device in the mesh.

    Args:
        tensor: TTNN tensor (single device or mesh)
        device: TTNN device (single or mesh)

    Returns:
        PyTorch tensor
    """
    if is_mesh_device(device):
        # For mesh tensors, use ttnn.to_torch() which handles mesh properly
        # This will gather the tensor from all devices
        return ttnn.to_torch(tensor)
    else:
        # Single device - use standard to_torch()
        return tensor.to_torch()


def to_device_replicated(tensor: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """
    Convert PyTorch tensor to TTNN tensor, replicating across all devices in a mesh.

    For single devices, behaves like normal to_device.
    For mesh devices, replicates the tensor to all devices.

    Args:
        tensor: PyTorch tensor
        device: TTNN device (single or mesh)
        dtype: TTNN dtype
        layout: TTNN layout

    Returns:
        TTNN tensor (replicated across mesh if mesh device)
    """
    if is_mesh_device(device):
        # Convert to TTNN tensor with replication across mesh
        tt_tensor = ttnn.from_torch(
            tensor,
            device=device,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device)
        )
        return tt_tensor
    else:
        # Single device - standard conversion
        tt_tensor = ttnn.from_torch(
            tensor,
            device=device,
            dtype=dtype,
            layout=layout
        )
        return tt_tensor


def get_num_devices(device) -> int:
    """Get the number of devices (1 for single device, N for mesh)."""
    if is_mesh_device(device):
        return device.get_num_devices()
    else:
        return 1
