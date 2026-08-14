"""TTNN device and mesh utilities."""

import ttnn


def _is_mesh_device(device) -> bool:
    """Return True if device is a multi-card MeshDevice."""
    return hasattr(device, "get_num_devices") and device.get_num_devices() > 1


def _make_mesh_mapper(device):
    """Return ReplicateTensorToMesh for a MeshDevice, else None."""
    if _is_mesh_device(device):
        return ttnn.ReplicateTensorToMesh(device)
    return None
