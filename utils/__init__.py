"""Generic TTNN utilities."""

from utils.base import to_torch_tensor, to_tt_tensor
from utils.device import (
    _is_mesh_device,
    _make_mesh_mapper,
)

__all__ = [
    "to_tt_tensor",
    "to_torch_tensor",
    "_is_mesh_device",
    "_make_mesh_mapper",
]
