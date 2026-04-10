"""TTNN device management - memory layouts, allocation tracking, conversions."""

from typing import Optional, Set, Any, Tuple
import torch
import ttnn

from utils import to_torch_tensor, to_tt_tensor
from mamba.config import MemoryConfig


class AllocationTracker:
    """Track TTNN tensor allocations for cleanup."""

    def __init__(self):
        self._allocations: Set[int] = set()
        self._freed: Set[int] = set()

    def track(self, tensor) -> Any:
        """Track a tensor allocation."""
        if tensor is not None and hasattr(tensor, '__hash__'):
            try:
                self._allocations.add(id(tensor))
            except:
                pass
        return tensor

    def mark_freed(self, tensor):
        """Mark a tensor as freed."""
        if tensor is not None:
            try:
                self._freed.add(id(tensor))
            except:
                pass

    def is_freed(self, tensor) -> bool:
        """Check if tensor was already freed."""
        if tensor is None:
            return True
        try:
            return id(tensor) in self._freed
        except:
            return False

    def cleanup(self):
        """Cleanup remaining allocations."""
        # In practice, TTNN handles cleanup automatically
        self._allocations.clear()
        self._freed.clear()


class TTNNDeviceManager:
    """
    Manages TTNN device operations:
    - Memory layouts (TILE vs ROW_MAJOR)
    - Tensor conversions (torch ↔ TTNN)
    - Explicit deallocation tracking
    - Memory configs for operations
    """

    def __init__(
        self,
        device,
        dtype=ttnn.bfloat16,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize device manager.

        Args:
            device: TTNN device (Device or MeshDevice)
            dtype: Default data type
            config: Memory configuration
        """
        self.device = device
        self.dtype = dtype
        self.config = config or MemoryConfig()
        self._allocation_tracker = AllocationTracker()

        # Check if device is mesh
        self._is_mesh = hasattr(device, "get_num_devices") and device.get_num_devices() > 1

    @property
    def is_mesh(self) -> bool:
        """Check if device is a mesh device."""
        return self._is_mesh

    def to_device(
        self,
        tensor: torch.Tensor,
        layout: str = "TILE",
        track: bool = True
    ):
        """
        Convert torch tensor to TTNN with proper layout.

        Args:
            tensor: PyTorch tensor
            layout: "TILE" or "ROW_MAJOR"
            track: Whether to track allocation

        Returns:
            TTNN tensor
        """
        tt_tensor = to_tt_tensor(tensor, self.device, self.dtype)

        if layout == "TILE":
            tt_tensor = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
        elif layout == "ROW_MAJOR":
            tt_tensor = ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)

        if track:
            self._allocation_tracker.track(tt_tensor)

        return tt_tensor

    def to_host(
        self,
        tensor,
        target_shape: Optional[Tuple[int, ...]] = None,
        deallocate: bool = True
    ) -> torch.Tensor:
        """
        Convert TTNN tensor to torch and optionally deallocate.

        Args:
            tensor: TTNN tensor
            target_shape: Optional reshape target
            deallocate: Whether to deallocate TTNN tensor

        Returns:
            PyTorch tensor
        """
        result = to_torch_tensor(tensor, target_shape=target_shape)

        if deallocate:
            self.deallocate(tensor)

        return result

    def deallocate(self, tensor, force: bool = False):
        """
        Safely deallocate TTNN tensor.

        Args:
            tensor: TTNN tensor to deallocate
            force: Force deallocation even if already freed
        """
        if tensor is None:
            return

        if not force and self._allocation_tracker.is_freed(tensor):
            return

        try:
            tensor.deallocate(True)
            self._allocation_tracker.mark_freed(tensor)
        except:
            # Tensor may already be deallocated
            pass

    def get_memory_config(self, shape: Tuple[int, ...], op_type: str):
        """
        Get optimal memory config for operation.

        Args:
            shape: Tensor shape
            op_type: Operation type (e.g., "matmul", "scan")

        Returns:
            Memory config or None
        """
        return self.config.get_config(shape, op_type)

    def change_layout(self, tensor, layout: str):
        """
        Change tensor layout.

        Args:
            tensor: TTNN tensor
            layout: "TILE" or "ROW_MAJOR"

        Returns:
            Tensor with new layout
        """
        if layout == "TILE":
            return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        elif layout == "ROW_MAJOR":
            return ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        return tensor

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - cleanup tracked allocations."""
        self._allocation_tracker.cleanup()
