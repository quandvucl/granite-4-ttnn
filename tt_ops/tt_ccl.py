"""
Simple TT_CCL wrapper for managing collective communication semaphores.
Based on tt-metal/models/tt_transformers/tt/ccl.py
"""
import ttnn


class SimpleTTCCL:
    """Simplified TT_CCL for managing semaphores in collective operations."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

        # Create core range set for all compute cores
        self.sub_device_crs = ttnn.CoreRangeSet({
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    mesh_device.compute_with_storage_grid_size().x - 1,
                    mesh_device.compute_with_storage_grid_size().y - 1,
                ),
            )
        })

        # Initialize semaphore indices (double-buffered)
        self.barrier_semaphore_idx = 0
        self.rs_semaphores_idx = 0

        # Create semaphore handles (2 sets for double buffering)
        self.barrier_semaphore_handles = [
            ttnn.create_global_semaphore(mesh_device, self.sub_device_crs, 0)
            for _ in range(2)
        ]

        self.rs_semaphore_handles = [
            [ttnn.create_global_semaphore(mesh_device, self.sub_device_crs, 0) for _ in range(3)]
            for _ in range(2)
        ]

    def get_and_cycle_barrier_semaphore_handle(self):
        """Get current barrier semaphore and cycle to next."""
        current_idx = self.barrier_semaphore_idx
        self.barrier_semaphore_idx = (current_idx + 1) % 2
        return self.barrier_semaphore_handles[current_idx]

    def get_and_cycle_rs_semaphore_handles(self):
        """Get current reduce_scatter semaphores and cycle to next."""
        current_idx = self.rs_semaphores_idx
        self.rs_semaphores_idx = (current_idx + 1) % 2
        return self.rs_semaphore_handles[current_idx]
