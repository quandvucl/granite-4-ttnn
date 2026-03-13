import ttnn
import torch
from tt_ops.base import TTOperation, to_tt_tensor, to_torch_tensor, tt_linear

DEBUG = False


class TTSharedMLP(TTOperation):
    """
    Gated MLP with SwiGLU activation (LLaMA-style).

    Architecture:
        gate = gate_proj(x)
        up = up_proj(x)
        activated = silu(gate) * up
        out = down_proj(activated)

    Supports tensor parallelism when weights are sharded across mesh devices.
    """

    def __init__(self, device, hidden_size: int, intermediate_size: int, dtype=ttnn.bfloat16, tensor_parallel: bool = False, tt_ccl=None):
        super().__init__(device, dtype)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tensor_parallel = tensor_parallel
        self.is_mesh = hasattr(device, 'get_num_devices')
        self.tt_ccl = tt_ccl

        # Weights (will be loaded separately)
        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with TTNN optimizations where possible.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            output: [batch, seq, hidden_size]
        """
        if self.tensor_parallel and self.is_mesh:
            return self._forward_tensor_parallel(hidden_states)
        else:
            return self._forward_replicated(hidden_states)

    def _forward_replicated(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with replicated weights (no tensor parallelism)."""
        # For mesh devices, use on-device computation to avoid CPU conversion overhead
        if self.is_mesh:
            return self._forward_on_device(hidden_states)

        # Single device: use PyTorch path (already optimized)
        hidden_states_torch = to_torch_tensor(hidden_states)
        gate_weight_torch = to_torch_tensor(self.gate_proj_weight).T
        up_weight_torch = to_torch_tensor(self.up_proj_weight).T
        down_weight_torch = to_torch_tensor(self.down_proj_weight).T

        # SwiGLU: gate proj + silu × up proj → down proj
        # All in bfloat16 to match hardware precision
        gate = torch.nn.functional.linear(hidden_states_torch, gate_weight_torch)
        gate = torch.nn.functional.silu(gate)

        up = torch.nn.functional.linear(hidden_states_torch, up_weight_torch)

        # Element-wise multiply (gating mechanism)
        activated = gate * up

        # Down projection (output)
        output = torch.nn.functional.linear(activated, down_weight_torch)

        # Convert back to TTNN
        output_tt = to_tt_tensor(output, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        return output_tt

    def _forward_on_device(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass keeping all computation on TTNN device (avoids CPU transfers).

        Optimized for decode mode (single token): minimizes memory operations
        and uses fused kernels where possible.
        """
        # Ensure tile layout for matmuls (required for optimal performance)
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # Gate and up projections (fused where possible)
        # These are independent and could be batched, but TTNN handles this internally
        gate = hidden_states @ self.gate_proj_weight  # [B, S, H] @ [H, I] = [B, S, I]
        up = hidden_states @ self.up_proj_weight      # [B, S, H] @ [H, I] = [B, S, I]

        # SwiGLU activation: silu(gate) * up
        # Using fused operation: applies silu to gate and multiplies with up in one kernel
        activated = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

        # Deallocate intermediate tensors to free memory faster
        gate.deallocate(True)
        up.deallocate(True)

        # Down projection
        output = activated @ self.down_proj_weight  # [B, S, I] @ [I, H] = [B, S, H]

        # Deallocate activated tensor
        activated.deallocate(True)

        return output

    def _forward_tensor_parallel(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with tensor parallelism.

        Column parallel for gate_proj and up_proj (weights sharded on dim 1).
        Row parallel for down_proj (weights sharded on dim 0, followed by all-reduce).

        Input tensors must be replicated across all devices (handled by to_tt_tensor with mesh_mapper).
        """
        DEBUG_TP = False  # Set to True to see if TP is being used
        if DEBUG_TP:
            print(f"[MLP TP] Using tensor parallel path with {self.device.get_num_devices()} devices")

        # Ensure tile layout for matmuls
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # Column parallel: gate_proj and up_proj
        # Input replicated, weights sharded column-wise, output sharded
        gate = hidden_states @ self.gate_proj_weight  # [B, S, H] @ [H, I/N] = [B, S, I/N] on each device
        up = hidden_states @ self.up_proj_weight      # [B, S, H] @ [H, I/N] = [B, S, I/N] on each device

        # SwiGLU activation: silu(gate) * up
        activated = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

        # Row parallel: down_proj
        # Input sharded, weights sharded row-wise, output needs all-reduce
        output_partial = activated @ self.down_proj_weight  # [B, S, I/N] @ [I/N, H] = [B, S, H] partial

        # All-reduce to sum partial results from all devices
        # Convert shape to tuple for later use (ttnn.Shape -> Python tuple)
        original_shape = tuple(output_partial.shape)

        # Reshape to [1, 1, seq, hidden] if needed
        if original_shape[0] != 1 or original_shape[1] != 1:
            output_partial = ttnn.reshape(
                output_partial,
                (1, 1, original_shape[0] * original_shape[1] * original_shape[2], original_shape[3])
            )

        # All-reduce to sum partial results from all devices
        # Manual CPU reduction since native all_reduce doesn't work on this mesh
        # Optimized: minimize copies and use efficient torch operations
        num_devices = self.device.get_num_devices()
        if num_devices > 1:
            import torch

            # Get shards from all devices (each has partial result from row-parallel matmul)
            shards = ttnn.get_device_tensors(output_partial)

            # Fast path: convert first shard, then add remaining shards
            summed = shards[0].cpu().to_torch().clone()
            for i in range(1, len(shards)):
                summed.add_(shards[i].cpu().to_torch())

            # Trim padding if needed (TILE_LAYOUT adds padding)
            if tuple(summed.shape) != original_shape:
                summed = summed.reshape(original_shape)

            # Convert back to TTNN and replicate across all devices
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
            output = ttnn.from_torch(
                summed,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper
            )
        else:
            output = output_partial

        # Reshape back to original shape if we reshaped earlier
        output_shape = tuple(output.shape)
        if output_shape != original_shape:
            output = ttnn.reshape(output, original_shape)

        return output


def split_combined_mlp_weight(combined_weight_tt, device, dtype, hidden_size, intermediate_size):
    """
    Split Granite's combined input_linear weight into gate_proj and up_proj.

    Args:
        combined_weight_tt: TTNN tensor [hidden_size, intermediate_size*2] in TILE layout
        device: TTNN device
        dtype: Data type
        hidden_size: Model hidden dimension
        intermediate_size: MLP intermediate dimension

    Returns:
        (gate_proj_weight, up_proj_weight) as TTNN tensors in TILE layout
    """
    # Convert to PyTorch for splitting
    combined_torch = to_torch_tensor(
        combined_weight_tt,
        target_shape=(hidden_size, intermediate_size * 2)
    )

    # Split along last dimension
    gate_torch = combined_torch[:, :intermediate_size].contiguous()
    up_torch = combined_torch[:, intermediate_size:].contiguous()

    # Convert back to TTNN in TILE layout
    gate_tt = to_tt_tensor(gate_torch, device, dtype, layout=ttnn.TILE_LAYOUT)
    up_tt = to_tt_tensor(up_torch, device, dtype, layout=ttnn.TILE_LAYOUT)

    return gate_tt, up_tt
