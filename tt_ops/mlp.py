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
    """

    def __init__(self, device, hidden_size: int, intermediate_size: int, dtype=ttnn.bfloat16):
        super().__init__(device, dtype)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

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
        # Convert to PyTorch for full computation
        # Native TTNN matmul requires manual program config which is complex
        # For now, use PyTorch for correctness and maintain bfloat16 precision
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
