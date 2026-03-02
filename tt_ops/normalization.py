import ttnn
import torch
from .base import TTOperation, to_tt_tensor, to_torch_tensor, to_tile_layout


class TTRMSNorm(TTOperation):
    """
    RMS Normalization on TTNN with minimized conversions.

    RMSNorm: output = x * rsqrt(mean(x²) + eps) * weight

    This is more efficient than LayerNorm as it doesn't center the data.

    Optimization: Accepts both ttnn and torch tensors to avoid unnecessary conversions
    when the input is already in the right format.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        eps: float = 1e-5,
        dtype=ttnn.bfloat16
    ):
        super().__init__(device, dtype)
        self.eps = eps
        self.device = device

        # Store weight as PyTorch for HF-compatible computation
        self.weight_torch = weight
        self.weight_tt = None  # Not used currently

    def forward(self, hidden_states_tt: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply RMSNorm to hidden states using HF-compatible FP32 intermediate precision.

        Args:
            hidden_states_tt: [batch, seq, hidden_size] (ROW_MAJOR or TILE)

        Returns:
            Normalized tensor [batch, seq, hidden_size] (ROW_MAJOR layout)

        Note: Uses PyTorch implementation to match HF exactly. Native ttnn.rms_norm
        produces slightly different results that break generation.
        """
        # Convert to PyTorch to match HF's FP32 intermediate precision
        hidden_states = to_torch_tensor(hidden_states_tt)

        # HF-compatible RMSNorm: use FP32 intermediate precision
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        output = self.weight_torch * hidden_states.to(input_dtype)

        # Convert back to TTNN (always ROW_MAJOR for consistency)
        output_tt = to_tt_tensor(output, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        return output_tt

    def forward_fused(
        self,
        hidden_states_tt: ttnn.Tensor,
        residual_tt: ttnn.Tensor,
        scale: float
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Fused operation: (residual + hidden_states * scale), then normalize.

        Args:
            hidden_states_tt: Input tensor (on device)
            residual_tt: Residual to add (on device)
            scale: Scale factor for hidden_states

        Returns:
            (updated_hidden, normalized): Both on device
            - updated_hidden: residual + hidden_states * scale (on device)
            - normalized: RMSNorm(updated_hidden) (on device)

        Benefits: Fuses residual add+scale into single kernel, stays on device
        between add and norm operations.
        """
        # Fused residual add + scale (single kernel!)
        from .fused_ops import fused_residual_add_scale
        updated_hidden_tt = fused_residual_add_scale(residual_tt, hidden_states_tt, scale)

        # Normalize (converts to torch, normalizes, converts back)
        normalized_tt = self.forward(updated_hidden_tt)

        return updated_hidden_tt, normalized_tt


class TTRMSNormGated(TTOperation):
    """
    Gated RMS Normalization on TTNN.

    This variant applies a gate activation before normalization.
    Used in some Granite model variants.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        gate_weight: torch.Tensor,
        eps: float = 1e-5,
        dtype=ttnn.bfloat16
    ):
        super().__init__(device, dtype)
        self.eps = eps
        self.device = device

        # Store weights
        self.weight_torch = weight
        self.gate_weight_tt = to_tt_tensor(gate_weight, device, dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, hidden_states_tt: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply gated RMSNorm to hidden states using HF-compatible FP32 intermediate precision.

        Args:
            hidden_states_tt: [batch, seq, hidden_size]

        Returns:
            Normalized tensor [batch, seq, hidden_size] (ROW_MAJOR layout)
        """
        # Convert to PyTorch for gate and normalization
        hidden_states = to_torch_tensor(hidden_states_tt)
        gate_weight = to_torch_tensor(self.gate_weight_tt)

        # Apply gate and SiLU activation
        gated = hidden_states * gate_weight
        gated = torch.nn.functional.silu(gated)

        # HF-compatible RMSNorm: use FP32 intermediate precision
        input_dtype = gated.dtype
        gated = gated.to(torch.float32)
        variance = gated.pow(2).mean(-1, keepdim=True)
        gated = gated * torch.rsqrt(variance + self.eps)
        output = self.weight_torch * gated.to(input_dtype)

        # Convert back to TTNN (always ROW_MAJOR for consistency)
        output_tt = to_tt_tensor(output, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        return output_tt


def test_rmsnorm_cpu_vs_tt():
    """Test RMSNorm implementation against PyTorch CPU version."""
    import torch.nn.functional as F

    batch_size, seq_len, hidden_size = 2, 4, 16
    eps = 1e-5

    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    # CPU reference implementation
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    cpu_output = x_normalized * weight

    # Initialize device and TT implementation
    try:
        device = ttnn.open_device(device_id=0)

        # TT implementation
        tt_norm = TTRMSNorm(device, weight, eps=eps)
        x_tt = to_tt_tensor(x, device)
        tt_output_tt = tt_norm(x_tt)
        tt_output = to_torch_tensor(tt_output_tt, target_shape=(batch_size, seq_len, hidden_size))

        # Compare
        diff = torch.abs(cpu_output - tt_output)
        max_diff = diff.max().item()
        rel_error = (diff / (torch.abs(cpu_output) + 1e-8)).mean().item()

        print(f"RMSNorm Test:")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean relative error: {rel_error:.6f}")
        print(f"  ✓ PASSED" if max_diff < 1e-2 else f"  ✗ FAILED")

        ttnn.close_device(device)
        return max_diff < 1e-2

    except Exception as e:
        print(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    test_rmsnorm_cpu_vs_tt()
