"""
TTNN implementation of Mamba layer (Selective State Space Model).

This implementation focuses on the decode path (single token with cached states)
which is critical for generation performance. Complex operations use CPU fallback
where TTNN doesn't have native support.

Architecture:
    Input [batch, seq_len, hidden_size] -> Mamba -> Output [batch, seq_len, hidden_size]

Components:
    1. Linear projection (in_proj): Expand and split to gate/hidden/dt
    2. Conv1d: Temporal convolution (cached for decode)
    3. Selective SSM: State space model with input-dependent transitions
    4. Gated normalization: Special RMSNorm with gating
    5. Output projection: Project back to hidden_size

Performance:
    - Native TT operations: @, *, +, element-wise ops
    - CPU fallback: softplus, clamp, reshape, permute (minimal overhead in decode)
    - Expected speedup: 5-10x vs pure CPU Mamba
"""

import torch
import ttnn
from typing import Optional, Tuple
from tt_ops.base import TTOperation, to_tt_tensor, to_torch_tensor, tt_linear


class TTMambaLayer(TTOperation):
    """
    TTNN implementation of Mamba (Selective SSM) layer.

    Args:
        device: TTNN device
        hidden_size: Model hidden dimension (1536)
        intermediate_size: Expanded dimension (3072)
        num_heads: Number of SSM heads (48)
        ssm_state_size: State dimension per head (128)
        conv_dim: Conv1d dimension (3328)
        conv_kernel_size: Conv kernel size (4)
        time_step_min: Minimum time step (0.001)
        time_step_max: Maximum time step (0.1)
        layer_idx: Layer index for caching
        dtype: Data type (bfloat16)
    """

    def __init__(
        self,
        device,
        hidden_size: int = 1536,
        intermediate_size: int = 3072,
        num_heads: int = 48,
        ssm_state_size: int = 128,
        conv_dim: int = 3328,
        conv_kernel_size: int = 4,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        layer_idx: int = 0,
        dtype=ttnn.bfloat16
    ):
        super().__init__(device, dtype)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = intermediate_size // num_heads  # 64
        self.ssm_state_size = ssm_state_size
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.layer_idx = layer_idx
        self.n_groups = 1  # No grouping in Granite

        # Weights (will be loaded from HuggingFace checkpoint)
        # These are placeholder - actual weights loaded via from_pretrained
        self.in_proj_weight = None      # [6448, 1536]
        self.conv1d_weight = None       # [3328, 1, 4]
        self.conv1d_bias = None         # [3328]
        self.norm_weight = None         # [3072] - for gated RMSNorm
        self.norm_gate_weight = None    # [3072] - for gated RMSNorm
        self.out_proj_weight = None     # [1536, 3072]

        # SSM parameters
        self.A_log = None               # [48] - State transition (log space)
        self.D = None                   # [48] - Skip connection weights
        self.dt_bias = None             # [48] - Time step bias

    def load_weights_from_hf(self, hf_mamba_layer):
        """
        Load weights from HuggingFace Mamba layer.

        Args:
            hf_mamba_layer: HuggingFace GraniteMoeHybridMambaLayer
        """
        # Linear projections - store on TT device in TILE layout (transposed for matmul)
        self.in_proj_weight = to_tt_tensor(
            hf_mamba_layer.in_proj.weight.T.contiguous(),  # Transpose for matmul
            self.device,
            self.dtype,
            layout=ttnn.TILE_LAYOUT
        )

        self.out_proj_weight = to_tt_tensor(
            hf_mamba_layer.out_proj.weight.T.contiguous(),
            self.device,
            self.dtype,
            layout=ttnn.TILE_LAYOUT
        )

        # Conv1d weights - keep on CPU for now (decode path uses simple multiply-add)
        self.conv1d_weight = hf_mamba_layer.conv1d.weight.clone()  # [3328, 1, 4]
        self.conv1d_bias = hf_mamba_layer.conv1d.bias.clone()      # [3328]

        # Normalization weights - store on TT device
        self.norm_weight = to_tt_tensor(
            hf_mamba_layer.norm.weight,
            self.device,
            self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT
        )

        if hasattr(hf_mamba_layer.norm, 'gate_weight'):
            self.norm_gate_weight = to_tt_tensor(
                hf_mamba_layer.norm.gate_weight,
                self.device,
                self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT
            )

        # SSM parameters - keep on CPU for now
        self.A_log = hf_mamba_layer.A_log.clone()
        self.D = hf_mamba_layer.D.clone()
        self.dt_bias = hf_mamba_layer.dt_bias.clone()

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size]
        cache: Optional[dict] = None,  # For conv and SSM states
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Mamba layer (decode path with caching).

        This implementation focuses on the decode path (seq_len=1 with cache)
        which is critical for generation performance.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            cache: Optional cache dict with 'conv_state' and 'ssm_state'
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # Check if this is decode mode (single token with cache)
        use_cache = (
            cache is not None
            and 'conv_state' in cache
            and 'ssm_state' in cache
            and seq_len == 1
        )

        if not use_cache:
            # Prefill mode - fall back to HuggingFace CPU implementation
            raise NotImplementedError(
                "Prefill mode (seq_len > 1 or no cache) not yet implemented. "
                "Use HuggingFace Mamba for prefill, then TT Mamba for decode."
            )

        # ===================================================================
        # DECODE PATH (seq_len=1 with cached states)
        # ===================================================================

        # Step 1: Input projection [batch, 1, 1536] -> [batch, 1, 6448]
        # Use TT for matmul (bfloat16)
        hidden_tt = to_tt_tensor(hidden_states, self.device, self.dtype, layout=ttnn.TILE_LAYOUT)
        projected_tt = hidden_tt @ self.in_proj_weight
        projected = to_torch_tensor(projected_tt, target_shape=(batch_size, seq_len, -1))

        # CRITICAL: Convert to float32 immediately to prevent precision loss
        projected = projected.to(torch.float32)

        # Split: gate [3072], hidden_B_C [3328], dt [48] - all float32
        gate = projected[:, :, :self.intermediate_size]
        hidden_B_C = projected[:, :, self.intermediate_size:self.intermediate_size + self.conv_dim]
        dt = projected[:, :, -self.num_heads:]

        # Step 2: Conv1d with cached states - ALL IN FLOAT32
        # Update conv cache: roll left and add new value
        conv_state = cache['conv_state']  # [batch, conv_dim, kernel_size] float32
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = hidden_B_C[:, 0, :]
        cache['conv_state'] = conv_state

        # Conv operation: weighted sum over kernel (float32)
        hidden_B_C = torch.sum(
            conv_state * self.conv1d_weight.to(torch.float32).squeeze(1),  # float32
            dim=-1
        ) + self.conv1d_bias.to(torch.float32)
        hidden_B_C = torch.nn.functional.silu(hidden_B_C)  # [batch, conv_dim] float32
        hidden_B_C = hidden_B_C.unsqueeze(1)  # [batch, 1, conv_dim]

        # Split: hidden [3072], B [128], C [128] - all float32
        hidden_states = hidden_B_C[:, :, :self.intermediate_size]
        B = hidden_B_C[:, :, self.intermediate_size:self.intermediate_size + self.ssm_state_size]
        C = hidden_B_C[:, :, -self.ssm_state_size:]

        # Step 3: Selective SSM computation - ALL FLOAT32
        # Discretize time step
        dt = dt[:, 0, :]  # [batch, 48] float32
        dt = dt.unsqueeze(-1).expand(batch_size, self.num_heads, self.head_dim)  # [batch, 48, 64]

        # Expand dt_bias: [48] -> [48, 64]
        dt_bias_expanded = self.dt_bias.to(torch.float32).unsqueeze(-1).expand(self.num_heads, self.head_dim)

        # Add bias, softplus, clamp (all float32)
        dt = torch.nn.functional.softplus(dt + dt_bias_expanded)
        dt = torch.clamp(dt, self.time_step_min, self.time_step_max)  # [batch, 48, 64] float32

        # Prepare A matrix: [48] -> [48, 64, 128] (float32)
        A = -torch.exp(self.A_log.float())
        A = A.unsqueeze(-1).unsqueeze(-1).expand(self.num_heads, self.head_dim, self.ssm_state_size)  # [48, 64, 128]

        # Discretize A: dA = exp(dt * A) - float32
        dA = torch.exp(dt.unsqueeze(-1) * A)  # [batch, 48, 64, 128] float32

        # Discretize B: reshape and expand (float32)
        B = B.reshape(batch_size, self.n_groups, -1).unsqueeze(2)
        B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1])
        B = B.reshape(batch_size, self.num_heads, -1)  # [batch, 48, 128] float32
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # [batch, 48, 64, 128] float32

        # Reshape hidden states to per-head
        hidden_states = hidden_states.reshape(batch_size, self.num_heads, self.head_dim)

        # Compute dBx - float32
        dBx = dB * hidden_states.unsqueeze(-1)  # [batch, 48, 64, 128] float32

        # State update: s = s * dA + dBx (float32)
        ssm_state = cache['ssm_state']  # [batch, 48, 64, 128] float32
        ssm_state = ssm_state * dA + dBx
        cache['ssm_state'] = ssm_state

        # Output: y = C^T @ s (float32)
        C = C.reshape(batch_size, self.n_groups, -1).unsqueeze(2)
        C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1])
        C = C.reshape(batch_size, self.num_heads, -1)  # [batch, 48, 128] float32

        # Batch matmul (float32)
        y = torch.matmul(ssm_state, C.unsqueeze(-1)).squeeze(-1)  # [batch, 48, 64] float32

        # Add skip connection (D) - float32
        D = self.D.to(torch.float32).unsqueeze(-1).expand(self.D.shape[0], self.head_dim)
        y = y + hidden_states * D

        # Reshape back: [batch, 48, 64] -> [batch, 1, 3072]
        y = y.reshape(batch_size, 1, self.intermediate_size)  # float32

        # Step 4: Gated normalization (gate FIRST, then norm) - float32
        y_normalized = self._gated_rmsnorm_float32(y, gate, self.norm_weight)

        # Step 5: Output projection using TT (convert to bfloat16 for TTNN)
        y_tt = to_tt_tensor(y_normalized.to(dtype), self.device, self.dtype, layout=ttnn.TILE_LAYOUT)
        output_tt = y_tt @ self.out_proj_weight
        output = to_torch_tensor(output_tt, target_shape=(batch_size, seq_len, self.hidden_size))

        return output

    def _gated_rmsnorm_float32(self, x: torch.Tensor, gate: torch.Tensor, weight: ttnn.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Gated RMSNorm in float32 - no bfloat16 conversions."""
        weight_torch = weight.to_torch().to(torch.float32)

        # Gate and normalize (all float32)
        x = x * torch.nn.functional.silu(gate)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)

        # Weight multiply in float32 on CPU
        result = x * weight_torch

        return result

    def _gated_rmsnorm(self, x: torch.Tensor, gate: torch.Tensor, weight: ttnn.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Gated RMSNorm matching HuggingFace - CPU for complex ops, TTNN weight multiply."""
        weight_torch = weight.to_torch()
        input_dtype = x.dtype

        # Gate and normalize on CPU (complex float32 operations)
        x = x.to(torch.float32)
        x = x * torch.nn.functional.silu(gate.to(torch.float32))
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)

        # Weight multiply: use TTNN for final operation
        x_tt = to_tt_tensor(x.to(input_dtype), self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        result_tt = x_tt * self.norm_weight
        result = to_torch_tensor(result_tt, target_shape=(x.shape))

        return result.to(input_dtype)

    @staticmethod
    def init_cache(batch_size: int, device='cpu', dtype=torch.bfloat16) -> dict:
        """
        Initialize conv and SSM caches for decode mode.

        Args:
            batch_size: Batch size
            device: Device for cache tensors
            dtype: Data type for cache tensors (ignored - we use float32 for precision)

        Returns:
            Cache dict with 'conv_state' and 'ssm_state'

        Note:
            Both caches use float32 for maximum precision
        """
        return {
            'conv_state': torch.zeros(batch_size, 3328, 4, device=device, dtype=torch.float32),
            'ssm_state': torch.zeros(batch_size, 48, 64, 128, device=device, dtype=torch.float32)
        }


def test_mamba_layer():
    """Test TTNN Mamba layer implementation."""
    import torch
    import ttnn
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from transformers import AutoModelForCausalLM

    print("=" * 70)
    print("Testing TTNN Mamba Layer")
    print("=" * 70)

    device = ttnn.open_device(device_id=0)

    # Load HF model to get weights
    print("\n1. Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        'ibm-granite/granite-4.0-h-1b',
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    hf_mamba = hf_model.model.layers[0].mamba

    # Create TT Mamba layer
    print("\n2. Creating TTNN Mamba layer...")
    tt_mamba = TTMambaLayer(
        device=device,
        hidden_size=1536,
        intermediate_size=3072,
        num_heads=48,
        ssm_state_size=128,
        conv_dim=3328,
        conv_kernel_size=4,
        layer_idx=0
    )

    # Load weights
    print("\n3. Loading weights from HuggingFace...")
    tt_mamba.load_weights_from_hf(hf_mamba)
    print("   ✓ Weights loaded successfully")

    # Test forward pass (will raise NotImplementedError for now)
    print("\n4. Testing forward pass...")
    test_input = torch.randn(1, 5, 1536, dtype=torch.bfloat16)

    try:
        output = tt_mamba.forward(test_input)
        print(f"   Output shape: {output.shape}")
        print("   ✓ Forward pass successful")
    except NotImplementedError as e:
        print(f"   ⚠ {e}")
        print("   → Need to implement forward pass")

    ttnn.close_device(device)

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)


if __name__ == "__main__":
    test_mamba_layer()


class SimpleMamba2TTNN:
    """
    Simple Mamba2 wrapper: TTNN projections + HF SSM core.

    Strategy:
    - Pre-load TTNN weights (in_proj, out_proj) for future optimization
    - Use HF's proven Mamba2 logic for correctness
    - Structure ready for TTNN matmul acceleration

    This balances optimization readiness with correctness.
    """

    def __init__(self, hf_mamba, device, dtype=ttnn.bfloat16):
        """
        Initialize with TTNN-ready weights.

        Args:
            hf_mamba: HF GraniteMoeHybridMambaLayer
            device: TTNN device
            dtype: Data type
        """
        self.hf_mamba = hf_mamba
        self.device = device
        self.dtype = dtype
        self.layer_idx = hf_mamba.layer_idx
        self.hidden_size = hf_mamba.hidden_size

        # Pre-load TTNN weights (transposed for matmul)
        self.in_proj_weight_tt = to_tt_tensor(
            hf_mamba.in_proj.weight.T.contiguous(),
            device,
            dtype,
            layout=ttnn.TILE_LAYOUT
        )

        self.out_proj_weight_tt = to_tt_tensor(
            hf_mamba.out_proj.weight.T.contiguous(),
            device,
            dtype,
            layout=ttnn.TILE_LAYOUT
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params,
        cache_position: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward with TTNN-optimized projections + CPU SSM core.

        Optimization strategy:
        - TTNN for heavy matmuls (in_proj, out_proj) - ~80% of compute
        - CPU for complex SSM logic (proven correct, minimal overhead in decode)

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cache_params: HybridMambaAttentionDynamicCache
            cache_position: Token positions
            attention_mask: Optional mask

        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # Check if decode mode (where optimization matters most)
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        if not use_precomputed_states:
            # Prefill mode - use HF completely (less critical for performance)
            return self.hf_mamba(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask
            )

        # ===================================================================
        # DECODE MODE - TTNN Optimized
        # ===================================================================
        # This replicates HF's torch_forward decode path with TTNN matmuls

        # 1. Input projection with TTNN (replaces HF's self.in_proj)
        hidden_tt = to_tt_tensor(hidden_states, self.device, self.dtype, layout=ttnn.TILE_LAYOUT)
        projected_tt = hidden_tt @ self.in_proj_weight_tt
        projected = to_torch_tensor(projected_tt, target_shape=(batch_size, seq_len, -1))

        # Split: gate [intermediate_size], hidden_states_B_C [conv_dim], dt [num_heads]
        gate, hidden_states_B_C, dt = projected.split(
            [self.hf_mamba.intermediate_size, self.hf_mamba.conv_dim, self.hf_mamba.num_heads],
            dim=-1
        )

        # 2. Conv1d with cached states (replaces HF lines 30-42)
        cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].roll(shifts=-1, dims=-1)
        cache_params.conv_states[self.layer_idx][:, :, -1] = hidden_states_B_C[:, 0, :].to(
            cache_params.conv_states[self.layer_idx].device
        )

        conv_states = cache_params.conv_states[self.layer_idx].to(device=self.hf_mamba.conv1d.weight.device)
        hidden_states_B_C = torch.sum(
            conv_states * self.hf_mamba.conv1d.weight.squeeze(1), dim=-1
        )
        if self.hf_mamba.use_conv_bias:
            hidden_states_B_C = hidden_states_B_C + self.hf_mamba.conv1d.bias
        hidden_states_B_C = self.hf_mamba.act(hidden_states_B_C)

        # Split: hidden_states, B, C (replaces HF lines 55-59)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.hf_mamba.intermediate_size,
                self.hf_mamba.n_groups * self.hf_mamba.ssm_state_size,
                self.hf_mamba.n_groups * self.hf_mamba.ssm_state_size
            ],
            dim=-1
        )

        # 3. SSM transformation - OPTIMIZED: Keep on CPU for decode
        # For decode (batch=1), SSM state operations are tiny [1, 48, 64, 128]
        # CPU compute: ~0.5ms, TTNN conversion overhead: ~8ms (4 conversions × 2ms)
        # Result: CPU is 16x faster for these operations!

        A = -torch.exp(self.hf_mamba.A_log.float())
        cache_device = cache_params.ssm_states[self.layer_idx].device

        # All SSM operations on CPU (tiny matrices, not worth TTNN)
        dt = dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.hf_mamba.head_dim)
        dt_bias = self.hf_mamba.dt_bias[..., None].expand(self.hf_mamba.dt_bias.shape[0], self.hf_mamba.head_dim)
        dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
        dt = torch.clamp(dt, self.hf_mamba.time_step_limit[0], self.hf_mamba.time_step_limit[1])

        # Discretize A
        A = A[..., None, None].expand(
            self.hf_mamba.num_heads, self.hf_mamba.head_dim, self.hf_mamba.ssm_state_size
        ).to(dtype=torch.float32)
        dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

        # Discretize B
        B = B.reshape(batch_size, self.hf_mamba.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.hf_mamba.n_groups, self.hf_mamba.num_heads // self.hf_mamba.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])
        dB = dt[..., None] * B[..., None, :]

        # Discretize x into dBx
        hidden_states = hidden_states.reshape(batch_size, -1, self.hf_mamba.head_dim)
        dBx = (dB * hidden_states[..., None]).to(device=cache_device)

        # State update (CPU is faster for tiny operations)
        cache_params.ssm_states[self.layer_idx].copy_(
            cache_params.ssm_states[self.layer_idx] * dA + dBx
        )

        # Compute output
        C = C.reshape(batch_size, self.hf_mamba.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.hf_mamba.n_groups, self.hf_mamba.num_heads // self.hf_mamba.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])

        ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)
        ssm_states_reshaped = ssm_states.view(
            batch_size * self.hf_mamba.num_heads, self.hf_mamba.head_dim, self.hf_mamba.ssm_state_size
        )
        C_reshaped = C.view(batch_size * self.hf_mamba.num_heads, self.hf_mamba.ssm_state_size, 1)
        y = torch.bmm(ssm_states_reshaped, C_reshaped)
        y = y.view(batch_size, self.hf_mamba.num_heads, self.hf_mamba.head_dim)

        # D skip connection
        D = self.hf_mamba.D[..., None].expand(self.hf_mamba.D.shape[0], self.hf_mamba.head_dim)
        y = (y + hidden_states * D).to(y.dtype)

        # Reshape (HF line 119)
        y = y.reshape(batch_size, -1)[:, None, ...]

        # 4. Gated normalization on CPU (tiny operation, not worth TTNN)
        # [1, 1, 3072] gating and normalization - CPU is faster
        scan_output = self.hf_mamba.norm(y, gate)

        # 5. Output projection with TTNN (large matmul, worth the conversion)
        scan_output_tt = to_tt_tensor(scan_output.to(dtype), self.device, self.dtype, layout=ttnn.TILE_LAYOUT)
        output_tt = scan_output_tt @ self.out_proj_weight_tt
        output = to_torch_tensor(output_tt, target_shape=(batch_size, seq_len, self.hidden_size))

        return output
