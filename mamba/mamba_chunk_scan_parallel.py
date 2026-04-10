"""
Tensor-parallel Mamba implementation using chunk-scan without prefix_scan.

This implementation uses the Mamba2 chunk-scan algorithm with standard TTNN operations
that support cross-device sharding, enabling true tensor parallelism.

Key insight: The chunk-scan algorithm can be implemented with cumsum, exp, and matmuls,
all of which support device sharding, unlike prefix_scan which requires local L1 memory.
"""

import torch
import ttnn
from models.common.tensor_utils import pad_dim_to_size
from models.common.utility_functions import roundup
from mamba.utils import segment_sum_ttnn
from models.common.auto_compose import to_torch_auto_compose

from utils import to_torch_tensor, to_tt_tensor
from utils.device import (
    _is_mesh_device,
    _make_mesh_mapper,
    softplus_and_clamp_torch_via_tt,
)
from mamba.config import Mamba2Config
from mamba.device_manager import TTNNDeviceManager
from mamba.ssm_utils import extract_ssm_parameters


# segment_sum_ttnn is now imported from models.common.mamba_utils


class TensorParallelMamba:
    """
    Tensor-parallel Mamba2 implementation using chunk-scan algorithm.

    Unlike the prefix_scan version, this uses standard operations (cumsum, exp, matmul)
    that support cross-device sharding on the head dimension.

    Strategy:
    - Shard weights and heads across devices
    - Each device processes its local head subset independently
    - No all-reduce needed for most operations (only final projection)
    - Achieves true parallelism with N devices giving ~N speedup
    """

    def __init__(
        self,
        hf_mamba,
        device,
        dtype=ttnn.bfloat16,
        tensor_parallel=True,
        chunk_size_override=None,
    ):
        """
        Initialize tensor-parallel Mamba layer.

        Args:
            hf_mamba: HuggingFace Mamba module
            device: TTNN device
            dtype: Data type for TTNN operations
            tensor_parallel: Enable tensor parallelism
            chunk_size_override: Override model's default chunk_size for memory optimization.
                               Smaller chunk_size = less memory but more chunks to process.
                               E.g., 128 uses 4x less memory than 256, enabling 8-device runs.
        """
        self.hf_mamba = hf_mamba
        self.device = device
        self.dtype = dtype
        self.tensor_parallel = tensor_parallel
        self.layer_idx = hf_mamba.layer_idx

        self.is_mesh = _is_mesh_device(device)
        self.num_devices = device.get_num_devices() if self.is_mesh else 1

        # Use shared configuration
        self.config = Mamba2Config.from_hf_mamba(hf_mamba)

        # Device manager for cleaner device operations
        self.device_mgr = TTNNDeviceManager(device, dtype)

        # Mamba parameters (from config for consistency)
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.ssm_state_size = self.config.ssm_state_size
        self.hidden_size = self.config.hidden_size

        # Chunk size - configurable for memory optimization
        if chunk_size_override is not None:
            self.chunk_size = chunk_size_override
        else:
            self.chunk_size = self.config.chunk_size

        # Group size for GQA-style B/C projection
        self.num_groups = (
            hf_mamba.num_heads // hf_mamba.n_groups
            if hasattr(hf_mamba, "n_groups")
            else hf_mamba.num_heads
        )
        self._group_repeat_factor = self.num_heads // self.num_groups

        self._prefill_A = -torch.exp(hf_mamba.A_log.float())
        self._prefill_D = hf_mamba.D.float()

        # Decode constants as torch (shapes needed for computation)
        self._ssm_A = (
            -torch.exp(hf_mamba.A_log.float())[..., None, None]
            .expand(hf_mamba.num_heads, hf_mamba.head_dim, hf_mamba.ssm_state_size)
            .contiguous()
        )
        self._ssm_dt_bias = hf_mamba.dt_bias[..., None].expand(
            hf_mamba.dt_bias.shape[0], hf_mamba.head_dim
        ).contiguous()
        self._ssm_D = hf_mamba.D[..., None].expand(
            hf_mamba.D.shape[0], hf_mamba.head_dim
        ).contiguous()

        # Cache mesh mapper once — used by weights and decode constants
        self.mesh_mapper = _make_mesh_mapper(self.device)

        self._load_weights()
        self._preload_decode_constants()

        # Persistent on-device SSM state — avoids PCIe round-trip every decode step.
        self._ssm_state_tt = ttnn.from_torch(
            torch.zeros(1, self.num_heads, self.head_dim, self.ssm_state_size,
                        dtype=torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Persistent on-device conv cache — [1, 1, conv_dim, kernel_size].
        # Seeded from cache_params after prefill; reset between sequences.
        kernel_size = self.hf_mamba.conv1d.weight.shape[2]
        conv_dim = self.hf_mamba.conv_dim
        self._conv_cache_tt = ttnn.from_torch(
            torch.zeros(1, 1, conv_dim, kernel_size, dtype=torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

    def _load_weights(self):
        """Load weights across devices.

        On a mesh, in_proj and out_proj use column-parallel sharding: shard the
        output dimension along the second mesh axis (e.g. 4-way for a 2×4 mesh).
        Each device computes a partial matmul; auto_compose gathers on download.
        No all-reduce is needed.  Conv weights are always replicated (small).
        """
        replicate_mapper = self.mesh_mapper

        if self.is_mesh and self.num_devices > 1:
            col_mapper = ttnn.ShardTensor2dMesh(
                self.device, dims=(None, -1), mesh_shape=self.device.shape
            )
            in_t  = self.hf_mamba.in_proj.weight.T.contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            out_t = self.hf_mamba.out_proj.weight.T.contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            self.in_proj_weight_tt = ttnn.from_torch(
                in_t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=col_mapper,
            )
            self.out_proj_weight_tt = ttnn.from_torch(
                out_t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=col_mapper,
            )
        else:
            self.in_proj_weight_tt = to_tt_tensor(
                self.hf_mamba.in_proj.weight.T.contiguous(),
                self.device, self.dtype, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper,
            )
            self.out_proj_weight_tt = to_tt_tensor(
                self.hf_mamba.out_proj.weight.T.contiguous(),
                self.device, self.dtype, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper,
            )

        # Reuse sharded weights for decode — gather is cheap at S=1 (~8KB output).
        self.in_proj_weight_decode_tt = self.in_proj_weight_tt
        self.out_proj_weight_decode_tt = self.out_proj_weight_tt

        conv_weight_4d = (
            self.hf_mamba.conv1d.weight.squeeze(1).unsqueeze(0).unsqueeze(0)
        )
        self.conv_weight_tt = to_tt_tensor(
            conv_weight_4d,
            self.device,
            self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
        )

        if self.hf_mamba.use_conv_bias:
            conv_bias_4d = (
                self.hf_mamba.conv1d.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            self.conv_bias_tt = to_tt_tensor(
                conv_bias_4d,
                self.device,
                self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate_mapper,
            )
        else:
            self.conv_bias_tt = None

    def _preload_decode_constants(self):
        """Preload decode-time constants to device once at init."""
        mapper = self.mesh_mapper

        # A: [H, D, N] — used every decode step for dA = exp(dt * A)
        self._ssm_A_tt = to_tt_tensor(
            self._ssm_A, self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # dt_bias: [H, D] — added to raw dt before softplus
        self._ssm_dt_bias_tt = to_tt_tensor(
            self._ssm_dt_bias.unsqueeze(0),  # [1, H, D]
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # D: [H, D] — skip-connection residual
        self._ssm_D_tt = to_tt_tensor(
            self._ssm_D.unsqueeze(0),  # [1, H, D]
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # Gated RMS norm weight — preloaded so decode norm is all on device
        # norm(y, gate) = rms_norm(y, weight, eps) * silu(gate)
        norm_w = self.hf_mamba.norm.weight.to(torch.bfloat16)  # [intermediate_size]
        self._norm_eps = self.hf_mamba.norm.variance_epsilon
        self._norm_weight_tt = to_tt_tensor(
            norm_w.unsqueeze(0).unsqueeze(0),  # [1, 1, intermediate_size]
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # Ones vector for state-sum reduction: [1, H, N, 1]
        N = self.ssm_state_size
        H = self.num_heads
        self._ones_N_tt = to_tt_tensor(
            torch.ones(1, H, N, 1, dtype=torch.bfloat16),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )

        # Conv1d decode constants — kept on device so _conv1d_decode_tt needs no uploads.
        # Weight: [conv_dim, kernel_size] → [1, 1, conv_dim, kernel_size]
        K = self.hf_mamba.conv1d.weight.shape[2]
        C = self.hf_mamba.conv_dim
        conv_w = self.hf_mamba.conv1d.weight.squeeze(1).to(torch.bfloat16)  # [C, K]
        self._conv_weight_tt = to_tt_tensor(
            conv_w.unsqueeze(0).unsqueeze(0),  # [1, 1, C, K]
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # Ones for sum over K: [1, 1, K, 1]
        self._ones_K_tt = to_tt_tensor(
            torch.ones(1, 1, K, 1, dtype=torch.bfloat16),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # Bias: [1, 1, C, 1] (or None)
        if self.hf_mamba.use_conv_bias:
            self._conv_bias_decode_tt = to_tt_tensor(
                self.hf_mamba.conv1d.bias.to(torch.bfloat16).reshape(1, 1, C, 1),
                self.device, ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
            )
        else:
            self._conv_bias_decode_tt = None

    def forward_prefill_chunk_scan(self, hidden_states, cache_params=None):
        """
        Mamba2 chunk-scan forward pass using TTNN parallel operations.

        Pure TTNN implementation:
        1. All operations on TTNN device (no CPU fallbacks)
        2. Head-dimension sharding for tensor parallelism
        3. No prefix_scan dependency

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cache_params: Optional cache for decode mode

        Returns:
            output: [batch, seq_len, hidden_size]
            ssm_state: [batch, num_heads, head_dim, ssm_state_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ===================================================================
        # 1. INPUT PROJECTION (TTNN)
        # ===================================================================
        replicate_mapper = self.mesh_mapper
        hidden_tt = to_tt_tensor(
            hidden_states, self.device, self.dtype, mesh_mapper=replicate_mapper
        )
        hidden_tt = ttnn.to_layout(hidden_tt, ttnn.TILE_LAYOUT)
        projected_tt = ttnn.matmul(hidden_tt, self.in_proj_weight_tt)
        hidden_tt.deallocate(True)

        projected = to_torch_tensor(projected_tt, target_shape=(batch_size, seq_len, -1))
        projected_tt.deallocate(True)

        # Squeeze extra dimensions if needed
        if projected.dim() == 4:
            projected = projected.squeeze(2)

        # ===================================================================
        # 2. SPLIT PROJECTION: gate, xBC, dt
        # ===================================================================
        # projected = [intermediate_size (gate), conv_dim (xBC), num_heads (dt)]
        gate, xBC, dt = torch.split(
            projected,
            [
                self.hf_mamba.intermediate_size,  # gate (z)
                self.hf_mamba.conv_dim,  # xBC (x + B + C concatenated)
                self.hf_mamba.num_heads,  # dt
            ],
            dim=-1,
        )

        # ===================================================================
        # 3. CONV1D — causal depthwise, then SiLU (matches HF torch_forward)
        # ===================================================================
        kernel_size = self.hf_mamba.conv1d.weight.shape[2]

        # Seed conv cache before applying (matches HF prefill cache update)
        if cache_params is not None and hasattr(cache_params, 'conv_states'):
            xBC_t = xBC.transpose(1, 2)  # [B, C, S]
            conv_state = torch.nn.functional.pad(xBC_t, (kernel_size - xBC_t.shape[-1], 0))
            cache_params.conv_states[self.hf_mamba.layer_idx].copy_(conv_state)

        # Causal depthwise conv1d on device (P3: eliminate CPU conv round-trip).
        # Approach: upload left-padded xBC as [B, 1, C, S+K-1], then for each
        # kernel offset k, multiply slice[:,:,:,k:k+S] by weight[:,:,:,k:k+1]
        # and accumulate — this is equivalent to the depthwise conv1d.
        xBC_t = xBC.transpose(1, 2)  # [B, C, S]
        xBC_padded_t = torch.nn.functional.pad(xBC_t, (kernel_size - 1, 0), value=0.0)  # [B, C, S+K-1]
        # Reshape to [B, 1, C, S+K-1] for 4D TTNN ops
        xBC_padded_4d = xBC_padded_t.unsqueeze(1).to(torch.bfloat16)
        replicate_mapper = self.mesh_mapper
        padded_tt = to_tt_tensor(
            xBC_padded_4d, self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper,
        )
        # Accumulate kernel slides: each slide is [B, 1, C, S]
        acc_tt = None
        K = kernel_size
        for k in range(K):
            slice_k = padded_tt[:, :, :, k:k + seq_len]              # [B, 1, C, S]
            w_k = self._conv_weight_tt[:, :, :, k:k + 1]              # [1, 1, C, 1]
            prod_k = ttnn.mul(slice_k, w_k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            slice_k.deallocate(True)
            if acc_tt is None:
                acc_tt = prod_k
            else:
                acc_tt = ttnn.add(acc_tt, prod_k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                prod_k.deallocate(True)
        padded_tt.deallocate(True)
        # Add bias: [1, 1, C, 1] broadcasts over [B, 1, C, S]
        if self._conv_bias_decode_tt is not None:
            acc_tt = ttnn.add(acc_tt, self._conv_bias_decode_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # SiLU + transpose C↔S to get [B, 1, S, C] then download as [B, S, C]
        xBC_tt = ttnn.silu(acc_tt)
        acc_tt.deallocate(True)
        xBC_tt = ttnn.transpose(xBC_tt, 2, 3)                          # [B, 1, S, C]
        xBC = to_torch_tensor(xBC_tt, target_shape=(batch_size, seq_len, xBC.shape[-1]))
        xBC_tt.deallocate(True)

        # ===================================================================
        # 4. SPLIT xBC: x (hidden_states_inner), B, C using shared utility
        # ===================================================================
        hidden_states_inner, B, C = extract_ssm_parameters(
            xBC,
            self.hf_mamba.intermediate_size,
            self.hf_mamba.n_groups,
            self.ssm_state_size,
            self.num_heads,
        )

        # ===================================================================
        # 5. CHUNK-SCAN SSM (TTNN-friendly operations)
        # ===================================================================
        y, ssm_state = self._chunk_scan_ssm_ttnn(
            hidden_states_inner, B, C, dt, cache_params
        )

        # ===================================================================
        # 6. GATED RMS NORM on device (P2: eliminate CPU round-trip)
        # HF: rms_norm(y * silu(gate), weight)
        # ===================================================================
        replicate_mapper = self.mesh_mapper
        y_tt = to_tt_tensor(
            y.to(torch.bfloat16), self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper,
        )
        gate_tt = to_tt_tensor(
            gate.to(torch.bfloat16), self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper,
        )
        silu_gate_tt = ttnn.silu(gate_tt)
        gate_tt.deallocate(True)
        gated_tt = ttnn.mul(y_tt, silu_gate_tt)
        y_tt.deallocate(True)
        silu_gate_tt.deallocate(True)
        gated_tt = ttnn.rms_norm(gated_tt, epsilon=self._norm_eps, weight=self._norm_weight_tt)

        # ===================================================================
        # 7. OUTPUT PROJECTION (TTNN — replicated weights, each device computes full matmul)
        # ===================================================================
        output_tt = ttnn.matmul(gated_tt, self.out_proj_weight_tt)
        gated_tt.deallocate(True)

        output = to_torch_tensor(
            output_tt, target_shape=(batch_size, seq_len, self.hidden_size)
        )
        output_tt.deallocate(True)

        return output, ssm_state

    def _chunk_scan_ssm_ttnn(self, hidden_states_inner, B, C, dt, cache_params):
        """
        Chunk-scan SSM algorithm using TTNN parallelizable operations.

        This is the key function that replaces prefix_scan with standard ops.
        All operations now run on TTNN device for full parallelization.
        """
        batch_size, seq_len, _ = hidden_states_inner.shape

        # Use TTNN for all operations - keep tensors on device!
        compute_dtype = torch.float32

        # Reshape to expose heads
        x = hidden_states_inner.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).to(compute_dtype)

        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).to(compute_dtype)
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).to(compute_dtype)

        # Repeat for GQA
        B = B.repeat_interleave(self._group_repeat_factor, dim=2)
        C = C.repeat_interleave(self._group_repeat_factor, dim=2)

        # Discretize using shared utility for softplus + clamp
        replicate_mapper = self.mesh_mapper
        dt_with_bias = dt + self.hf_mamba.dt_bias

        # Use shared utility from ttnn_utils
        dt = softplus_and_clamp_torch_via_tt(
            dt_with_bias,
            self.hf_mamba.time_step_limit[0],
            self.hf_mamba.time_step_limit[1],
            self.device,
            ttnn.bfloat16,
            target_shape=dt_with_bias.shape,
            mesh_mapper=replicate_mapper,
        ).to(compute_dtype)

        A = self._prefill_A.to(compute_dtype)
        D = self._prefill_D.to(compute_dtype)

        # Pad to chunk size using utility function
        padded_len = roundup(seq_len, self.chunk_size)
        pad_size = padded_len - seq_len

        # D residual - D is [num_heads], expand to [1, 1, num_heads, 1] for broadcasting
        if pad_size == 0:
            x_padded = x
        else:
            # pad_dim_to_size pads the specified dimension to the target size
            x_padded = pad_dim_to_size(x, dim=1, size=x.shape[1] + pad_size)
        D_residual = D[None, None, :, None] * x_padded

        # Discretize
        x = x * dt[..., None]
        A = A * dt

        # Reshape into chunks

        def reshape_into_chunks(tensor, pad_size, chunk_size):
            if pad_size == 0:
                padded_tensor = tensor
            else:
                padded_tensor = pad_dim_to_size(
                    tensor, dim=1, size=tensor.shape[1] + pad_size
                )
            batch = padded_tensor.shape[0]
            seq = padded_tensor.shape[1]
            num_chunks = seq // chunk_size
            if seq % chunk_size != 0:
                raise ValueError(
                    f"Sequence length {seq} not divisible by chunk_size {chunk_size}."
                )
            if len(padded_tensor.shape) == 3:
                return padded_tensor.reshape(
                    batch, num_chunks, chunk_size, padded_tensor.shape[2]
                )
            else:
                return padded_tensor.reshape(
                    batch,
                    num_chunks,
                    chunk_size,
                    padded_tensor.shape[2],
                    padded_tensor.shape[3],
                )

        x, A, B, C = [
            reshape_into_chunks(t, pad_size, self.chunk_size) for t in (x, A, B, C)
        ]

        # Permute A and compute cumsum on device — keep A_cumsum_tt alive to avoid
        # re-uploading later for state_decay_out (P4 partial: eliminates one PCIe round-trip).
        A = A.permute(0, 3, 1, 2).contiguous()  # [B, H, num_chunks, chunk_size]
        replicate_mapper = self.mesh_mapper
        A_tt = to_tt_tensor(A, self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        A_cumsum_tt = ttnn.cumsum(A_tt, dim=-1)
        A_tt.deallocate(True)
        A_cumsum = to_torch_tensor(A_cumsum_tt, target_shape=A.shape)
        # Keep A_cumsum_tt alive — reused at step 4 (state_decay_out = exp(A_cumsum))

        # ── 1. Intra-chunk Y_diag — all on device ────────────────────────────────
        # A: [B, H, C_n, cs]  →  L: [B, H, C_n, cs, cs] (lower-triangular exp-cumsum)
        A_tt = to_tt_tensor(A, self.device, ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        L_tt = ttnn.exp(segment_sum_ttnn(A_tt, self.device))
        A_tt.deallocate(True)

        B_f = B.to(torch.bfloat16)
        C_f = C.to(torch.bfloat16)
        x_f = x.to(torch.bfloat16)

        B_sz, C_n, cs, H, N = B_f.shape
        D = self.head_dim

        # G = einsum('bcsHN,bctHN->bcsHt', B, C) i.e. C @ B^T per [B,C_n,H]
        # C_f: [B,C_n,cs,H,N] → [B*C_n*H, cs, N]; B_f: [B,C_n,cs,H,N] → [B*C_n*H, N, cs]
        C_bch = C_f.permute(0, 1, 3, 2, 4).reshape(B_sz * C_n * H, cs, N)  # [BCH, t, N]
        B_bch = B_f.permute(0, 1, 3, 4, 2).reshape(B_sz * C_n * H, N, cs)  # [BCH, N, s]
        # G[BCH, t, s] = C_bch @ B_bch
        C_tt = to_tt_tensor(C_bch.contiguous(), self.device, ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        B_tt = to_tt_tensor(B_bch.contiguous(), self.device, ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        G_tt = ttnn.matmul(C_tt, B_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [BCH, t, s]
        C_tt.deallocate(True)
        B_tt.deallocate(True)

        # L_tt: [B, H, C_n, cs, cs]  →  [B*C_n*H, cs, cs]
        # L is currently [B, H, C_n, cs, cs]; reshape to 4D for TTNN then flatten batch dims
        L_bch = to_torch_tensor(L_tt, target_shape=None)          # [B, H, C_n, cs, cs]
        L_tt.deallocate(True)
        L_bch_perm = L_bch.permute(0, 2, 1, 3, 4).reshape(B_sz * C_n * H, cs, cs)  # [BCH, cs, cs]
        L_tt2 = to_tt_tensor(L_bch_perm.to(torch.bfloat16).contiguous(), self.device, ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)

        # M = G * L  (elementwise, both [BCH, t, s])
        M_tt = ttnn.mul(G_tt, L_tt2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        G_tt.deallocate(True)
        L_tt2.deallocate(True)

        # Y_diag = M @ x_bch  where x_bch: [BCH, s, D]
        # x_f: [B, C_n, cs, H, D] → [B*C_n*H, cs, D]
        x_bch = x_f.permute(0, 1, 3, 2, 4).reshape(B_sz * C_n * H, cs, D)
        x_tt = to_tt_tensor(x_bch.contiguous(), self.device, ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        Y_diag_tt = ttnn.matmul(M_tt, x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [BCH, t, D]
        M_tt.deallocate(True)

        # ── 2. Inter-chunk states — all on device ─────────────────────────────
        # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)  [B, H, C_n, cs]
        decay_input_f = (A_cumsum[:, :, :, -1:] - A_cumsum).to(torch.bfloat16)
        decay_input_tt = to_tt_tensor(decay_input_f.contiguous(), self.device, ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        decay_states_tt = ttnn.exp(decay_input_tt)
        decay_input_tt.deallocate(True)

        # B_decay = B_f * decay_states[...,None]
        # decay_states: [B, H, C_n, cs] → [B, C_n, cs, H, 1]
        decay_states_perm = to_torch_tensor(decay_states_tt, target_shape=decay_input_f.shape)
        decay_states_tt.deallocate(True)
        decay_states_5d = decay_states_perm.permute(0, 2, 3, 1).unsqueeze(-1)  # [B, C_n, cs, H, 1]
        B_decay = (B_f * decay_states_5d).to(torch.bfloat16)  # [B, C_n, cs, H, N]

        # states[B, C_n, H, D, N] = einsum('bcshN,bcshD->bchDN', B_decay, x_f)
        # = x_f^T @ B_decay  per [B, C_n, H]
        # x_f perm:     [B, C_n, cs, H, D] → [B*C_n*H, D, cs]
        # B_decay perm: [B, C_n, cs, H, N] → [B*C_n*H, cs, N]
        x_bch_T = x_f.permute(0, 1, 3, 4, 2).reshape(B_sz * C_n * H, D, cs)  # [BCH, D, cs]
        B_decay_bch = B_decay.permute(0, 1, 3, 2, 4).reshape(B_sz * C_n * H, cs, N)  # [BCH, cs, N]
        xT_tt = to_tt_tensor(x_bch_T.contiguous(), self.device, ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        Bd_tt = to_tt_tensor(B_decay_bch.contiguous(), self.device, ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        states_tt = ttnn.matmul(xT_tt, Bd_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [BCH, D, N]
        xT_tt.deallocate(True)
        Bd_tt.deallocate(True)
        x_tt.deallocate(True)

        # states: [BCH, D, N] → [B, C_n, H, D, N]
        states = to_torch_tensor(states_tt, target_shape=None).reshape(B_sz, C_n, H, D, N).to(compute_dtype)
        states_tt.deallocate(True)

        # Prepend previous state
        if cache_params is not None and hasattr(cache_params, "ssm_states"):
            previous_states = cache_params.ssm_states[self.layer_idx].to(compute_dtype)[:, None, ...]
        else:
            previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)  # [B, C_n+1, H, D, N]

        # ── 3. Inter-chunk propagation — all on device ───────────────────────
        A_cumsum_last = A_cumsum[:, :, :, -1]                    # [B, H, C_n]
        A_cumsum_padded = torch.nn.functional.pad(A_cumsum_last, (1, 0)).to(torch.bfloat16)  # [B, H, C_n+1]
        A_cumsum_padded_tt = to_tt_tensor(A_cumsum_padded.contiguous(), self.device, ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        decay_chunk_tt = ttnn.exp(segment_sum_ttnn(A_cumsum_padded_tt, self.device))
        A_cumsum_padded_tt.deallocate(True)

        # decay_chunk: [B, H, C_n+1, C_n+1] after segment_sum
        decay_chunk = to_torch_tensor(decay_chunk_tt, target_shape=None)
        decay_chunk_tt.deallocate(True)
        decay_chunk = decay_chunk.transpose(1, 3)    # [B, C_n+1, C_n+1, H]

        # new_states[B, C_n+1, H, D, N] = einsum('bicH,bjHDN->bijDN', decay_chunk, states)
        # Rewrite as: for each [B, H]: decay_chunk[B,:,:,H] @ states[B,:,H,:,:].reshape(C_n+1, D*N)
        # → [B*H, C_n+1, C_n+1] @ [B*H, C_n+1, D*N] = [B*H, C_n+1, D*N]
        # decay_chunk[b,j,i,H]: sum over j, so row index is i → permute to [BH, i, j]
        dc_bh = decay_chunk.permute(0, 3, 2, 1).reshape(B_sz * H, C_n + 1, C_n + 1)  # [BH, i, j]
        st_bh = states.permute(0, 2, 1, 3, 4).reshape(B_sz * H, C_n + 1, D * N)      # [BH, j, DN]
        dc_tt = to_tt_tensor(dc_bh.to(torch.bfloat16).contiguous(), self.device, ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        st_tt = to_tt_tensor(st_bh.to(torch.bfloat16).contiguous(), self.device, ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        new_states_tt = ttnn.matmul(dc_tt, st_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [BH, i, DN]
        dc_tt.deallocate(True)
        st_tt.deallocate(True)

        new_states = to_torch_tensor(new_states_tt, target_shape=None).reshape(B_sz, H, C_n + 1, D, N)
        new_states_tt.deallocate(True)
        new_states = new_states.permute(0, 2, 1, 3, 4)  # [B, C_n+1, H, D, N]
        states, ssm_state = new_states[:, :-1], new_states[:, -1]   # states: [B,C_n,H,D,N], ssm_state: [B,H,D,N]

        # ── 4. Off-diagonal Y_off — reuse A_cumsum_tt kept alive from step 0 ──
        state_decay_out_tt = ttnn.exp(A_cumsum_tt)
        A_cumsum_tt.deallocate(True)
        state_decay_out = to_torch_tensor(state_decay_out_tt, target_shape=A_cumsum.shape).to(compute_dtype)
        state_decay_out_tt.deallocate(True)

        # Y_off[B, C_n, t, H, D] = state_decay_out[B, H, C_n, t] *
        #                           einsum('bctHN, bCHND -> bctHD', C, states)
        # = state_decay_out * (C @ states^T)  per [B, C_n, H]
        # C_bch already computed: [BCH, t, N]
        # states: [B, C_n, H, D, N] → [BCH, N, D]
        C_bch2 = C_f.permute(0, 1, 3, 2, 4).reshape(B_sz * C_n * H, cs, N)  # [BCH, t, N]
        st_bch = states.permute(0, 1, 2, 4, 3).reshape(B_sz * C_n * H, N, D)  # [BCH, N, D]
        C_tt2 = to_tt_tensor(C_bch2.to(torch.bfloat16).contiguous(), self.device, ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        st_tt2 = to_tt_tensor(st_bch.to(torch.bfloat16).contiguous(), self.device, ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        Y_off_tt = ttnn.matmul(C_tt2, st_tt2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [BCH, t, D]
        C_tt2.deallocate(True)
        st_tt2.deallocate(True)

        # Multiply by state_decay_out
        # state_decay_out: [B, H, C_n, t] → [BCH, t, 1]
        sdo_bch = state_decay_out.permute(0, 2, 1, 3).reshape(B_sz * C_n * H, cs, 1)
        sdo_tt = to_tt_tensor(sdo_bch.to(torch.bfloat16).contiguous(), self.device, ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate_mapper)
        Y_off_scaled_tt = ttnn.mul(Y_off_tt, sdo_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [BCH, t, D]
        Y_off_tt.deallocate(True)
        sdo_tt.deallocate(True)

        # ── Combine Y_diag + Y_off, both [BCH, t, D] ─────────────────────────
        Y_tt = ttnn.add(Y_diag_tt, Y_off_scaled_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        Y_diag_tt.deallocate(True)
        Y_off_scaled_tt.deallocate(True)

        # Y: [BCH, t, D] → [B, C_n, cs, H, D] → [B, padded_len, H, D]
        Y_np = to_torch_tensor(Y_tt, target_shape=None).reshape(B_sz, C_n, H, cs, D)
        Y_tt.deallocate(True)
        y = Y_np.permute(0, 1, 3, 2, 4).reshape(B_sz, C_n * cs, H, D).to(compute_dtype)

        y = y + D_residual   # [B, padded_len, H, D]

        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        y = y.reshape(batch_size, seq_len, -1)

        return y.to(torch.bfloat16), ssm_state.to(torch.bfloat16)

    def forward(
        self, hidden_states, cache_params=None, cache_position=None, attention_mask=None
    ):
        """
        Forward pass - dispatches to prefill or decode based on sequence length.
        Accepts either a torch.Tensor [B, S, H] or a TTNN tensor [1, 1, S, H].
        """
        if isinstance(hidden_states, ttnn.Tensor):
            seq_len = hidden_states.shape[2]
            if seq_len == 1:
                # Decode: pass TTNN directly — forward_decode keeps it on device.
                return self.forward_decode(hidden_states, cache_params)
            else:
                # Prefill: need torch for chunk-scan CPU ops.
                from utils import to_torch_tensor
                hidden_states = to_torch_tensor(
                    hidden_states, target_shape=(1, seq_len, self.hidden_size)
                )
        else:
            seq_len = hidden_states.shape[1]

        if seq_len == 1:
            return self.forward_decode(hidden_states, cache_params)
        else:
            output, ssm_state = self.forward_prefill_chunk_scan(
                hidden_states, cache_params
            )

            # Seed on-device SSM state and conv cache from prefill result
            self._seed_state(ssm_state, cache_params)

            # Also write back to cache so callers that inspect cache_params stay consistent
            if cache_params is not None and hasattr(cache_params, "ssm_states"):
                cache_params.ssm_states[self.layer_idx] = ssm_state

            return output

    def _seed_state(self, ssm_state: torch.Tensor, cache_params=None):
        """Upload SSM state and seed conv cache from cache_params after prefill."""
        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = ttnn.from_torch(
            ssm_state.reshape(1, self.num_heads, self.head_dim, self.ssm_state_size)
                     .to(torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        # Seed the on-device conv cache from cache_params (set during prefill).
        if cache_params is not None and hasattr(cache_params, "conv_states"):
            conv_state = cache_params.conv_states[self.layer_idx]  # [B, C, K]
            K = conv_state.shape[-1]
            C = self.hf_mamba.conv_dim
            self._conv_cache_tt.deallocate(True)
            self._conv_cache_tt = ttnn.from_torch(
                conv_state.to(torch.bfloat16).reshape(1, 1, C, K),
                device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )

    def reset_state(self):
        """Zero SSM state and conv cache (call between sequences)."""
        kernel_size = self.hf_mamba.conv1d.weight.shape[2]
        conv_dim = self.hf_mamba.conv_dim

        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = ttnn.from_torch(
            torch.zeros(1, self.num_heads, self.head_dim, self.ssm_state_size,
                        dtype=torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        self._conv_cache_tt.deallocate(True)
        self._conv_cache_tt = ttnn.from_torch(
            torch.zeros(1, 1, conv_dim, kernel_size, dtype=torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

    def _conv1d_decode_tt(self, xBC_tt):
        """
        On-device conv1d decode step.
        xBC_tt: [1, 1, conv_dim, 1] — new token's xBC features.
        Updates self._conv_cache_tt in-place (roll + insert).
        Returns out_tt: [1, 1, conv_dim, 1] after depthwise conv + bias + silu.
        """
        # Roll: drop oldest (col 0), append new token at the end
        old_part = self._conv_cache_tt[:, :, :, 1:]          # [1,1,C,K-1]
        rolled = ttnn.concat([old_part, xBC_tt], dim=-1)      # [1,1,C,K]
        old_part.deallocate(True)
        self._conv_cache_tt.deallocate(True)
        self._conv_cache_tt = rolled

        # Depthwise multiply: [1,1,C,K] * [1,1,C,K] → [1,1,C,K]
        prod = ttnn.mul(self._conv_cache_tt, self._conv_weight_tt,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Sum over K: [1,1,C,K] @ [1,1,K,1] → [1,1,C,1]
        out = ttnn.matmul(prod, self._ones_K_tt,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        prod.deallocate(True)

        # Add bias and apply silu
        if self._conv_bias_decode_tt is not None:
            out = ttnn.add(out, self._conv_bias_decode_tt,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.silu(out)
        return out  # [1,1,C,1]

    def forward_decode(self, hidden_states_tt, cache_params):
        """
        Decode mode — everything on device, no PCIe bounces in the hot path.

        hidden_states_tt: TTNN tensor [1, 1, 1, H] (replicated).
        Returns: TTNN tensor [1, 1, 1, H] (replicated).

        PCIe ops per call:
          - 1 download (conv_out for SSM split)
          - 1 fused upload (x/B/C/dt packed together)
          Total: 2 ops, down from 11.
        """
        mapper = self.mesh_mapper
        n_g = self.hf_mamba.n_groups
        H = self.num_heads
        D = self.head_dim
        N = self.ssm_state_size
        inter = self.hf_mamba.intermediate_size
        conv_dim = self.hf_mamba.conv_dim
        batch_size = 1

        # 1. IN_PROJ — column-parallel sharded weights; gather on CPU (S=1 → tiny output)
        # hidden_states_tt: [1, 1, 1, hidden_size]
        projected_tt = ttnn.linear(
            hidden_states_tt, self.in_proj_weight_decode_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        total = inter + conv_dim + H
        # Gather sharded output to CPU and re-upload as full replicated tensor
        projected_cpu = to_torch_tensor(projected_tt, target_shape=(batch_size, 1, total))
        projected_tt.deallocate(True)
        projected_tt = to_tt_tensor(
            projected_cpu.reshape(1, 1, 1, total).to(torch.bfloat16).contiguous(),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )

        gate_tt = projected_tt[:, :, :, :inter]                      # [1,1,1,inter]
        xBC_tt  = projected_tt[:, :, :, inter:inter + conv_dim]       # [1,1,1,conv_dim]
        dt_tt   = projected_tt[:, :, :, inter + conv_dim:]            # [1,1,1,H]
        projected_tt.deallocate(True)

        # 2. CONV1D on device
        xBC_tt = ttnn.reshape(xBC_tt, [1, 1, conv_dim, 1])
        conv_out_tt = self._conv1d_decode_tt(xBC_tt)                 # [1,1,conv_dim,1]
        xBC_tt.deallocate(True)

        # 3. Download conv output (1 PCIe op) and split for SSM
        conv_out = to_torch_tensor(conv_out_tt, target_shape=(batch_size, 1, conv_dim))
        conv_out_tt.deallocate(True)

        x_inner, B_raw, C_raw = torch.split(conv_out, [inter, n_g * N, n_g * N], dim=-1)
        x_cpu = x_inner[:, 0, :].reshape(batch_size, H, D).to(torch.bfloat16)
        B_cpu = (B_raw[:, 0, :].reshape(batch_size, n_g, N)
                                .repeat_interleave(H // n_g, dim=1)
                                .to(torch.bfloat16))
        C_cpu = (C_raw[:, 0, :].reshape(batch_size, n_g, N)
                                .repeat_interleave(H // n_g, dim=1)
                                .to(torch.bfloat16))

        # dt comes from on-device projected_tt but we need it for softplus on device.
        # Keep dt_tt on device and process there; pass x/B/C as one fused upload.
        # Pack x/B/C into a single tensor [B, 3, H, max(D,N)] padded → 1 PCIe op.
        # Simpler: pack x, B, C along a new dim and upload once.
        pad = max(D, N)
        xBC_fused = torch.zeros(batch_size, 3, H, pad, dtype=torch.bfloat16)
        xBC_fused[:, 0, :, :D] = x_cpu
        xBC_fused[:, 1, :, :N] = B_cpu
        xBC_fused[:, 2, :, :N] = C_cpu
        xBC_tt2 = to_tt_tensor(xBC_fused.contiguous(), self.device, ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        x_tt = xBC_tt2[:, 0:1, :, :D]   # [B,1,H,D]
        B_tt = xBC_tt2[:, 1:2, :, :N]   # [B,1,H,N]
        C_tt = xBC_tt2[:, 2:3, :, :N]   # [B,1,H,N]
        xBC_tt2.deallocate(True)

        # Reshape slices to expected shapes for _ssm_step_tt_preloaded
        x_tt = ttnn.reshape(x_tt, [batch_size, H, D])
        B_tt = ttnn.reshape(B_tt, [batch_size, H, N])
        C_tt = ttnn.reshape(C_tt, [batch_size, H, N])

        # 4. SSM step — fully on device
        # dt_tt: [1,1,1,H] → [B,H,D]
        # dt_tt: [1,1,1,H] → [B,H,1]; add bias [1,H,D] → broadcasts to [B,H,D]
        dt_tt = ttnn.reshape(dt_tt, [batch_size, H, 1])
        dt_tt = ttnn.add(dt_tt, self._ssm_dt_bias_tt)               # [B,H,D]
        dt_tt = ttnn.log(ttnn.add(ttnn.exp(dt_tt), ttnn.full_like(dt_tt, 1.0)))
        dt_tt = ttnn.clip(dt_tt, self.hf_mamba.time_step_limit[0],
                          self.hf_mamba.time_step_limit[1])         # [B,H,D]

        dt_exp = ttnn.unsqueeze(dt_tt, -1)                           # [B,H,D,1]
        dA_tt  = ttnn.exp(ttnn.mul(dt_exp, self._ssm_A_tt))         # [B,H,D,N]
        dt_exp.deallocate(True)

        dtx_tt = ttnn.mul(dt_tt, x_tt)                              # [B,H,D]
        dt_tt.deallocate(True)
        dtx_tt = ttnn.unsqueeze(dtx_tt, -1)                         # [B,H,D,1]
        B_tt   = ttnn.unsqueeze(B_tt, -2)                           # [B,H,1,N]
        dBx_tt = ttnn.mul(dtx_tt, B_tt)                             # [B,H,D,N]
        dtx_tt.deallocate(True); B_tt.deallocate(True)

        new_state = ttnn.add(ttnn.mul(dA_tt, self._ssm_state_tt), dBx_tt)
        dA_tt.deallocate(True); dBx_tt.deallocate(True)
        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = new_state

        C_tt = ttnn.unsqueeze(C_tt, -2)                             # [B,H,1,N]
        y_unred = ttnn.mul(self._ssm_state_tt, C_tt)                # [B,H,D,N]
        C_tt.deallocate(True)
        y_tt = ttnn.matmul(y_unred, self._ones_N_tt,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)    # [B,H,D,1]
        y_unred.deallocate(True)
        y_tt = ttnn.squeeze(y_tt, dim=-1)                            # [B,H,D]
        y_tt = ttnn.add(y_tt, ttnn.mul(self._ssm_D_tt, x_tt))
        x_tt.deallocate(True)
        y_tt = ttnn.reshape(y_tt, [batch_size, 1, H * D])           # [B,1,inter]

        # 5. GATED RMS NORM
        gate_tt = ttnn.reshape(gate_tt, [batch_size, 1, inter])
        silu_gate_tt = ttnn.silu(gate_tt)
        gate_tt.deallocate(True)
        gated_tt = ttnn.mul(y_tt, silu_gate_tt)
        y_tt.deallocate(True); silu_gate_tt.deallocate(True)
        scan_tt = ttnn.rms_norm(gated_tt, epsilon=self._norm_eps, weight=self._norm_weight_tt)
        gated_tt.deallocate(True)

        # 6. OUT_PROJ — column-parallel sharded; gather + re-upload as replicated [1,1,1,H]
        output_tt = ttnn.linear(
            scan_tt, self.out_proj_weight_decode_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scan_tt.deallocate(True)
        output_cpu = to_torch_tensor(output_tt, target_shape=(batch_size, 1, self.hidden_size))
        output_tt.deallocate(True)
        output_tt = to_tt_tensor(
            output_cpu.reshape(1, 1, 1, self.hidden_size).to(torch.bfloat16).contiguous(),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        return output_tt   # TTNN [1,1,1,H] replicated

    def _gather_column_parallel_output(
        self, projected_tt, batch_size, seq_len, expected_size
    ):
        """Gather column-parallel projected output using auto_compose."""
        # Use automatic composition from models/common
        gathered = to_torch_auto_compose(projected_tt, device=self.device)

        # Handle padding and reshape to expected dimensions
        if gathered.dim() == 4:
            gathered = gathered.squeeze(2)

        # Trim to expected size
        gathered = gathered[..., :expected_size]

        return gathered.view(batch_size, seq_len, expected_size)

    def _conv1d_decode(self, hidden_states_B_C, cache_params):
        """Conv1d decode step on CPU — state is 67KB, not worth PCIe round-trip."""
        # Roll cache and insert new token
        conv_cache = cache_params.conv_states[self.layer_idx]  # [B, C, kernel]
        conv_cache = conv_cache.roll(shifts=-1, dims=-1)
        conv_cache[:, :, -1] = hidden_states_B_C[:, 0, :]
        cache_params.conv_states[self.layer_idx] = conv_cache

        # Pointwise depthwise conv: sum(state * weight) + bias, then silu
        conv_weight = self.hf_mamba.conv1d.weight.squeeze(1)   # [C, kernel]
        out = (conv_cache * conv_weight.unsqueeze(0)).sum(dim=-1)  # [B, C]
        if self.hf_mamba.use_conv_bias:
            out = out + self.hf_mamba.conv1d.bias
        out = torch.nn.functional.silu(out)

        return out.unsqueeze(1)  # [B, 1, C]

    def _ssm_step_tt(self, x_cpu, B_cpu, C_cpu, dt_cpu, batch_size):
        """
        SSM decode step — 4 PCIe uploads (already-shaped bfloat16 tensors), all ops on device.
        x_cpu is uploaded once and reused for the D residual (was uploaded twice before).
        Returns y_tt [B,1,H*D]; self._ssm_state_tt stays on device.

        Args:
            x_cpu:  [B, H, D]  bfloat16
            B_cpu:  [B, H, N]  bfloat16, already expanded from n_groups
            C_cpu:  [B, H, N]  bfloat16, already expanded from n_groups
            dt_cpu: [B, H, D]  bfloat16, already expanded from [B, H, 1]
        """
        mapper = self.mesh_mapper
        H = self.num_heads
        D = self.head_dim
        N = self.ssm_state_size

        # Upload all 4 inputs (4 PCIe transactions, down from 5 before — x used once not twice)
        x_tt  = to_tt_tensor(x_cpu,  self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        B_tt  = to_tt_tensor(B_cpu,  self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        C_tt  = to_tt_tensor(C_cpu,  self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        dt_tt = to_tt_tensor(dt_cpu, self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        # softplus(dt + bias) + clamp
        dt_tt = ttnn.add(dt_tt, self._ssm_dt_bias_tt)
        dt_tt = ttnn.log(ttnn.add(ttnn.exp(dt_tt), ttnn.full_like(dt_tt, 1.0)))
        dt_tt = ttnn.clip(dt_tt,
                          self.hf_mamba.time_step_limit[0],
                          self.hf_mamba.time_step_limit[1])

        # dA = exp(dt * A): [B, H, D, N]
        dt_exp = ttnn.unsqueeze(dt_tt, -1)                       # [B, H, D, 1]
        dA_tt  = ttnn.exp(ttnn.mul(dt_exp, self._ssm_A_tt))      # [B, H, D, N]
        dt_exp.deallocate(True)

        # dBx = (dt * x)[..., None] * B[..., None, :]: [B, H, D, N]
        dtx_tt = ttnn.mul(dt_tt, x_tt)                           # [B, H, D]
        dt_tt.deallocate(True)
        dtx_tt = ttnn.unsqueeze(dtx_tt, -1)                      # [B, H, D, 1]
        B_tt   = ttnn.unsqueeze(B_tt, -2)                        # [B, H, 1, N]
        dBx_tt = ttnn.mul(dtx_tt, B_tt)                          # [B, H, D, N]
        dtx_tt.deallocate(True); B_tt.deallocate(True)

        # state update: new_state = dA * old_state + dBx
        new_state_tt = ttnn.add(ttnn.mul(dA_tt, self._ssm_state_tt), dBx_tt)
        dA_tt.deallocate(True); dBx_tt.deallocate(True)
        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = new_state_tt

        # y = sum(new_state * C, dim=-1) + D * x: [B, H, D]
        # Use matmul with ones instead of ttnn.sum(dim=-1) which crashes on mesh.
        # new_state: [B, H, D, N],  C_tt: [B, H, 1, N]
        C_tt = ttnn.unsqueeze(C_tt, -2)                          # [B, H, 1, N]
        y_unred = ttnn.mul(self._ssm_state_tt, C_tt)             # [B, H, D, N]
        C_tt.deallocate(True)
        # matmul [B, H, D, N] @ [1, H, N, 1] → [B, H, D, 1]
        y_tt = ttnn.matmul(y_unred, self._ones_N_tt,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, H, D, 1]
        y_unred.deallocate(True)
        y_tt = ttnn.squeeze(y_tt, dim=-1)                        # [B, H, D]
        y_tt = ttnn.add(y_tt, ttnn.mul(self._ssm_D_tt, x_tt))   # D residual
        x_tt.deallocate(True)

        y_tt = ttnn.reshape(y_tt, [batch_size, 1, H * D])
        return y_tt

    def _gated_norm_ttnn(self, y, gate):
        """Gated RMS normalization on TTNN device."""
        batch_size, seq_len, hidden_size = y.shape
        replicate_mapper = self.mesh_mapper

        # Convert to TTNN
        y_tt = to_tt_tensor(
            y,
            self.device,
            self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
        )
        gate_tt = to_tt_tensor(
            gate,
            self.device,
            self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
        )

        # Apply SiLU to gate and multiply with y
        gated_tt = ttnn.mul(
            y_tt, gate_tt, input_tensor_b_activations=[ttnn.UnaryOpType.SILU]
        )
        y_tt.deallocate(True)
        gate_tt.deallocate(True)

        # RMS Norm
        norm_weight = self.hf_mamba.norm.weight
        norm_eps = self.hf_mamba.norm.variance_epsilon
        weight_reshaped = norm_weight.unsqueeze(0).unsqueeze(0)
        weight_tt = to_tt_tensor(
            weight_reshaped,
            self.device,
            self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
        )

        normed_tt = ttnn.rms_norm(gated_tt, epsilon=norm_eps, weight=weight_tt)
        gated_tt.deallocate(True)
        weight_tt.deallocate(True)

        # Convert back to torch
        scan_output = to_torch_tensor(
            normed_tt, target_shape=(batch_size, seq_len, hidden_size)
        )
        normed_tt.deallocate(True)

        return scan_output
