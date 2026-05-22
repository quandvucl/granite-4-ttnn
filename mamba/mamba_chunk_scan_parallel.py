"""
Tensor-parallel Mamba implementation using chunk-scan without prefix_scan.

This implementation uses the Mamba2 chunk-scan algorithm with standard TTNN operations
that support cross-device sharding, enabling true tensor parallelism.

Key insight: The chunk-scan algorithm can be implemented with cumsum, exp, and matmuls,
all of which support device sharding, unlike prefix_scan which requires local L1 memory.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tt-metal"))

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

        On a mesh, in_proj and out_proj are column-parallel sharded across the
        last mesh axis (cluster_axis=1, e.g. 4-way for a 2×4 mesh).
        Each device computes a partial matmul; all_gather(cluster_axis=1) gathers
        on fabric without touching the host.  Conv weights are replicated (small).
        """
        replicate_mapper = self.mesh_mapper

        if self.is_mesh and self.num_devices > 1:
            # Shard output dim across the last mesh axis (cols of the 2D mesh).
            mesh_shape = self.device.shape  # e.g. MeshShape(2, 4)
            col_devices = mesh_shape[1]     # number of devices along col axis
            col_mapper = ttnn.ShardTensor2dMesh(
                self.device, dims=(None, -1), mesh_shape=ttnn.MeshShape(1, col_devices)
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

        self.in_proj_weight_decode_tt = self.in_proj_weight_tt
        self.out_proj_weight_decode_tt = self.out_proj_weight_tt

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
        N = self.ssm_state_size
        H = self.num_heads
        D = self.head_dim


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

        # Prefill-specific constants — same values as decode but in shapes used by
        # _chunk_scan_ssm_ttnn so we avoid 108 uploads per forward pass (3 × 36 layers).
        # _prefill_A: [H] → [1, 1, H]
        self._prefill_A_tt = to_tt_tensor(
            self._prefill_A.to(torch.bfloat16).reshape(1, 1, H),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # _prefill_D: [H] → [1, 1, H, 1]
        self._prefill_D_tt = to_tt_tensor(
            self._prefill_D.to(torch.bfloat16).reshape(1, 1, H, 1),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # dt_bias for prefill: [H] → [1, 1, H]  (broadcast over seq dim)
        self._prefill_dt_bias_tt = to_tt_tensor(
            self.hf_mamba.dt_bias.to(torch.bfloat16).reshape(1, 1, H),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # Conv prefill left-pad zeros: [1, 1, C, K-1] — same every call, preloaded once.
        self._prefill_conv_pad_tt = to_tt_tensor(
            torch.zeros(1, 1, C, K - 1, dtype=torch.bfloat16),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        # zero_col for sub-step 3: [1, H, 1] prepended to A_cumsum_last.
        # Preloaded to avoid one PCIe upload per Mamba layer.
        self._seg_zero_col_tt = to_tt_tensor(
            torch.zeros(1, H, 1, dtype=torch.bfloat16),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )

        # Max-size zero pads for _pad_seq — avoids PCIe upload per chunk-scan call.
        # Worst case: pad_size = cs - 1 tokens; each call uploads 5 tensors.
        cs = self.chunk_size
        D = self.head_dim
        N = self.ssm_state_size
        _max_pad = cs  # slightly over-allocate so slice always works
        self._pad_H_tt = to_tt_tensor(
            torch.zeros(_max_pad, H, dtype=torch.bfloat16).reshape(1, _max_pad, H),
            self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._pad_HD_tt = to_tt_tensor(
            torch.zeros(_max_pad, H, D, dtype=torch.bfloat16).reshape(1, _max_pad, H, D),
            self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._pad_HN_tt = to_tt_tensor(
            torch.zeros(_max_pad, H, N, dtype=torch.bfloat16).reshape(1, _max_pad, H, N),
            self.device, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )

    def forward_prefill_chunk_scan(self, hidden_states, cache_params=None, real_seq_len=None):
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
        replicate_mapper = self.mesh_mapper
        inter   = self.hf_mamba.intermediate_size
        conv_d  = self.hf_mamba.conv_dim
        n_heads = self.hf_mamba.num_heads

        _owns_hidden_tt = True
        if isinstance(hidden_states, ttnn.Tensor):
            # Already on device as [1, 1, S, H] — no PCIe upload needed.
            # shape[2] may be tile-padded to 32; use real_seq_len from caller.
            batch_size = hidden_states.shape[0]
            seq_len = real_seq_len if real_seq_len is not None else hidden_states.shape[2]
            hidden_tt = hidden_states
            _owns_hidden_tt = False  # caller owns this tensor; don't deallocate
        else:
            batch_size, seq_len, _ = hidden_states.shape
            hidden_tt = to_tt_tensor(
                hidden_states, self.device, self.dtype, mesh_mapper=replicate_mapper
            )
            hidden_tt = ttnn.to_layout(hidden_tt, ttnn.TILE_LAYOUT)

        # ===================================================================
        # 1. INPUT PROJECTION (TTNN) — all on device, no PCIe download
        # ===================================================================
        projected_tt = ttnn.matmul(hidden_tt, self.in_proj_weight_tt)
        if _owns_hidden_tt:
            hidden_tt.deallocate(True)
        if self.is_mesh and self.num_devices > 1:
            projected_tt = ttnn.all_gather(
                projected_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # projected_tt: [1, 1, S_padded, inter+conv_d+n_heads] — split on device
        # When input was a TTNN tensor, S_padded may be tile-aligned (>=32).
        # Trim to real seq_len before slicing columns.
        # ===================================================================
        # 2. SPLIT PROJECTION: gate, xBC, dt — on device via slicing
        # ===================================================================
        padded_s = projected_tt.shape[2]
        if padded_s != seq_len:
            projected_tt = projected_tt[:, :, :seq_len, :]
        gate_tt = projected_tt[:, :, :, :inter]                         # [1,1,S,inter]
        xBC_tt  = projected_tt[:, :, :, inter:inter + conv_d]           # [1,1,S,conv_d]
        dt_tt   = projected_tt[:, :, :, inter + conv_d:]                # [1,1,S,n_heads]
        projected_tt.deallocate(True)

        # Reshape for downstream: [1,1,S,X] → [B,S,X]  (B=1 always for prefill)
        gate_tt = ttnn.reshape(gate_tt, [batch_size, seq_len, inter])
        xBC_tt  = ttnn.reshape(xBC_tt,  [batch_size, seq_len, conv_d])
        dt_tt   = ttnn.reshape(dt_tt,   [batch_size, seq_len, n_heads])

        # ===================================================================
        # 3. CONV1D — causal depthwise, then SiLU — fully on device, no PCIe
        # ===================================================================
        kernel_size = self.hf_mamba.conv1d.weight.shape[2]

        # Seed conv cache: download only the last (kernel_size-1) tokens of xBC
        # — tiny transfer (~1 KB) vs the full xBC (~390 KB for 176 tokens).
        if cache_params is not None and hasattr(cache_params, 'conv_states'):
            tail_len = min(kernel_size - 1, seq_len)
            tail_tt = xBC_tt[:, seq_len - tail_len:, :]                # [B, tail, conv_d]
            if self.is_mesh and self.num_devices > 1:
                tail_cpu = ttnn.to_torch(
                    tail_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)
                )[0:1].reshape(batch_size, tail_len, conv_d)
            else:
                tail_cpu = to_torch_tensor(tail_tt, target_shape=(batch_size, tail_len, conv_d))
            tail_tt.deallocate(True)
            tail_t = tail_cpu.transpose(1, 2)                           # [B, conv_d, tail]
            conv_state = torch.nn.functional.pad(
                tail_t, (kernel_size - 1 - tail_len, 0)
            )                                                           # [B, conv_d, K-1]
            cache_params.conv_states[self.hf_mamba.layer_idx].copy_(
                torch.cat([conv_state,
                           torch.zeros(batch_size, conv_d, 1, dtype=conv_state.dtype)],
                          dim=-1)[:, :, -(kernel_size):]
            )

        # xBC_tt: [B, S, conv_d] → [B, 1, conv_d, S] for conv kernel
        xBC_tt = ttnn.reshape(xBC_tt, [batch_size, 1, seq_len, conv_d])
        xBC_tt = ttnn.transpose(xBC_tt, 2, 3)                          # [B, 1, conv_d, S]

        # Left-pad with K-1 context tokens.
        # _prefill_conv_pad_tt is zeros for the first chunk.
        # _seed_state updates it with the true historical context after each chunk,
        # so subsequent chunks see the correct causal boundary.
        padded_tt = ttnn.concat([self._prefill_conv_pad_tt, xBC_tt], dim=3,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xBC_tt.deallocate(True)

        # Weighted sum over K shifts: acc = sum_k( padded[:,k:k+S] * w[k] )
        # Each slice is [B,1,conv_d,S]; weight scalar w[k] = _conv_weight_tt[:,:,:,k].
        # Avoids a K-dim permute that creates a [BCD,S,K,32] tile (2 GB for large models).
        acc_tt = None
        w = self._conv_weight_tt                                         # [1,1,conv_d,K]
        for k in range(kernel_size):
            s = padded_tt[:, :, :, k:k + seq_len]                       # [B,1,conv_d,S]
            wk = ttnn.reshape(w[:, :, :, k:k+1], [1, 1, conv_d, 1])    # [1,1,conv_d,1]
            term = ttnn.mul(s, wk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s.deallocate(True)
            if acc_tt is None:
                acc_tt = term
            else:
                acc_tt = ttnn.add(acc_tt, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                term.deallocate(True)
        padded_tt.deallocate(True)
        # acc_tt: [B,1,conv_d,S]

        # acc_tt is already [B,1,conv_d,S] from the weighted sum above

        if self._conv_bias_decode_tt is not None:
            acc_tt = ttnn.add(acc_tt, self._conv_bias_decode_tt,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SiLU → [B, 1, conv_d, S] → [B, 1, S, conv_d] → [B, S, conv_d]
        xBC_tt = ttnn.silu(acc_tt)
        acc_tt.deallocate(True)
        xBC_tt = ttnn.transpose(xBC_tt, 2, 3)                          # [B, 1, S, conv_d]
        xBC_tt = ttnn.reshape(xBC_tt, [batch_size, seq_len, conv_d])   # [B, S, conv_d]

        # ===================================================================
        # 4. SPLIT xBC on device: x, B_raw, C_raw
        # ===================================================================
        # inter = H*D, then n_g*N for B, n_g*N for C
        n_g = self.hf_mamba.n_groups
        N   = self.ssm_state_size
        H   = self.num_heads
        group_repeat = H // n_g

        x_tt    = xBC_tt[:, :, :inter]                                 # [B,S,inter]
        B_raw_tt = xBC_tt[:, :, inter:inter + n_g * N]                 # [B,S,n_g*N]
        C_raw_tt = xBC_tt[:, :, inter + n_g * N:]                      # [B,S,n_g*N]
        xBC_tt.deallocate(True)

        # Reshape to head-decomposed: [B,S,inter] → [B,S,H,D]
        x_tt = ttnn.reshape(x_tt, [batch_size, seq_len, H, self.head_dim])
        # B/C: [B,S,n_g,N] then expand each group to group_repeat heads → [B,S,H,N]
        B_tt = ttnn.reshape(B_raw_tt, [batch_size, seq_len, n_g, N])
        C_tt = ttnn.reshape(C_raw_tt, [batch_size, seq_len, n_g, N])
        B_raw_tt.deallocate(True); C_raw_tt.deallocate(True)
        if group_repeat > 1:
            # repeat_interleave via unsqueeze+repeat+reshape — mesh-safe
            B_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(B_tt, 3), ttnn.Shape([1, 1, 1, group_repeat, 1])), [batch_size, seq_len, H, N])
            C_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(C_tt, 3), ttnn.Shape([1, 1, 1, group_repeat, 1])), [batch_size, seq_len, H, N])

        # ===================================================================
        # 5. CHUNK-SCAN SSM — passes TTNN tensors, returns TTNN y_tt + cpu ssm_state
        # ===================================================================
        y_tt, ssm_state = self._chunk_scan_ssm_ttnn(
            x_tt, B_tt, C_tt, dt_tt, cache_params
        )
        x_tt.deallocate(True); B_tt.deallocate(True)
        C_tt.deallocate(True); dt_tt.deallocate(True)

        # ===================================================================
        # 6. GATED RMS NORM — y_tt and gate_tt already on device
        # ===================================================================
        silu_gate_tt = ttnn.silu(gate_tt)
        gate_tt.deallocate(True)
        gated_tt = ttnn.mul(y_tt, silu_gate_tt)
        y_tt.deallocate(True); silu_gate_tt.deallocate(True)
        gated_tt = ttnn.rms_norm(gated_tt, epsilon=self._norm_eps,
                                 weight=self._norm_weight_tt)

        # ===================================================================
        # 7. OUTPUT PROJECTION — column-parallel; gather shards on fabric.
        # ===================================================================
        output_tt = ttnn.matmul(gated_tt, self.out_proj_weight_tt)
        gated_tt.deallocate(True)
        if self.is_mesh and self.num_devices > 1:
            output_tt = ttnn.all_gather(
                output_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Return TTNN tensor directly — no PCIe download needed.
        # decoder_layer._mamba_forward checks isinstance(out, ttnn.Tensor) and returns it.
        # Reshape to [1,1,S,H] so it matches the hidden_states layout in decoder_layer.forward.
        output_tt = ttnn.reshape(output_tt, [1, 1, seq_len, self.hidden_size])
        return output_tt, ssm_state

    def _chunk_scan_ssm_ttnn(self, x_tt, B_tt, C_tt, dt_tt, cache_params):
        """
        Chunk-scan SSM — fully on device, zero CPU round-trips.

        Inputs (all TTNN tensors):
          x_tt  : [B, S, H, D]
          B_tt  : [B, S, H, N]
          C_tt  : [B, S, H, N]
          dt_tt : [B, S, H]
        Returns:
          y_tt      : TTNN [B, S, H*D]
          ssm_state : torch [B, H, D, N] (small, downloaded once for cache)

        Key invariant: A_cumsum_tt is computed once and kept alive through all 4
        sub-steps (intra-chunk L, inter-chunk states, propagation, Y_off scaling).
        """
        B_sz    = x_tt.shape[0]
        seq_len = x_tt.shape[1]
        H       = self.num_heads
        Dh      = self.head_dim
        N       = self.ssm_state_size
        rm      = self.mesh_mapper

        # ── Adaptive chunk size ─────────────────────────────────────────────
        tile = 32
        seq_aligned = max(tile, roundup(seq_len, tile))
        cs = seq_aligned if seq_aligned < self.chunk_size else self.chunk_size
        padded_len = roundup(seq_len, cs)
        pad_size   = padded_len - seq_len
        C_n        = padded_len // cs

        # ── Discretize dt on device ─────────────────────────────────────────
        dt_tt = ttnn.add(dt_tt, self._prefill_dt_bias_tt,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_tt = ttnn.softplus(dt_tt, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lo, hi = self.hf_mamba.time_step_limit
        dt_tt = ttnn.clip(dt_tt, lo, hi)                               # [B,S,H]

        # ── D residual — uses original (pre-discretized) x ─────────────────
        D_residual_tt = ttnn.mul(self._prefill_D_tt, x_tt,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B,S,H,D]

        # ── A*dt per token ──────────────────────────────────────────────────
        A_dt_tt = ttnn.mul(self._prefill_A_tt, dt_tt,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)      # [B,S,H]

        # ── Discretize x: x *= dt ───────────────────────────────────────────
        dt_exp = ttnn.unsqueeze(dt_tt, -1)                             # [B,S,H,1]
        x_tt   = ttnn.mul(x_tt, dt_exp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_tt.deallocate(True); dt_exp.deallocate(True)                # x_tt: [B,S,H,D]

        # ── Pad all tensors to padded_len ───────────────────────────────────
        def _pad_seq(t_tt, extra_cols, preloaded_pad):
            if extra_cols == 0:
                return t_tt
            pad = preloaded_pad[:, :extra_cols, ...]
            return ttnn.concat([t_tt, pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        A_dt_tt       = _pad_seq(A_dt_tt,       pad_size, self._pad_H_tt)   # [B,padded,H]
        x_tt          = _pad_seq(x_tt,          pad_size, self._pad_HD_tt)  # [B,padded,H,D]
        D_residual_tt = _pad_seq(D_residual_tt, pad_size, self._pad_HD_tt)  # [B,padded,H,D]
        B_tt          = _pad_seq(B_tt,          pad_size, self._pad_HN_tt)  # [B,padded,H,N]
        C_tt          = _pad_seq(C_tt,          pad_size, self._pad_HN_tt)  # [B,padded,H,N]

        # ── Reshape A_dt into chunks: [B,padded,H] → [B,H,C_n,cs] ─────────
        A_dt_tt = ttnn.reshape(A_dt_tt, [B_sz, C_n, cs, H])
        A_dt_tt = ttnn.permute(A_dt_tt, [0, 3, 1, 2])                 # [B,H,C_n,cs]

        # ── A_cumsum — kept alive through ALL 4 sub-steps ───────────────────
        # Used in: sub-step 1 (segment_sum for L), sub-step 2 (decay_states),
        # sub-step 3 (A_cumsum_last), sub-step 4 (state_decay_out).
        A_cumsum_tt = ttnn.cumsum(A_dt_tt, dim=-1)                    # [B,H,C_n,cs]

        # ── Sub-step 1: Intra-chunk Y_diag ──────────────────────────────────
        L_tt = ttnn.exp(segment_sum_ttnn(A_dt_tt, self.device))       # [B,H,C_n,cs,cs]
        A_dt_tt.deallocate(True)

        # [B,H,C_n,cs,cs] → [B,C_n,H,cs,cs] → [BCH,cs,cs]
        L_tt = ttnn.permute(L_tt, [0, 2, 1, 3, 4])
        L_tt = ttnn.reshape(L_tt, [B_sz * C_n * H, cs, cs])

        # x: [B,padded,H,D] → [B,C_n,cs,H,D] → [B,C_n,H,cs,D] → [BCH,cs,D]
        x_tt = ttnn.reshape(x_tt, [B_sz, C_n, cs, H, Dh])
        x_tt = ttnn.permute(x_tt, [0, 1, 3, 2, 4])
        x_tt = ttnn.reshape(x_tt, [B_sz * C_n * H, cs, Dh])

        # B: [B,padded,H,N] → [B,C_n,cs,H,N] → [B,C_n,H,N,cs] → [BCH,N,cs]
        B_tt = ttnn.reshape(B_tt, [B_sz, C_n, cs, H, N])
        B_tt = ttnn.permute(B_tt, [0, 1, 3, 4, 2])
        B_tt = ttnn.reshape(B_tt, [B_sz * C_n * H, N, cs])

        # C: [B,padded,H,N] → [B,C_n,cs,H,N] → [B,C_n,H,cs,N] → [BCH,cs,N]
        C_tt = ttnn.reshape(C_tt, [B_sz, C_n, cs, H, N])
        C_tt = ttnn.permute(C_tt, [0, 1, 3, 2, 4])
        C_tt = ttnn.reshape(C_tt, [B_sz * C_n * H, cs, N])

        # M = (C @ B) * L, then Y_diag = M @ x
        G_tt = ttnn.matmul(C_tt, B_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        M_tt = ttnn.mul(G_tt, L_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        G_tt.deallocate(True); L_tt.deallocate(True)
        Y_diag_tt = ttnn.matmul(M_tt, x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        M_tt.deallocate(True)                                          # Y_diag: [BCH,cs,D]

        # ── Sub-step 2: Inter-chunk states ───────────────────────────────────
        # decay_states[b,h,c,t] = exp(A_cumsum[b,h,c,-1] - A_cumsum[b,h,c,t])
        A_last_tt = A_cumsum_tt[:, :, :, cs-1:cs]                     # [B,H,C_n,1]
        decay_states_tt = ttnn.exp(
            ttnn.sub(A_last_tt, A_cumsum_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )                                                              # [B,H,C_n,cs]
        A_last_tt.deallocate(True)
        # A_cumsum_tt is NOT deallocated — needed for sub-steps 3 and 4.

        # B_decay = B * decay_states broadcast: [B,C_n,H,N,cs] * [B,C_n,H,1,cs]
        B_tt_5d  = ttnn.reshape(B_tt, [B_sz, C_n, H, N, cs])
        decay_5d = ttnn.reshape(
            ttnn.permute(decay_states_tt, [0, 2, 1, 3]),
            [B_sz, C_n, H, 1, cs]
        )
        decay_states_tt.deallocate(True)
        B_decay_tt = ttnn.mul(B_tt_5d, decay_5d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        B_tt_5d.deallocate(True); decay_5d.deallocate(True); B_tt.deallocate(True)

        # states = x^T @ B_decay: [BCH,D,cs] @ [BCH,cs,N] → [BCH,D,N]
        x_T_tt = ttnn.permute(x_tt, [0, 2, 1])                        # [BCH,D,cs]
        Bd_tt  = ttnn.permute(
            ttnn.reshape(B_decay_tt, [B_sz * C_n * H, N, cs]), [0, 2, 1]
        )                                                              # [BCH,cs,N]
        B_decay_tt.deallocate(True)
        states_tt = ttnn.matmul(x_T_tt, Bd_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_T_tt.deallocate(True); Bd_tt.deallocate(True)
        x_tt.deallocate(True)                                          # x consumed

        states_tt = ttnn.reshape(states_tt, [B_sz, C_n, H, Dh, N])

        # Prepend previous SSM state — use on-device _ssm_state_tt, no PCIe upload.
        # _ssm_state_tt: [1, H, D, N] → reshape to [B, 1, H, D, N] for concat.
        prev_tt = ttnn.reshape(self._ssm_state_tt, [B_sz, 1, H, Dh, N])
        states_tt = ttnn.concat([prev_tt, states_tt], dim=1,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B,C_n+1,H,D,N]
        prev_tt.deallocate(True)

        # ── Sub-step 3: Inter-chunk propagation ─────────────────────────────
        A_cumsum_last_tt = A_cumsum_tt[:, :, :, cs-1]                 # [B,H,C_n]
        A_cumsum_padded_tt = ttnn.concat([self._seg_zero_col_tt, A_cumsum_last_tt], dim=2,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        A_cumsum_last_tt.deallocate(True)
        # A_cumsum_tt still alive — needed for sub-step 4.

        decay_chunk_tt = ttnn.exp(segment_sum_ttnn(A_cumsum_padded_tt, self.device))
        A_cumsum_padded_tt.deallocate(True)

        # [B*H, Cn+1, Cn+1] @ [B*H, Cn+1, D*N] → [B*H, Cn+1, D*N]
        dc_tt = ttnn.reshape(decay_chunk_tt, [B_sz * H, C_n + 1, C_n + 1])
        decay_chunk_tt.deallocate(True)
        st_tt = ttnn.reshape(
            ttnn.permute(states_tt, [0, 2, 1, 3, 4]),
            [B_sz * H, C_n + 1, Dh * N]
        )
        states_tt.deallocate(True)
        new_states_tt = ttnn.matmul(dc_tt, st_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dc_tt.deallocate(True); st_tt.deallocate(True)

        new_states_tt = ttnn.reshape(new_states_tt, [B_sz, H, C_n + 1, Dh, N])
        states_tt    = new_states_tt[:, :, :C_n, :, :]                # [B,H,C_n,D,N]
        ssm_state_tt = new_states_tt[:, :, C_n:, :, :]               # [B,H,1,D,N]
        new_states_tt.deallocate(True)
        ssm_state_tt = ttnn.reshape(ssm_state_tt, [B_sz, H, Dh, N])

        if self.is_mesh and self.num_devices > 1:
            ssm_state = ttnn.to_torch(
                ssm_state_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)
            )[0:1].reshape(B_sz, H, Dh, N).to(torch.bfloat16)
        else:
            ssm_state = to_torch_tensor(ssm_state_tt,
                                        target_shape=(B_sz, H, Dh, N)).to(torch.bfloat16)
        ssm_state_tt.deallocate(True)

        # states: [B,H,C_n,D,N] → [B,C_n,H,D,N]
        states_tt = ttnn.permute(states_tt, [0, 2, 1, 3, 4])

        # ── Sub-step 4: Off-diagonal Y_off ───────────────────────────────────
        # state_decay_out = exp(A_cumsum): A_cumsum_tt is still alive here.
        state_decay_tt = ttnn.exp(A_cumsum_tt)                        # [B,H,C_n,cs]
        A_cumsum_tt.deallocate(True)                                  # now safe to free

        # [B,H,C_n,cs] → [B,C_n,H,cs] → [BCH,cs,1]
        state_decay_tt = ttnn.reshape(
            ttnn.permute(state_decay_tt, [0, 2, 1, 3]),
            [B_sz * C_n * H, cs, 1]
        )

        # states: [B,C_n,H,D,N] → [BCH,N,D]
        st_bch = ttnn.permute(
            ttnn.reshape(states_tt, [B_sz * C_n * H, Dh, N]),
            [0, 2, 1]
        )                                                              # [BCH,N,D]
        states_tt.deallocate(True)

        # Y_off = (C @ states) * state_decay: [BCH,cs,N] @ [BCH,N,D] → [BCH,cs,D]
        Y_off_tt = ttnn.matmul(C_tt, st_bch, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        C_tt.deallocate(True); st_bch.deallocate(True)
        Y_off_tt = ttnn.mul(Y_off_tt, state_decay_tt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)    # [BCH,cs,D]
        state_decay_tt.deallocate(True)

        # ── Combine Y_diag + Y_off, add D residual, reshape to [B,seq,H*D] ──
        Y_tt = ttnn.add(Y_diag_tt, Y_off_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        Y_diag_tt.deallocate(True); Y_off_tt.deallocate(True)

        # [BCH,cs,D] → [B,C_n,H,cs,D] → [B,C_n,cs,H,D] → [B,padded,H,D]
        Y_tt = ttnn.reshape(Y_tt, [B_sz, C_n, H, cs, Dh])
        Y_tt = ttnn.permute(Y_tt, [0, 1, 3, 2, 4])
        Y_tt = ttnn.reshape(Y_tt, [B_sz, padded_len, H, Dh])

        # Add D residual: [B,padded,H,D]
        D_residual_tt = ttnn.reshape(D_residual_tt, [B_sz, padded_len, H, Dh])
        Y_tt = ttnn.add(Y_tt, D_residual_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        D_residual_tt.deallocate(True)

        # Trim padding and flatten heads: [B,seq_len,H,D] → [B,seq_len,H*D]
        if pad_size > 0:
            Y_tt = Y_tt[:, :seq_len, :, :]
        Y_tt = ttnn.reshape(Y_tt, [B_sz, seq_len, H * Dh])

        return Y_tt, ssm_state

    def forward(
        self, hidden_states, cache_params=None, cache_position=None, attention_mask=None
    ):
        """
        Forward pass - dispatches to prefill or decode based on sequence length.
        Accepts either a torch.Tensor [B, S, H] or a TTNN tensor [1, 1, S, H].
        """
        if isinstance(hidden_states, ttnn.Tensor):
            # Use cache_position length as real seq_len — TTNN shape[2] may be tile-padded.
            seq_len = len(cache_position) if cache_position is not None else hidden_states.shape[2]
            if seq_len == 1:
                return self.forward_decode(hidden_states, cache_params)
            # Prefill: pass TTNN tensor directly — no PCIe download needed.
            # forward_prefill_chunk_scan accepts both torch and ttnn tensors.
        else:
            seq_len = hidden_states.shape[1]

        if seq_len == 1:
            return self.forward_decode(hidden_states, cache_params)
        else:
            output, ssm_state = self.forward_prefill_chunk_scan(
                hidden_states, cache_params, real_seq_len=seq_len
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
            # Update prefill conv pad so the next chunk sees the correct causal context.
            # conv_state[:, :, :K-1] holds the last K-1 input tokens; use them as left-pad.
            pad = conv_state[:, :, :K - 1].to(torch.bfloat16).reshape(1, 1, C, K - 1)
            new_pad_tt = ttnn.from_torch(
                pad,
                device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            self._prefill_conv_pad_tt.deallocate(True)
            self._prefill_conv_pad_tt = new_pad_tt

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
        # Reset prefill conv pad to zeros for the next sequence.
        new_pad = ttnn.from_torch(
            torch.zeros(1, 1, conv_dim, kernel_size - 1, dtype=torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        self._prefill_conv_pad_tt.deallocate(True)
        self._prefill_conv_pad_tt = new_pad

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

        # 1. IN_PROJ — column-parallel sharded; gather on fabric (no PCIe).
        # hidden_states_tt: [1, 1, 1, hidden_size]
        projected_tt = ttnn.linear(
            hidden_states_tt, self.in_proj_weight_decode_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.is_mesh and self.num_devices > 1:
            projected_tt = ttnn.all_gather(
                projected_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        gate_tt = projected_tt[:, :, :, :inter]                      # [1,1,1,inter]
        xBC_tt  = projected_tt[:, :, :, inter:inter + conv_dim]       # [1,1,1,conv_dim]
        dt_tt   = projected_tt[:, :, :, inter + conv_dim:]            # [1,1,1,H]
        projected_tt.deallocate(True)

        # 2. CONV1D on device
        xBC_tt = ttnn.reshape(xBC_tt, [1, 1, conv_dim, 1])
        conv_out_tt = self._conv1d_decode_tt(xBC_tt)                 # [1,1,conv_dim,1]
        xBC_tt.deallocate(True)

        # 3. Split conv output on-device — no PCIe download.
        # conv_out_tt: [1, 1, conv_dim, 1] — slice along dim 2.
        x_tt  = conv_out_tt[:, :, :inter, :]                           # [1,1,inter,1]
        B_raw_tt = conv_out_tt[:, :, inter:inter + n_g * N, :]         # [1,1,n_g*N,1]
        C_raw_tt = conv_out_tt[:, :, inter + n_g * N:, :]              # [1,1,n_g*N,1]
        conv_out_tt.deallocate(True)

        x_tt = ttnn.reshape(x_tt, [batch_size, H, D])                  # [B,H,D]

        # B/C: reshape to [B,n_g,N] then repeat heads per group → [B,H,N]
        group_repeat = H // n_g
        B_tt = ttnn.reshape(B_raw_tt, [batch_size, n_g, N])
        C_tt = ttnn.reshape(C_raw_tt, [batch_size, n_g, N])
        B_raw_tt.deallocate(True); C_raw_tt.deallocate(True)
        if group_repeat > 1:
            # repeat_interleave via unsqueeze+repeat+reshape — mesh-safe
            B_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(B_tt, 2), ttnn.Shape([1, 1, group_repeat, 1])), [batch_size, H, N])
            C_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(C_tt, 2), ttnn.Shape([1, 1, group_repeat, 1])), [batch_size, H, N])

        # 4. SSM step — fully on device
        # dt_tt: [1,1,1,H] → [B,H,D]
        # dt_tt: [1,1,1,H] → [B,H,1]; add bias [1,H,D] → broadcasts to [B,H,D]
        dt_tt = ttnn.reshape(dt_tt, [batch_size, H, 1])
        dt_tt = ttnn.add(dt_tt, self._ssm_dt_bias_tt)               # [B,H,D]
        dt_tt = ttnn.softplus(dt_tt, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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

        # new_state = dBx + dA * state  →  addcmul(dBx, dA, state) = dBx + dA*state
        new_state = ttnn.addcmul(dBx_tt, dA_tt, self._ssm_state_tt,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dA_tt.deallocate(True); dBx_tt.deallocate(True)
        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = new_state

        C_tt = ttnn.unsqueeze(C_tt, -2)                             # [B,H,1,N]
        y_unred = ttnn.mul(self._ssm_state_tt, C_tt)                # [B,H,D,N]
        C_tt.deallocate(True)
        y_tt = ttnn.sum(y_unred, dim=-1,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)       # [B,H,D]
        y_unred.deallocate(True)
        # y += D * x  →  addcmul(y, D, x)
        y_tt = ttnn.addcmul(y_tt, self._ssm_D_tt, x_tt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
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

        # 6. OUT_PROJ — column-parallel sharded; gather on fabric (no PCIe).
        output_tt = ttnn.linear(
            scan_tt, self.out_proj_weight_decode_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scan_tt.deallocate(True)
        if self.is_mesh and self.num_devices > 1:
            output_tt = ttnn.all_gather(
                output_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        dt_tt = ttnn.softplus(dt_tt, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
