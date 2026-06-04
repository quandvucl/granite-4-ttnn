"""Tensor-parallel Mamba2 chunk-scan implementation for TTNN."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tt-metal"))

import torch
import ttnn
from models.common.tensor_utils import pad_dim_to_size
from models.common.utility_functions import roundup
from mamba.utils import segment_sum_ttnn

from utils import to_torch_tensor, to_tt_tensor
from utils.device import (
    _is_mesh_device,
    _make_mesh_mapper,
    softplus_and_clamp_torch_via_tt,
)
from mamba.config import Mamba2Config
from mamba.device_manager import TTNNDeviceManager
from mamba.ssm_utils import extract_ssm_parameters

class TensorParallelMamba:

    def __init__(
        self,
        hf_mamba,
        device,
        dtype=ttnn.bfloat16,
        tensor_parallel=True,
        chunk_size_override=None,
        weight_dtype=None,
    ):
        self.hf_mamba = hf_mamba
        self.device = device
        self.dtype = dtype
        self.weight_dtype = weight_dtype if weight_dtype is not None else dtype
        self.tensor_parallel = tensor_parallel
        self.layer_idx = hf_mamba.layer_idx

        self.is_mesh = _is_mesh_device(device)
        self.num_devices = device.get_num_devices() if self.is_mesh else 1

        self.config = Mamba2Config.from_hf_mamba(hf_mamba)
        self.device_mgr = TTNNDeviceManager(device, dtype)

        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.ssm_state_size = self.config.ssm_state_size
        self.hidden_size = self.config.hidden_size

        self.chunk_size = chunk_size_override if chunk_size_override is not None else self.config.chunk_size

        self.num_groups = (
            hf_mamba.num_heads // hf_mamba.n_groups
            if hasattr(hf_mamba, "n_groups")
            else hf_mamba.num_heads
        )
        self._group_repeat_factor = self.num_heads // self.num_groups

        self._prefill_A = -torch.exp(hf_mamba.A_log.float())
        self._prefill_D = hf_mamba.D.float()

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

        self.mesh_mapper = _make_mesh_mapper(self.device)

        self._load_weights()
        self._preload_decode_constants()

        self._ssm_state_tt = ttnn.from_torch(
            torch.zeros(1, self.num_heads, self.head_dim, self.ssm_state_size,
                        dtype=torch.bfloat16),
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        kernel_size = self.hf_mamba.conv1d.weight.shape[2]
        conv_dim = self.hf_mamba.conv_dim
        # K separate [1,1,C,1] column tensors; _conv_pos is the next write slot.
        self._conv_cache_cols = [
            ttnn.from_torch(
                torch.zeros(1, 1, conv_dim, 1, dtype=torch.bfloat16),
                device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            for _ in range(kernel_size)
        ]
        self._conv_pos = 0

    def _load_weights(self):
        replicate_mapper = self.mesh_mapper

        if self.is_mesh and self.num_devices > 1 and self.tensor_parallel:
            # Use actual mesh shape so row ≥ 1 also gets valid column-shards.
            mesh_shape = self.device.shape
            col_mapper = ttnn.ShardTensor2dMesh(
                self.device, dims=(None, -1), mesh_shape=mesh_shape
            )
            in_t  = self.hf_mamba.in_proj.weight.T.contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            out_t = self.hf_mamba.out_proj.weight.T.contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            self.in_proj_weight_tt = ttnn.from_torch(
                in_t, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=col_mapper,
            )
            self.out_proj_weight_tt = ttnn.from_torch(
                out_t, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=col_mapper,
            )
        else:
            self.in_proj_weight_tt = to_tt_tensor(
                self.hf_mamba.in_proj.weight.T.contiguous(),
                self.device, self.weight_dtype, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper,
            )
            self.out_proj_weight_tt = to_tt_tensor(
                self.hf_mamba.out_proj.weight.T.contiguous(),
                self.device, self.weight_dtype, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper,
            )

        self._use_col_parallel = self.is_mesh and self.num_devices > 1 and self.tensor_parallel

        self.in_proj_weight_decode_tt = self.in_proj_weight_tt
        self.out_proj_weight_decode_tt = self.out_proj_weight_tt

        if self.hf_mamba.use_conv_bias:
            conv_bias_4d = self.hf_mamba.conv1d.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.conv_bias_tt = to_tt_tensor(
                conv_bias_4d, self.device, self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate_mapper,
            )
        else:
            self.conv_bias_tt = None

    def _preload_decode_constants(self):
        mapper = self.mesh_mapper

        self._ssm_A_tt = to_tt_tensor(
            self._ssm_A, self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._ssm_dt_bias_tt = to_tt_tensor(
            self._ssm_dt_bias.unsqueeze(0),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._ssm_D_tt = to_tt_tensor(
            self._ssm_D.unsqueeze(0),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        norm_w = self.hf_mamba.norm.weight.to(torch.bfloat16)
        self._norm_eps = self.hf_mamba.norm.variance_epsilon
        self._norm_weight_tt = to_tt_tensor(
            norm_w.unsqueeze(0).unsqueeze(0),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )

        H = self.num_heads
        K = self.hf_mamba.conv1d.weight.shape[2]
        C = self.hf_mamba.conv_dim
        conv_w = self.hf_mamba.conv1d.weight.squeeze(1).to(torch.bfloat16)  # [C, K]
        # K separate [1,1,C,1] weight column tensors — newest weight first (index 0).
        # conv1d weight col [:, k] corresponds to lag k (0=newest, K-1=oldest).
        self._conv_w_cols = [
            to_tt_tensor(
                conv_w[:, k].reshape(1, 1, C, 1),
                self.device, ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
            )
            for k in range(K)
        ]
        if self.hf_mamba.use_conv_bias:
            self._conv_bias_decode_tt = to_tt_tensor(
                self.hf_mamba.conv1d.bias.to(torch.bfloat16).reshape(1, 1, C, 1),
                self.device, ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
            )
        else:
            self._conv_bias_decode_tt = None

        self._prefill_A_tt = to_tt_tensor(
            self._prefill_A.to(torch.bfloat16).reshape(1, 1, H),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._prefill_D_tt = to_tt_tensor(
            self._prefill_D.to(torch.bfloat16).reshape(1, 1, H, 1),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._prefill_dt_bias_tt = to_tt_tensor(
            self.hf_mamba.dt_bias.to(torch.bfloat16).reshape(1, 1, H),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._prefill_conv_pad_tt = to_tt_tensor(
            torch.zeros(1, 1, C, K - 1, dtype=torch.bfloat16),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )
        self._seg_zero_col_tt = to_tt_tensor(
            torch.zeros(1, H, 1, dtype=torch.bfloat16),
            self.device, ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
        )

        cs = self.chunk_size
        N = self.ssm_state_size
        D = self.head_dim
        _max_pad = cs
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
        replicate_mapper = self.mesh_mapper
        inter   = self.hf_mamba.intermediate_size
        conv_d  = self.hf_mamba.conv_dim
        n_heads = self.hf_mamba.num_heads

        _owns_hidden_tt = True
        if isinstance(hidden_states, ttnn.Tensor):
            batch_size = hidden_states.shape[0]
            seq_len = real_seq_len if real_seq_len is not None else hidden_states.shape[2]
            hidden_tt = hidden_states
            _owns_hidden_tt = False
        else:
            batch_size, seq_len, _ = hidden_states.shape
            hidden_tt = to_tt_tensor(
                hidden_states, self.device, self.dtype, mesh_mapper=replicate_mapper
            )
            hidden_tt = ttnn.to_layout(hidden_tt, ttnn.TILE_LAYOUT)

        projected_tt = ttnn.matmul(hidden_tt, self.in_proj_weight_tt)
        if _owns_hidden_tt:
            hidden_tt.deallocate(True)
        if self._use_col_parallel:
            projected_tt = ttnn.all_gather(
                projected_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        padded_s = projected_tt.shape[2]
        if padded_s != seq_len:
            projected_tt = projected_tt[:, :, :seq_len, :]
        gate_tt = projected_tt[:, :, :, :inter]
        xBC_tt  = projected_tt[:, :, :, inter:inter + conv_d]
        dt_tt   = projected_tt[:, :, :, inter + conv_d:]
        projected_tt.deallocate(True)

        gate_tt = ttnn.reshape(gate_tt, [batch_size, seq_len, inter])
        xBC_tt  = ttnn.reshape(xBC_tt,  [batch_size, seq_len, conv_d])
        dt_tt   = ttnn.reshape(dt_tt,   [batch_size, seq_len, n_heads])

        kernel_size = self.hf_mamba.conv1d.weight.shape[2]

        if cache_params is not None and hasattr(cache_params, 'conv_states'):
            tail_len = min(kernel_size - 1, seq_len)
            tail_tt = xBC_tt[:, seq_len - tail_len:, :]
            if self.is_mesh and self.num_devices > 1:
                tail_cpu = ttnn.to_torch(
                    tail_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)
                )[0:1].reshape(batch_size, tail_len, conv_d)
            else:
                tail_cpu = to_torch_tensor(tail_tt, target_shape=(batch_size, tail_len, conv_d))
            tail_tt.deallocate(True)
            tail_t = tail_cpu.transpose(1, 2)
            conv_state = torch.nn.functional.pad(tail_t, (kernel_size - 1 - tail_len, 0))
            cache_params.conv_states[self.hf_mamba.layer_idx].copy_(
                torch.cat([conv_state,
                           torch.zeros(batch_size, conv_d, 1, dtype=conv_state.dtype)],
                          dim=-1)[:, :, -(kernel_size):]
            )

        xBC_tt = ttnn.reshape(xBC_tt, [batch_size, 1, seq_len, conv_d])
        xBC_tt = ttnn.transpose(xBC_tt, 2, 3)                          # [B, 1, conv_d, S]

        padded_tt = ttnn.concat([self._prefill_conv_pad_tt, xBC_tt], dim=3,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xBC_tt.deallocate(True)

        acc_tt = None
        for k in range(kernel_size):
            s = padded_tt[:, :, :, k:k + seq_len]
            term = ttnn.mul(s, self._conv_w_cols[k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s.deallocate(True)
            if acc_tt is None:
                acc_tt = term
            else:
                acc_tt = ttnn.add(acc_tt, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                term.deallocate(True)
        padded_tt.deallocate(True)

        if self._conv_bias_decode_tt is not None:
            acc_tt = ttnn.add(acc_tt, self._conv_bias_decode_tt,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)

        xBC_tt = ttnn.silu(acc_tt)
        acc_tt.deallocate(True)
        xBC_tt = ttnn.transpose(xBC_tt, 2, 3)                          # [B, 1, S, conv_d]
        xBC_tt = ttnn.reshape(xBC_tt, [batch_size, seq_len, conv_d])

        n_g = self.hf_mamba.n_groups
        N   = self.ssm_state_size
        H   = self.num_heads
        group_repeat = H // n_g

        x_tt    = xBC_tt[:, :, :inter]
        B_raw_tt = xBC_tt[:, :, inter:inter + n_g * N]
        C_raw_tt = xBC_tt[:, :, inter + n_g * N:]
        xBC_tt.deallocate(True)

        x_tt = ttnn.reshape(x_tt, [batch_size, seq_len, H, self.head_dim])
        B_tt = ttnn.reshape(B_raw_tt, [batch_size, seq_len, n_g, N])
        C_tt = ttnn.reshape(C_raw_tt, [batch_size, seq_len, n_g, N])
        B_raw_tt.deallocate(True); C_raw_tt.deallocate(True)
        if group_repeat > 1:
            B_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(B_tt, 3), ttnn.Shape([1, 1, 1, group_repeat, 1])), [batch_size, seq_len, H, N])
            C_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(C_tt, 3), ttnn.Shape([1, 1, 1, group_repeat, 1])), [batch_size, seq_len, H, N])

        y_tt = self._chunk_scan_ssm_ttnn(x_tt, B_tt, C_tt, dt_tt, cache_params)
        x_tt.deallocate(True); B_tt.deallocate(True)
        C_tt.deallocate(True); dt_tt.deallocate(True)

        silu_gate_tt = ttnn.silu(gate_tt)
        gate_tt.deallocate(True)
        gated_tt = ttnn.mul(y_tt, silu_gate_tt)
        y_tt.deallocate(True); silu_gate_tt.deallocate(True)
        gated_tt = ttnn.rms_norm(gated_tt, epsilon=self._norm_eps,
                                 weight=self._norm_weight_tt)

        output_tt = ttnn.matmul(gated_tt, self.out_proj_weight_tt)
        gated_tt.deallocate(True)
        if self._use_col_parallel:
            output_tt = ttnn.all_gather(
                output_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        output_tt = ttnn.reshape(output_tt, [1, 1, seq_len, self.hidden_size])
        return output_tt

    def _chunk_scan_ssm_ttnn(self, x_tt, B_tt, C_tt, dt_tt, cache_params):
        B_sz    = x_tt.shape[0]
        seq_len = x_tt.shape[1]
        H       = self.num_heads
        Dh      = self.head_dim
        N       = self.ssm_state_size

        tile = 32
        seq_aligned = max(tile, roundup(seq_len, tile))
        cs = seq_aligned if seq_aligned < self.chunk_size else self.chunk_size
        padded_len = roundup(seq_len, cs)
        pad_size   = padded_len - seq_len
        C_n        = padded_len // cs

        dt_tt = ttnn.add(dt_tt, self._prefill_dt_bias_tt,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_tt = ttnn.softplus(dt_tt, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lo, hi = self.hf_mamba.time_step_limit
        dt_tt = ttnn.clip(dt_tt, lo, hi)

        D_residual_tt = ttnn.mul(self._prefill_D_tt, x_tt,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)

        A_dt_tt = ttnn.mul(self._prefill_A_tt, dt_tt,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

        dt_exp = ttnn.unsqueeze(dt_tt, -1)
        x_tt   = ttnn.mul(x_tt, dt_exp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_tt.deallocate(True); dt_exp.deallocate(True)

        def _pad_seq(t_tt, extra_cols, preloaded_pad):
            if extra_cols == 0:
                return t_tt
            pad = preloaded_pad[:, :extra_cols, ...]
            return ttnn.concat([t_tt, pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        A_dt_tt       = _pad_seq(A_dt_tt,       pad_size, self._pad_H_tt)
        x_tt          = _pad_seq(x_tt,          pad_size, self._pad_HD_tt)
        D_residual_tt = _pad_seq(D_residual_tt, pad_size, self._pad_HD_tt)
        B_tt          = _pad_seq(B_tt,          pad_size, self._pad_HN_tt)
        C_tt          = _pad_seq(C_tt,          pad_size, self._pad_HN_tt)

        A_dt_tt = ttnn.reshape(A_dt_tt, [B_sz, C_n, cs, H])
        A_dt_tt = ttnn.permute(A_dt_tt, [0, 3, 1, 2])                 # [B,H,C_n,cs]

        A_cumsum_tt = ttnn.cumsum(A_dt_tt, dim=-1)                    # [B,H,C_n,cs]

        L_tt = ttnn.exp(segment_sum_ttnn(A_dt_tt, self.device))       # [B,H,C_n,cs,cs]
        A_dt_tt.deallocate(True)

        L_tt = ttnn.permute(L_tt, [0, 2, 1, 3, 4])
        L_tt = ttnn.reshape(L_tt, [B_sz * C_n * H, cs, cs])

        x_tt = ttnn.reshape(x_tt, [B_sz, C_n, cs, H, Dh])
        x_tt = ttnn.permute(x_tt, [0, 1, 3, 2, 4])
        x_tt = ttnn.reshape(x_tt, [B_sz * C_n * H, cs, Dh])

        B_tt = ttnn.reshape(B_tt, [B_sz, C_n, cs, H, N])
        B_tt = ttnn.permute(B_tt, [0, 1, 3, 4, 2])
        B_tt = ttnn.reshape(B_tt, [B_sz * C_n * H, N, cs])

        C_tt = ttnn.reshape(C_tt, [B_sz, C_n, cs, H, N])
        C_tt = ttnn.permute(C_tt, [0, 1, 3, 2, 4])
        C_tt = ttnn.reshape(C_tt, [B_sz * C_n * H, cs, N])

        G_tt = ttnn.matmul(C_tt, B_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        M_tt = ttnn.mul(G_tt, L_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        G_tt.deallocate(True); L_tt.deallocate(True)
        Y_diag_tt = ttnn.matmul(M_tt, x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        M_tt.deallocate(True)

        A_last_tt = A_cumsum_tt[:, :, :, cs-1:cs]
        decay_states_tt = ttnn.exp(
            ttnn.sub(A_last_tt, A_cumsum_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        A_last_tt.deallocate(True)

        B_tt_5d  = ttnn.reshape(B_tt, [B_sz, C_n, H, N, cs])
        decay_5d = ttnn.reshape(
            ttnn.permute(decay_states_tt, [0, 2, 1, 3]),
            [B_sz, C_n, H, 1, cs]
        )
        decay_states_tt.deallocate(True)
        B_decay_tt = ttnn.mul(B_tt_5d, decay_5d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        B_tt_5d.deallocate(True); decay_5d.deallocate(True); B_tt.deallocate(True)

        x_T_tt = ttnn.permute(x_tt, [0, 2, 1])
        Bd_tt  = ttnn.permute(
            ttnn.reshape(B_decay_tt, [B_sz * C_n * H, N, cs]), [0, 2, 1]
        )
        B_decay_tt.deallocate(True)
        states_tt = ttnn.matmul(x_T_tt, Bd_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_T_tt.deallocate(True); Bd_tt.deallocate(True)
        x_tt.deallocate(True)

        states_tt = ttnn.reshape(states_tt, [B_sz, C_n, H, Dh, N])

        prev_tt = ttnn.reshape(self._ssm_state_tt, [B_sz, 1, H, Dh, N])
        states_tt = ttnn.concat([prev_tt, states_tt], dim=1,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        prev_tt.deallocate(True)

        A_cumsum_last_tt = A_cumsum_tt[:, :, :, cs-1]
        A_cumsum_padded_tt = ttnn.concat([self._seg_zero_col_tt, A_cumsum_last_tt], dim=2,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        A_cumsum_last_tt.deallocate(True)

        decay_chunk_tt = ttnn.exp(segment_sum_ttnn(A_cumsum_padded_tt, self.device))
        A_cumsum_padded_tt.deallocate(True)

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
        states_tt    = new_states_tt[:, :, :C_n, :, :]
        ssm_state_tt = new_states_tt[:, :, C_n:, :, :]
        new_states_tt.deallocate(True)
        ssm_state_tt = ttnn.reshape(ssm_state_tt, [B_sz, H, Dh, N])

        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = ssm_state_tt

        states_tt = ttnn.permute(states_tt, [0, 2, 1, 3, 4])

        state_decay_tt = ttnn.exp(A_cumsum_tt)
        A_cumsum_tt.deallocate(True)

        state_decay_tt = ttnn.reshape(
            ttnn.permute(state_decay_tt, [0, 2, 1, 3]),
            [B_sz * C_n * H, cs, 1]
        )

        st_bch = ttnn.permute(
            ttnn.reshape(states_tt, [B_sz * C_n * H, Dh, N]),
            [0, 2, 1]
        )
        states_tt.deallocate(True)

        Y_off_tt = ttnn.matmul(C_tt, st_bch, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        C_tt.deallocate(True); st_bch.deallocate(True)
        Y_off_tt = ttnn.mul(Y_off_tt, state_decay_tt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state_decay_tt.deallocate(True)

        Y_tt = ttnn.add(Y_diag_tt, Y_off_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        Y_diag_tt.deallocate(True); Y_off_tt.deallocate(True)

        Y_tt = ttnn.reshape(Y_tt, [B_sz, C_n, H, cs, Dh])
        Y_tt = ttnn.permute(Y_tt, [0, 1, 3, 2, 4])
        Y_tt = ttnn.reshape(Y_tt, [B_sz, padded_len, H, Dh])

        D_residual_tt = ttnn.reshape(D_residual_tt, [B_sz, padded_len, H, Dh])
        Y_tt = ttnn.add(Y_tt, D_residual_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        D_residual_tt.deallocate(True)

        if pad_size > 0:
            Y_tt = Y_tt[:, :seq_len, :, :]
        Y_tt = ttnn.reshape(Y_tt, [B_sz, seq_len, H * Dh])

        return Y_tt

    def forward(
        self, hidden_states, cache_params=None, cache_position=None, attention_mask=None
    ):
        if isinstance(hidden_states, ttnn.Tensor):
            seq_len = len(cache_position) if cache_position is not None else hidden_states.shape[2]
            if seq_len == 1:
                return self.forward_decode(hidden_states, cache_params)
        else:
            seq_len = hidden_states.shape[1]

        if seq_len == 1:
            return self.forward_decode(hidden_states, cache_params)
        else:
            output = self.forward_prefill_chunk_scan(
                hidden_states, cache_params, real_seq_len=seq_len
            )
            self._seed_state(cache_params)
            return output

    def _seed_state(self, cache_params=None):
        """Seed conv cache after prefill; write ssm state to cache_params if needed."""
        if cache_params is not None and hasattr(cache_params, "ssm_states"):
            B_sz = 1
            H, Dh, N = self.num_heads, self.head_dim, self.ssm_state_size
            if self.is_mesh and self.num_devices > 1:
                ssm_state = ttnn.to_torch(
                    self._ssm_state_tt,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)
                )[0:1].reshape(B_sz, H, Dh, N).to(torch.bfloat16)
            else:
                ssm_state = to_torch_tensor(
                    self._ssm_state_tt, target_shape=(B_sz, H, Dh, N)
                ).to(torch.bfloat16)
            cache_params.ssm_states[self.layer_idx] = ssm_state

        if cache_params is not None and hasattr(cache_params, "conv_states"):
            conv_state = cache_params.conv_states[self.layer_idx]
            K = conv_state.shape[-1]
            C = self.hf_mamba.conv_dim
            for col in self._conv_cache_cols:
                col.deallocate(True)
            # conv_state is [1, C, K] — col k is the k-th time step (0=oldest)
            self._conv_cache_cols = [
                ttnn.from_torch(
                    conv_state[0, :, k].reshape(1, 1, C, 1).to(torch.bfloat16),
                    device=self.device, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                for k in range(K)
            ]
            # Next write replaces oldest (col 0), so pos = 0 after loading.
            self._conv_pos = 0
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
        """Zero SSM state and conv cache between sequences."""
        self._ssm_state_tt = ttnn.zeros_like(self._ssm_state_tt)
        self._conv_cache_cols = [ttnn.zeros_like(col) for col in self._conv_cache_cols]
        self._conv_pos = 0
        self._prefill_conv_pad_tt = ttnn.zeros_like(self._prefill_conv_pad_tt)

    def _conv1d_decode_tt(self, xBC_tt):
        """xBC_tt: [1,1,C,1]. Updates conv cache cols in-place. Returns [1,1,C,1]."""
        K = len(self._conv_cache_cols)
        # Write new input into the oldest slot and advance the write pointer.
        self._conv_cache_cols[self._conv_pos].deallocate(True)
        self._conv_cache_cols[self._conv_pos] = xBC_tt
        self._conv_pos = (self._conv_pos + 1) % K

        # Weighted sum: w[0]*newest + w[1]*2nd-newest + ... + w[K-1]*oldest
        # Newest is at (self._conv_pos - 1) % K, oldest at self._conv_pos.
        out = None
        for k in range(K):
            col_idx = (self._conv_pos - 1 - k) % K
            term = ttnn.mul(self._conv_w_cols[k], self._conv_cache_cols[col_idx],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if out is None:
                out = term
            else:
                new_out = ttnn.add(out, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out.deallocate(True)
                term.deallocate(True)
                out = new_out

        if self._conv_bias_decode_tt is not None:
            out = ttnn.add(out, self._conv_bias_decode_tt,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.silu(out)
        return out

    def forward_decode(self, hidden_states_tt, cache_params):
        """
        Decode mode — fully on-device, no PCIe transfers.

        hidden_states_tt: TTNN tensor [1, 1, 1, H] (replicated).
        Returns: TTNN tensor [1, 1, 1, H] (replicated).
        """
        n_g = self.hf_mamba.n_groups
        H = self.num_heads
        D = self.head_dim
        N = self.ssm_state_size
        inter = self.hf_mamba.intermediate_size
        conv_dim = self.hf_mamba.conv_dim
        batch_size = 1

        projected_tt = ttnn.linear(
            hidden_states_tt, self.in_proj_weight_decode_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._use_col_parallel:
            projected_tt = ttnn.all_gather(
                projected_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        gate_tt = projected_tt[:, :, :, :inter]
        xBC_tt  = projected_tt[:, :, :, inter:inter + conv_dim]
        dt_tt   = projected_tt[:, :, :, inter + conv_dim:]
        projected_tt.deallocate(True)

        xBC_tt = ttnn.reshape(xBC_tt, [1, 1, conv_dim, 1])
        # _conv1d_decode_tt takes ownership of xBC_tt (stores it in cache cols).
        conv_out_tt = self._conv1d_decode_tt(xBC_tt)

        x_tt     = conv_out_tt[:, :, :inter, :]
        B_raw_tt = conv_out_tt[:, :, inter:inter + n_g * N, :]
        C_raw_tt = conv_out_tt[:, :, inter + n_g * N:, :]
        conv_out_tt.deallocate(True)

        x_tt = ttnn.reshape(x_tt, [batch_size, H, D])

        group_repeat = H // n_g
        B_tt = ttnn.reshape(B_raw_tt, [batch_size, n_g, N])
        C_tt = ttnn.reshape(C_raw_tt, [batch_size, n_g, N])
        B_raw_tt.deallocate(True); C_raw_tt.deallocate(True)
        if group_repeat > 1:
            B_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(B_tt, 2), ttnn.Shape([1, 1, group_repeat, 1])), [batch_size, H, N])
            C_tt = ttnn.reshape(ttnn.repeat(ttnn.unsqueeze(C_tt, 2), ttnn.Shape([1, 1, group_repeat, 1])), [batch_size, H, N])

        dt_tt = ttnn.reshape(dt_tt, [batch_size, H, 1])
        dt_tt = ttnn.add(dt_tt, self._ssm_dt_bias_tt)
        dt_tt = ttnn.softplus(dt_tt, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_tt = ttnn.clip(dt_tt, self.hf_mamba.time_step_limit[0],
                          self.hf_mamba.time_step_limit[1])

        dt_exp = ttnn.unsqueeze(dt_tt, -1)
        dA_tt  = ttnn.exp(ttnn.mul(dt_exp, self._ssm_A_tt))
        dt_exp.deallocate(True)

        dtx_tt = ttnn.mul(dt_tt, x_tt)
        dt_tt.deallocate(True)
        dtx_tt = ttnn.unsqueeze(dtx_tt, -1)
        B_tt   = ttnn.unsqueeze(B_tt, -2)
        dBx_tt = ttnn.mul(dtx_tt, B_tt)
        dtx_tt.deallocate(True); B_tt.deallocate(True)

        C_tt = ttnn.unsqueeze(C_tt, -2)   # [B, H, N] -> [B, H, 1, N]

        new_state = ttnn.addcmul(dBx_tt, dA_tt, self._ssm_state_tt,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dA_tt.deallocate(True); dBx_tt.deallocate(True)
        self._ssm_state_tt.deallocate(True)
        self._ssm_state_tt = new_state

        y_unred = ttnn.mul(new_state, C_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        C_tt.deallocate(True)

        y_tt = ttnn.sum(y_unred, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y_unred.deallocate(True)
        y_tt = ttnn.addcmul(y_tt, self._ssm_D_tt, x_tt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_tt.deallocate(True)
        y_tt = ttnn.reshape(y_tt, [batch_size, 1, H * D])

        gate_tt = ttnn.reshape(gate_tt, [batch_size, 1, inter])
        silu_gate_tt = ttnn.silu(gate_tt)
        gate_tt.deallocate(True)
        gated_tt = ttnn.mul(y_tt, silu_gate_tt)
        y_tt.deallocate(True); silu_gate_tt.deallocate(True)
        scan_tt = ttnn.rms_norm(gated_tt, epsilon=self._norm_eps, weight=self._norm_weight_tt)
        gated_tt.deallocate(True)

        output_tt = ttnn.linear(
            scan_tt, self.out_proj_weight_decode_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scan_tt.deallocate(True)
        if self._use_col_parallel:
            output_tt = ttnn.all_gather(
                output_tt, dim=3, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return output_tt
