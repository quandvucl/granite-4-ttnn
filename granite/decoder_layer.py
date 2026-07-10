"""Hybrid Granite decoder layer — hidden_states stays on TTNN throughout all 40 layers."""

import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.append(str(Path(__file__).parent.parent))
from granite.cache import MambaCacheManager
from mamba import TensorParallelMamba
from models.common.rmsnorm import RMSNorm
from granite.moe_tt import GraniteTTMoE
from models.tt_transformers.tt.common import Mode


def _replicate(tensor: torch.Tensor, device, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None
    return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout, mesh_mapper=mapper,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _tt_to_torch(tt: ttnn.Tensor, shape) -> torch.Tensor:
    device = tt.device()
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        result = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
    else:
        result = tt.cpu().to_torch()
    return result.reshape(shape)


class TTGraniteDecoderLayer:
    """
    Granite hybrid decoder layer.
    hidden_states lives as a TTNN tensor [1, 1, S, H] for the full 40-layer stack.
    """

    def __init__(
        self,
        device,
        layer_idx: int,
        config,
        hf_layer,
        shared_mlp,
        is_attention_layer: bool,
        hf_config=None,
        dtype=ttnn.bfloat16,
        tensor_parallel=False,
        tt_ccl=None,
        use_tt_attention=True,
        use_tt_mamba=True,
        use_tt_moe=True,
        mamba_chunk_size=None,
        moe_weight_dtype=ttnn.bfloat8_b,
        mamba_weight_dtype=None,
        moe_use_all_gather=True,
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.config = config
        self.hf_config = hf_config
        self.hf_layer = hf_layer
        self.shared_mlp = shared_mlp
        self.is_attention_layer = is_attention_layer
        self.dtype = dtype
        self.tensor_parallel = tensor_parallel
        self.tt_ccl = tt_ccl
        self.use_tt_attention = use_tt_attention
        self.use_tt_mamba = use_tt_mamba
        self.use_tt_moe = use_tt_moe
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_weight_dtype = mamba_weight_dtype

        self.residual_multiplier = config.residual_multiplier if hasattr(config, "residual_multiplier") else 1.0
        self.is_mesh = hasattr(device, "get_num_devices") and device.get_num_devices() > 1

        rm = torch.tensor([[[[self.residual_multiplier]]]], dtype=torch.bfloat16)
        self.residual_multiplier_tt = _replicate(rm, device, ttnn.bfloat16)

        input_norm_sd = {"input_layernorm.weight": hf_layer.input_layernorm.weight}
        self.input_layernorm = RMSNorm(
            device=device, dim=config.hidden_size,
            state_dict=input_norm_sd, weight_key="input_layernorm",
            weight_dtype=dtype, eps=config.rms_norm_eps, fp32_dest_acc_en=True,
        )
        post_attn_norm_sd = {"post_attention_layernorm.weight": hf_layer.post_attention_layernorm.weight}
        self.post_attention_layernorm = RMSNorm(
            device=device, dim=config.hidden_size,
            state_dict=post_attn_norm_sd, weight_key="post_attention_layernorm",
            weight_dtype=dtype, eps=config.rms_norm_eps, fp32_dest_acc_en=True,
        )

        if use_tt_moe and hasattr(hf_layer, "block_sparse_moe"):
            self.tt_moe = GraniteTTMoE(hf_layer.block_sparse_moe, device,
                                       weight_dtype=moe_weight_dtype, act_dtype=dtype,
                                       use_all_gather=moe_use_all_gather)
        else:
            self.tt_moe = None

        if is_attention_layer:
            head_dim = config.hidden_size // config.num_attention_heads
            if use_tt_attention:
                from granite.attention_nope import AttentionNoPE
                self.simple_attention = AttentionNoPE(
                    device=device,
                    q_weight=hf_layer.self_attn.q_proj.weight,
                    k_weight=hf_layer.self_attn.k_proj.weight,
                    v_weight=hf_layer.self_attn.v_proj.weight,
                    o_weight=hf_layer.self_attn.o_proj.weight,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=head_dim,
                    hidden_size=config.hidden_size,
                    max_seq_len=config.max_cache_length,
                    dtype=dtype,
                    layer_idx=layer_idx,
                )
            else:
                self.simple_attention = None
            self.mamba = None
        else:
            self.simple_attention = None
            if hasattr(hf_layer, "mamba") and use_tt_mamba:
                self.mamba = TensorParallelMamba(
                    hf_mamba=hf_layer.mamba,
                    device=device,
                    dtype=dtype,
                    # TP all_gather latency (2×36=72 per step) exceeds compute savings
                    # for H=1536 (tiny). Keep tensor_parallel=False until model is larger.
                    tensor_parallel=False,
                    chunk_size_override=mamba_chunk_size,
                    weight_dtype=mamba_weight_dtype,
                )
            else:
                self.mamba = None

        self.last_timing = {
            "attention": 0.0, "mamba": 0.0,
            "mamba_prefill": 0.0, "mamba_decode": 0.0,
            "mlp": 0.0, "total": 0.0,
        }

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cache_manager: MambaCacheManager,
        position_ids: torch.Tensor,
        attention_mask=None,
        position_embeddings=None,
        cache_position=None,
        has_previous_state=None,
    ) -> ttnn.Tensor:
        t0 = time.time()
        attn_t = mamba_t = mamba_pre_t = mamba_dec_t = mlp_t = 0.0

        seq_len = len(cache_position) if cache_position is not None else hidden_states.shape[2]
        mode = Mode.DECODE if seq_len == 1 else Mode.PREFILL
        if not hasattr(cache_manager, "hybrid_cache"):
            raise RuntimeError("hybrid_cache must be initialized before decoder layer forward")

        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states, mode=mode)

        t1 = time.time()
        if self.is_attention_layer:
            mixer_out = self._attention_forward(
                normed, cache_manager, position_ids,
                attention_mask, position_embeddings, cache_position, seq_len,
            )
            attn_t = time.time() - t1
        else:
            mixer_out = self._mamba_forward(
                normed, cache_manager, position_ids, cache_position, seq_len,
            )
            mamba_t = time.time() - t1
            if seq_len == 1:
                mamba_dec_t = mamba_t
            else:
                mamba_pre_t = mamba_t

        normed.deallocate(True)

        hidden_states = ttnn.mac(mixer_out, self.residual_multiplier_tt, residual,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mixer_out.deallocate(True)
        residual.deallocate(True)

        residual = hidden_states
        normed2 = self.post_attention_layernorm.forward(hidden_states, mode=mode)

        t2 = time.time()
        mlp_out = self._mlp_forward(normed2, seq_len, mode)
        normed2.deallocate(True)
        mlp_t = time.time() - t2

        hidden_states = ttnn.mac(mlp_out, self.residual_multiplier_tt, residual,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp_out.deallocate(True)
        residual.deallocate(True)

        self.last_timing = {
            "attention": attn_t, "mamba": mamba_t,
            "mamba_prefill": mamba_pre_t, "mamba_decode": mamba_dec_t,
            "mlp": mlp_t, "total": time.time() - t0,
        }
        return hidden_states

    def _attention_forward(self, normed, cache_manager, position_ids,
                           attention_mask, position_embeddings,
                           cache_position, seq_len) -> ttnn.Tensor:
        if self.simple_attention is not None:
            return self.simple_attention.forward(
                normed, position_ids=position_ids, cache_manager=cache_manager,
            )
        hs_torch = _tt_to_torch(normed, (1, seq_len, self.config.hidden_size))
        attn_out = self.hf_layer.self_attn(
            hs_torch,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache_manager.hybrid_cache,
            cache_position=cache_position,
        )[0]
        return _replicate(attn_out.reshape(1, 1, seq_len, self.config.hidden_size),
                          self.device, self.dtype)

    def _mamba_forward(self, normed, cache_manager, position_ids,
                       cache_position, seq_len) -> ttnn.Tensor:
        start_pos = position_ids[0, 0].item()
        if cache_position is None:
            cache_position = torch.arange(start_pos, start_pos + seq_len)

        if self.mamba is not None:
            out = self.mamba.forward(
                normed,
                cache_params=cache_manager.hybrid_cache,
                cache_position=cache_position,
                attention_mask=None,
            )
            if isinstance(out, ttnn.Tensor):
                return out
            return _replicate(out.reshape(1, 1, seq_len, self.config.hidden_size),
                              self.device, self.dtype)

        hs_torch = _tt_to_torch(normed, (1, seq_len, self.config.hidden_size))
        out_torch = self.hf_layer.mamba(
            hs_torch,
            cache_params=cache_manager.hybrid_cache,
            cache_position=cache_position,
            attention_mask=None,
        )
        return _replicate(out_torch.reshape(1, 1, seq_len, self.config.hidden_size),
                          self.device, self.dtype)

    def _mlp_forward(self, normed, seq_len, mode) -> ttnn.Tensor:
        hidden_size = self.config.hidden_size

        if self.tt_moe is not None and (seq_len == 1 or seq_len >= 32):
            moe_out_tt = self.tt_moe.forward(normed)
        else:
            hs_torch = _tt_to_torch(normed, (1, seq_len, hidden_size))
            moe_np = self.hf_layer.block_sparse_moe(hs_torch)[0]
            moe_out_tt = _replicate(moe_np.reshape(1, 1, seq_len, hidden_size), self.device, self.dtype)

        if self.shared_mlp is not None:
            shared_out_tt = self.shared_mlp.forward(normed, mode=mode)
        else:
            hs_torch = _tt_to_torch(normed, (1, seq_len, hidden_size))
            shared_np = self.hf_layer.shared_mlp(hs_torch)
            shared_out_tt = _replicate(shared_np.reshape(1, 1, seq_len, hidden_size), self.device, self.dtype)

        out = ttnn.add(moe_out_tt, shared_out_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        moe_out_tt.deallocate(True)
        shared_out_tt.deallocate(True)
        return out

    def reset_cache(self):
        if self.is_attention_layer and self.simple_attention is not None:
            attn = self.simple_attention
            mesh_mapper = attn._mesh_mapper
            k_shape = list(attn.cache_k.shape)
            v_shape = list(attn.cache_v.shape)
            attn.cache_k.deallocate(True)
            attn.cache_v.deallocate(True)
            attn.cache_k = ttnn.from_torch(
                torch.zeros(k_shape, dtype=torch.bfloat16),
                device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            attn.cache_v = ttnn.from_torch(
                torch.zeros(v_shape, dtype=torch.bfloat16),
                device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        elif self.mamba is not None:
            self.mamba.reset_state()
