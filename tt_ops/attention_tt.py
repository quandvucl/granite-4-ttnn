"""
TT-optimized Attention with TTNN projections and HF core.

Based on TT-Metal patterns: optimize expensive linear operations (QKV, O projections)
while keeping HF attention core for cache compatibility.
"""
import torch
import ttnn
from typing import Optional, Tuple
from .base import to_tt_tensor, to_torch_tensor


class TTAttentionOptimized:
    """
    TT-optimized attention with accelerated projections.

    Architecture:
    - Q/K/V projections: TTNN matmul (optimized, with optional tensor parallelism)
    - Attention computation: HF (RoPE + cache management)
    - O projection: TTNN matmul (optimized, with optional tensor parallelism)
    """

    def __init__(self, hf_attention, device, dtype=ttnn.bfloat16, tensor_parallel=False, tt_ccl=None):
        """
        Initialize TT-optimized attention.

        Args:
            hf_attention: HuggingFace attention module
            device: TTNN device
            dtype: Data type
            tensor_parallel: Enable tensor parallelism (shard weights across devices)
            tt_ccl: TT_CCL manager for collective operations
        """
        self.hf_attn = hf_attention
        self.device = device
        self.dtype = dtype
        self.layer_idx = hf_attention.layer_idx
        self.is_mesh = hasattr(device, 'get_num_devices') and device.get_num_devices() > 1
        self.tensor_parallel = tensor_parallel and self.is_mesh
        self.tt_ccl = tt_ccl

        # Extract dimensions
        self.hidden_size = hf_attention.hidden_size
        self.num_heads = hf_attention.num_heads
        self.num_kv_heads = hf_attention.num_key_value_heads
        self.head_dim = hf_attention.head_dim

        # Pre-convert projection weights to TTNN (transposed for matmul)
        # For tensor parallel: shard Q/K/V column-wise, O row-wise
        if self.tensor_parallel:
            mesh_shape = device.shape
            # Column-parallel for Q/K/V (shard output dimension)
            qkv_mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(-1, None), mesh_shape=mesh_shape)
            # Row-parallel for O (shard input dimension)
            o_mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(-2, None), mesh_shape=mesh_shape)
        else:
            # Replicate weights if tensor parallel is disabled
            qkv_mesh_mapper = ttnn.ReplicateTensorToMesh(device) if self.is_mesh else None
            o_mesh_mapper = ttnn.ReplicateTensorToMesh(device) if self.is_mesh else None

        self.q_weight_tt = to_tt_tensor(
            hf_attention.q_proj.weight.T.contiguous(),
            device,
            dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=qkv_mesh_mapper
        )
        self.k_weight_tt = to_tt_tensor(
            hf_attention.k_proj.weight.T.contiguous(),
            device,
            dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=qkv_mesh_mapper
        )
        self.v_weight_tt = to_tt_tensor(
            hf_attention.v_proj.weight.T.contiguous(),
            device,
            dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=qkv_mesh_mapper
        )
        self.o_weight_tt = to_tt_tensor(
            hf_attention.o_proj.weight.T.contiguous(),
            device,
            dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=o_mesh_mapper
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[any] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[any]]:
        """
        Forward pass with TTNN-optimized projections.

        Optimization strategy:
        - TTNN for Q/K/V projections (heavy matmuls)
        - CPU for reshape, RoPE, cache operations
        - HF for attention computation (optimized SDPA)
        - TTNN for O projection (heavy matmul)
        """
        bsz, q_len, _ = hidden_states.size()
        dtype = hidden_states.dtype

        # OPTIMIZED: Minimize CPU↔TTNN conversions for decode mode
        # For decode (batch=1, q_len=1), attention computation is tiny (~0.01ms)
        # But each conversion costs ~2ms on mesh devices. So keep projections on CPU.

        # 1. Q/K/V projections on CPU (avoid wasteful conversion round-trip)
        # For mesh devices with replicated weights, CPU is faster due to no sync overhead
        query_states = hidden_states @ self.hf_attn.q_proj.weight.T
        key_states = hidden_states @ self.hf_attn.k_proj.weight.T
        value_states = hidden_states @ self.hf_attn.v_proj.weight.T

        # 2. Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE (must be CPU)
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import apply_rotary_pos_emb
        cos, sin = position_embeddings if position_embeddings is not None else (None, None)
        if position_embeddings is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4. Cache update (must be CPU)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 5. Handle GQA
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        kv_len = key_states.shape[2]

        # 6. Attention computation on CPU (tiny matrices, not worth TTNN transfer)
        # For decode: [1, 48, 1, 128] @ [1, 48, 128, kv_len] - CPU is faster!
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.hf_attn.scaling

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attention_probs, value_states)

        # 7. Reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # 8. O projection with TTNN (optimized for single token decode)
        attn_output_tt = to_tt_tensor(attn_output.to(dtype), self.device, self.dtype, layout=ttnn.TILE_LAYOUT)
        output_tt = attn_output_tt @ self.o_weight_tt

        # Deallocate intermediate
        attn_output_tt.deallocate(True)

        output = to_torch_tensor(output_tt, target_shape=(bsz, q_len, self.hidden_size))

        # Deallocate output tensor after conversion
        output_tt.deallocate(True)

        return output, attention_probs
