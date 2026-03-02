import ttnn
import torch
from typing import Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tt_ops.base import to_tt_tensor, to_torch_tensor, to_tile_layout
from tt_ops.attention_tt import TTAttentionOptimized
from tt_ops.mlp import TTSharedMLP
from tt_ops.mamba import SimpleMamba2TTNN
from tt_ops.normalization import TTRMSNorm
from tt_ops.cache import HybridKVCacheManager


class TTGraniteDecoderLayer:
    """
    Hybrid Granite decoder layer supporting both attention and mamba variants.

    For attention layers (5, 15, 25, 35):
    - Runs attention and MLP on TT hardware
    - Uses KV cache for incremental decoding

    For mamba layers (all others):
    - Runs mamba on CPU using HF implementation
    - MLP still runs on TT hardware

    Architecture:
    - residual = hidden
    - hidden = norm(hidden)
    - hidden = attention/mamba(hidden)
    - hidden = residual + hidden * residual_multiplier
    - residual = hidden
    - hidden = norm(hidden)
    - hidden = mlp(hidden)
    - hidden = residual + hidden * residual_multiplier
    """

    def __init__(
        self,
        device,
        layer_idx: int,
        config,
        hf_layer,
        shared_mlp: TTSharedMLP,
        weight_cache,
        is_attention_layer: bool,
        hf_config=None,
        dtype=ttnn.bfloat16
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.config = config
        self.hf_config = hf_config  # HF config for Mamba cache
        self.hf_layer = hf_layer  # Reference to HF layer for mamba
        self.shared_mlp = shared_mlp
        self.is_attention_layer = is_attention_layer
        self.dtype = dtype

        # Residual multiplier (0.22 for granite-1b)
        self.residual_multiplier = config.residual_multiplier if hasattr(config, 'residual_multiplier') else 1.0

        # Layer norms
        input_norm_weight = weight_cache.get(f"layers.{layer_idx}.input_layernorm.weight")
        post_attn_norm_weight = weight_cache.get(f"layers.{layer_idx}.post_attention_layernorm.weight")

        if input_norm_weight is not None and post_attn_norm_weight is not None:
            # Convert to torch first
            input_norm_weight_torch = input_norm_weight.to_torch()
            post_attn_norm_weight_torch = post_attn_norm_weight.to_torch()

            self.input_layernorm = TTRMSNorm(device, input_norm_weight_torch, eps=config.rms_norm_eps, dtype=dtype)
            self.post_attention_layernorm = TTRMSNorm(device, post_attn_norm_weight_torch, eps=config.rms_norm_eps, dtype=dtype)
        else:
            # Fallback: load directly from HF layer
            self.input_layernorm = TTRMSNorm(device, hf_layer.input_layernorm.weight, eps=config.rms_norm_eps, dtype=dtype)
            self.post_attention_layernorm = TTRMSNorm(device, hf_layer.post_attention_layernorm.weight, eps=config.rms_norm_eps, dtype=dtype)

        # Attention or Mamba
        if is_attention_layer:
            # Use TT-optimized attention with HF core
            self.attention_optimized = TTAttentionOptimized(
                hf_attention=hf_layer.self_attn,
                device=device,
                dtype=dtype
            )
            self.attention = None
            self.mamba = None
        else:
            # Mamba layer - use SimpleMamba2TTNN (TTNN-ready wrapper)
            self.attention = None
            self.mamba = SimpleMamba2TTNN(
                hf_mamba=hf_layer.mamba,
                device=device,
                dtype=dtype
            ) if hasattr(hf_layer, 'mamba') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_manager: HybridKVCacheManager,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size] (PyTorch tensor)
            cache_manager: KV cache manager
            position_ids: [batch, seq_len] position indices
            attention_mask: Optional attention mask

        Returns:
            Output hidden states [batch, seq_len, hidden_size] (PyTorch tensor)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if self.is_attention_layer:
            # ===== ATTENTION PATH (use HF directly with hybrid cache) =====

            # Initialize hybrid cache on first use
            if not hasattr(cache_manager, 'hybrid_cache'):
                from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
                cache_manager.hybrid_cache = HybridMambaAttentionDynamicCache(
                    config=self.hf_config,
                    batch_size=batch_size,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )

            # Determine if we're in prefill or decode
            start_pos = position_ids[0, 0].item()
            is_prefill = (start_pos == 0)
            cache_manager.hybrid_cache.has_previous_state = not is_prefill

            # Pre-attention norm (use TT RMSNorm)
            residual = hidden_states
            hidden_states_tt = to_tt_tensor(hidden_states, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states_tt = self.input_layernorm(hidden_states_tt)
            hidden_states = to_torch_tensor(hidden_states_tt, target_shape=(batch_size, seq_len, hidden_size))

            # Attention - use TT-optimized attention (QKV/O projections with TT)
            # TT accelerates heavy matmuls, HF handles RoPE + cache
            attn_output, _ = self.attention_optimized.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache_manager.hybrid_cache,
                use_cache=True
            )

            # Residual connection with multiplier
            hidden_states = residual + attn_output * self.residual_multiplier

            # Pre-MLP norm (use TT RMSNorm)
            residual = hidden_states
            hidden_states_tt = to_tt_tensor(hidden_states, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states_tt = self.post_attention_layernorm(hidden_states_tt)

            # MLP (shared) - on TT hardware
            mlp_output_tt = self.shared_mlp.forward(hidden_states_tt)
            mlp_output = to_torch_tensor(mlp_output_tt, target_shape=(batch_size, seq_len, hidden_size))

            # Residual connection with multiplier
            hidden_states = residual + mlp_output * self.residual_multiplier

        else:
            # ===== MAMBA PATH (Use HF Mamba with proper cache) =====

            # Ensure correct shape for Mamba [batch, seq, hidden]
            if hidden_states.ndim != 3:
                hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

            # Pre-mamba norm (use TT RMSNorm)
            residual = hidden_states
            hidden_states_tt = to_tt_tensor(hidden_states, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states_tt = self.input_layernorm(hidden_states_tt)
            hidden_states = to_torch_tensor(hidden_states_tt, target_shape=(batch_size, seq_len, hidden_size))

            # Use SimpleMamba2TTNN (TTNN-ready wrapper)
            if self.mamba is not None:
                # Initialize cache on first use
                if not hasattr(cache_manager, 'hybrid_cache'):
                    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
                    cache_manager.hybrid_cache = HybridMambaAttentionDynamicCache(
                        config=self.hf_config,
                        batch_size=batch_size,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device
                    )

                # Determine if we're in prefill (start_pos == 0) or decode (start_pos > 0)
                start_pos = position_ids[0, 0].item()
                is_prefill = (start_pos == 0)

                # Set has_previous_state based on whether we're in prefill or decode
                cache_manager.hybrid_cache.has_previous_state = not is_prefill

                # Calculate cache_position based on current position
                cache_position = torch.arange(start_pos, start_pos + seq_len, device=hidden_states.device)

                # Run SimpleMamba2TTNN (uses HF core with TTNN-ready weights)
                hidden_states = self.mamba.forward(
                    hidden_states,
                    cache_params=cache_manager.hybrid_cache,
                    cache_position=cache_position,
                    attention_mask=None
                )
            else:
                raise ValueError(f"Layer {self.layer_idx} does not have mamba attribute")

            # Residual connection with multiplier
            hidden_states = residual + hidden_states * self.residual_multiplier

            # Pre-MLP norm (use TT RMSNorm)
            residual = hidden_states
            hidden_states_tt = to_tt_tensor(hidden_states, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states_tt = self.post_attention_layernorm(hidden_states_tt)

            # MLP (shared) - on TT hardware
            mlp_output_tt = self.shared_mlp.forward(hidden_states_tt)
            mlp_output = to_torch_tensor(mlp_output_tt, target_shape=(batch_size, seq_len, hidden_size))

            # Residual connection with multiplier
            hidden_states = residual + mlp_output * self.residual_multiplier

        return hidden_states
