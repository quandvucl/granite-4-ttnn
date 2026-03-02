import ttnn
import torch
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tt_ops.base import to_tt_tensor, to_torch_tensor


class HybridKVCacheManager:
    """
    Manages KV cache for hybrid Granite model (attention + mamba layers).

    For attention layers: stores K and V tensors on TT device
    For mamba layers: stores conv_state and ssm_state on CPU (empty on TT)
    """

    def __init__(
        self,
        device,
        num_layers: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_length: int = 2048,
        batch_size: int = 1,
        attention_layer_indices: Optional[List[int]] = None,
        dtype=ttnn.bfloat16
    ):
        self.device = device
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_length = max_cache_length
        self.batch_size = batch_size
        self.dtype = dtype

        # Default attention layers for granite-1b (layers 5, 15, 25, 35)
        self.attention_layer_indices = attention_layer_indices or [5, 15, 25, 35]

        # Cache storage
        # Format: {layer_idx: {"k": ttnn.Tensor, "v": ttnn.Tensor, "length": int}}
        self.kv_cache: Dict[int, Dict[str, Any]] = {}

        # Mamba state storage (CPU only)
        self.mamba_states: Dict[int, Dict[str, torch.Tensor]] = {}

        # Current position tracker
        self.current_position = 0

        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize empty cache for all layers."""
        for layer_idx in range(self.num_layers):
            if layer_idx in self.attention_layer_indices:
                # Attention layer - initialize empty K/V cache
                # Shape: [batch, n_kv_heads, 0, head_dim] (will grow with concat)
                self.kv_cache[layer_idx] = {
                    "k": None,
                    "v": None,
                    "length": 0
                }
            else:
                # Mamba layer - initialize empty state dict (float32 for precision)
                from tt_ops.mamba import TTMambaLayer
                self.mamba_states[layer_idx] = TTMambaLayer.init_cache(
                    batch_size=self.batch_size,
                    device='cpu',
                    dtype=torch.bfloat16
                )

    def update_attention_cache(
        self,
        layer_idx: int,
        k_new: ttnn.Tensor,
        v_new: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Update KV cache for an attention layer.

        Args:
            layer_idx: Layer index
            k_new: New key tensor [batch, n_kv_heads, seq_len, head_dim]
            v_new: New value tensor [batch, n_kv_heads, seq_len, head_dim]

        Returns:
            Full K and V tensors including new and cached
        """
        if layer_idx not in self.attention_layer_indices:
            raise ValueError(f"Layer {layer_idx} is not an attention layer")

        cache_entry = self.kv_cache[layer_idx]

        if cache_entry["k"] is None:
            # First time - initialize cache
            cache_entry["k"] = k_new
            cache_entry["v"] = v_new
            cache_entry["length"] = k_new.shape[2] if hasattr(k_new, 'shape') else 1
        else:
            # Concatenate new KV to existing cache along seq dimension (dim=2)
            # Convert to PyTorch for concatenation
            k_old_torch = to_torch_tensor(cache_entry["k"])
            v_old_torch = to_torch_tensor(cache_entry["v"])
            k_new_torch = to_torch_tensor(k_new)
            v_new_torch = to_torch_tensor(v_new)

            k_concat = torch.cat([k_old_torch, k_new_torch], dim=2)
            v_concat = torch.cat([v_old_torch, v_new_torch], dim=2)

            # Convert back to TTNN
            cache_entry["k"] = to_tt_tensor(k_concat, self.device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            cache_entry["v"] = to_tt_tensor(v_concat, self.device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            cache_entry["length"] += k_new.shape[2] if hasattr(k_new, 'shape') else 1

            # Check if we've exceeded max cache length
            if cache_entry["length"] > self.max_cache_length:
                print(f"Warning: Cache length ({cache_entry['length']}) exceeded max ({self.max_cache_length})")

        return cache_entry["k"], cache_entry["v"]

    def get_attention_cache(
        self,
        layer_idx: int
    ) -> Tuple[Optional[ttnn.Tensor], Optional[ttnn.Tensor]]:
        """
        Get cached K and V for an attention layer.

        Args:
            layer_idx: Layer index

        Returns:
            Cached K and V tensors, or (None, None) if cache is empty
        """
        if layer_idx not in self.attention_layer_indices:
            return None, None

        cache_entry = self.kv_cache[layer_idx]
        return cache_entry["k"], cache_entry["v"]

    def get_cache_length(self, layer_idx: int) -> int:
        """Get current cache length for a layer."""
        if layer_idx in self.attention_layer_indices:
            return self.kv_cache[layer_idx]["length"]
        else:
            return 0

    def update_mamba_state(
        self,
        layer_idx: int,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None
    ):
        """
        Update Mamba state for a mamba layer (CPU only).

        Args:
            layer_idx: Layer index
            conv_state: Convolution state
            ssm_state: SSM state
        """
        if layer_idx in self.attention_layer_indices:
            raise ValueError(f"Layer {layer_idx} is an attention layer, not mamba")

        if conv_state is not None:
            self.mamba_states[layer_idx]["conv_state"] = conv_state
        if ssm_state is not None:
            self.mamba_states[layer_idx]["ssm_state"] = ssm_state

    def get_mamba_state(
        self,
        layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get Mamba state for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Convolution state and SSM state
        """
        if layer_idx not in self.mamba_states:
            return None, None

        state = self.mamba_states[layer_idx]
        return state["conv_state"], state["ssm_state"]

    def get_mamba_cache(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get Mamba cache dict for TTMambaLayer.

        Args:
            layer_idx: Layer index

        Returns:
            Cache dict with 'conv_state' and 'ssm_state'
        """
        if layer_idx not in self.mamba_states:
            # Initialize if not present
            from tt_ops.mamba import TTMambaLayer
            self.mamba_states[layer_idx] = TTMambaLayer.init_cache(
                batch_size=self.batch_size,
                device='cpu',
                dtype=torch.bfloat16
            )

        return self.mamba_states[layer_idx]

    def increment_position(self, num_tokens: int = 1):
        """Increment current position counter."""
        self.current_position += num_tokens

    def get_position(self) -> int:
        """Get current position."""
        return self.current_position

    def reset(self):
        """Reset all cache and position."""
        self._initialize_cache()
        self.current_position = 0

    def clear_layer(self, layer_idx: int):
        """Clear cache for a specific layer."""
        if layer_idx in self.attention_layer_indices:
            self.kv_cache[layer_idx] = {
                "k": None,
                "v": None,
                "length": 0
            }
        else:
            from tt_ops.mamba import TTMambaLayer
            self.mamba_states[layer_idx] = TTMambaLayer.init_cache(
                batch_size=self.batch_size,
                device='cpu',
                dtype=torch.bfloat16
            )

    def get_memory_usage(self) -> int:
        """Estimate cache memory usage in bytes."""
        total_bytes = 0

        for layer_idx, cache_entry in self.kv_cache.items():
            if cache_entry["k"] is not None:
                # K and V: [batch, n_kv_heads, cache_len, head_dim]
                cache_len = cache_entry["length"]
                # bfloat16 = 2 bytes per element
                bytes_per_kv = self.batch_size * self.num_kv_heads * cache_len * self.head_dim * 2
                total_bytes += 2 * bytes_per_kv  # K and V

        return total_bytes

    def print_summary(self):
        """Print cache summary."""
        print(f"\n=== KV Cache Summary ===")
        print(f"Current position: {self.current_position}")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory usage: {self.get_memory_usage() / 1024**2:.2f} MB")
        print(f"\nAttention layers:")
        for layer_idx in self.attention_layer_indices:
            cache_len = self.kv_cache[layer_idx]["length"]
            print(f"  Layer {layer_idx}: cache_length={cache_len}")
        print(f"\nMamba layers: {len(self.mamba_states)} layers with CPU state")
