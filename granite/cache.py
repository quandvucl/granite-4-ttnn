"""Mamba cache manager for Granite hybrid model."""

from typing import Dict, List, Optional, Tuple
import torch
from mamba.utils import init_mamba_cache


class MambaCacheManager:
    """
    Manages Mamba state cache for Granite hybrid model.

    Note: Attention layers now use Attention1D's internal KV cache.
    This manager only handles Mamba conv_state and ssm_state.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int = 1,
        attention_layer_indices: Optional[List[int]] = None,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Default attention layers for granite (layers 5, 15, 25, 35)
        self.attention_layer_indices = attention_layer_indices or [5, 15, 25, 35]

        # Mamba state storage (CPU only)
        self.mamba_states: Dict[int, Dict[str, torch.Tensor]] = {}

        # Current position tracker
        self.current_position = 0

        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize empty Mamba state for all non-attention layers."""
        for layer_idx in range(self.num_layers):
            if layer_idx not in self.attention_layer_indices:
                # Mamba layer - initialize empty state dict
                self.mamba_states[layer_idx] = init_mamba_cache(
                    batch_size=self.batch_size, device="cpu", dtype=torch.bfloat16
                )

    def update_mamba_state(
        self,
        layer_idx: int,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
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
        self, layer_idx: int
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
        Get Mamba cache dict for Mamba forward pass.

        Args:
            layer_idx: Layer index

        Returns:
            Cache dict with 'conv_state' and 'ssm_state'
        """
        if layer_idx not in self.mamba_states:
            self.mamba_states[layer_idx] = init_mamba_cache(
                batch_size=self.batch_size, device="cpu", dtype=torch.bfloat16
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
        """Clear Mamba state for a specific layer."""
        if layer_idx not in self.attention_layer_indices:
            self.mamba_states[layer_idx] = init_mamba_cache(
                batch_size=self.batch_size, device="cpu", dtype=torch.bfloat16
            )

    def print_summary(self):
        """Print cache summary."""
        print(f"\n=== Mamba Cache Summary ===")
        print(f"Current position: {self.current_position}")
        print(f"Batch size: {self.batch_size}")
        print(
            f"Attention layers (managed by Attention1D): {len(self.attention_layer_indices)}"
        )
        print(f"Mamba layers: {len(self.mamba_states)} layers with CPU state")
        print("=" * 30)
