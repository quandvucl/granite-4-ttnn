"""TT-Granite model configuration."""

from dataclasses import dataclass
from typing import List

import ttnn


@dataclass
class TTGraniteConfig:
    """
    Configuration for Tenstorrent Granite model.

    Extends HuggingFace config with TT-specific settings.
    """

    # Model architecture (from HF config)
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 40
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    vocab_size: int = 49152
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    residual_multiplier: float = 0.22
    logits_scaling: float = 1.0  # Logits scaling factor (granite-1b uses 6.0)
    embedding_multiplier: float = 1.0  # Embedding scaling factor (granite-1b uses 12.0)

    # Attention layer indices (granite-1b: layers 5, 15, 25, 35)
    attention_layer_indices: List[int] = None

    # TT-specific settings
    device_id: int = 0
    dtype: str = "bfloat16"
    use_tile_layout_for_compute: bool = True
    max_cache_length: int = 2048
    batch_size: int = 1  # Fixed at 1 for single-prompt optimization

    # Performance tuning
    enable_profiling: bool = False
    verbose: bool = True

    def __post_init__(self):
        # Default attention layer indices
        if self.attention_layer_indices is None:
            self.attention_layer_indices = [5, 15, 25, 35]

        # Validate attention layer indices
        for idx in self.attention_layer_indices:
            if idx >= self.num_hidden_layers:
                raise ValueError(
                    f"Attention layer index {idx} exceeds num_hidden_layers {self.num_hidden_layers}"
                )

    @classmethod
    def from_hf_config(cls, hf_config, **kwargs):
        """
        Create TTGraniteConfig from HuggingFace config.

        Args:
            hf_config: HuggingFace model config
            **kwargs: Additional TT-specific overrides

        Returns:
            TTGraniteConfig instance
        """
        # Extract HF config attributes
        config_dict = {
            "hidden_size": hf_config.hidden_size,
            "intermediate_size": hf_config.intermediate_size,
            "num_hidden_layers": hf_config.num_hidden_layers,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_key_value_heads": hf_config.num_key_value_heads,
            "vocab_size": hf_config.vocab_size,
            "max_position_embeddings": hf_config.max_position_embeddings,
            "rms_norm_eps": hf_config.rms_norm_eps,
        }

        # Add optional attributes if they exist
        if hasattr(hf_config, "rope_theta"):
            config_dict["rope_theta"] = hf_config.rope_theta
        if hasattr(hf_config, "attention_dropout"):
            config_dict["attention_dropout"] = hf_config.attention_dropout
        if hasattr(hf_config, "residual_multiplier"):
            config_dict["residual_multiplier"] = hf_config.residual_multiplier
        if hasattr(hf_config, "attention_layer_indices"):
            config_dict["attention_layer_indices"] = hf_config.attention_layer_indices
        if hasattr(hf_config, "logits_scaling"):
            config_dict["logits_scaling"] = hf_config.logits_scaling
        if hasattr(hf_config, "embedding_multiplier"):
            config_dict["embedding_multiplier"] = hf_config.embedding_multiplier

        # Override with kwargs
        config_dict.update(kwargs)

        return cls(**config_dict)

    def get_ttnn_dtype(self):
        """Get TTNN dtype from string."""
        if self.dtype == "bfloat16":
            return ttnn.bfloat16
        elif self.dtype == "bfloat8":
            return ttnn.bfloat8_b
        elif self.dtype == "float32":
            return ttnn.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def print_summary(self):
        """Print configuration summary."""
        print("\n=== TTGranite Configuration ===")
        print(f"Model Architecture:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Intermediate size: {self.intermediate_size}")
        print(f"  Num layers: {self.num_hidden_layers}")
        print(f"  Num attention heads: {self.num_attention_heads}")
        print(f"  Num KV heads: {self.num_key_value_heads}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Max position embeddings: {self.max_position_embeddings}")
        print(f"  Residual multiplier: {self.residual_multiplier}")
        print(f"\nLayer Configuration:")
        print(f"  Attention layers: {self.attention_layer_indices}")
        print(
            f"  Mamba layers: {[i for i in range(self.num_hidden_layers) if i not in self.attention_layer_indices]}"
        )
        print(f"\nTT Settings:")
        print(f"  Device ID: {self.device_id}")
        print(f"  Dtype: {self.dtype}")
        print(f"  Max cache length: {self.max_cache_length}")
        print(f"  Batch size: {self.batch_size}")
