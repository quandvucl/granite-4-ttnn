"""TT-Granite model configuration."""

from dataclasses import dataclass, field
from typing import List

import ttnn


@dataclass
class TTGraniteConfig:
    """Tenstorrent Granite model configuration (architecture + TT-specific settings)."""

    # Architecture (from HF config)
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 40
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    vocab_size: int = 49152
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    residual_multiplier: float = 0.22
    logits_scaling: float = 1.0
    embedding_multiplier: float = 1.0
    attention_layer_indices: List[int] = field(default_factory=lambda: [5, 15, 25, 35])

    # TT-specific
    dtype: str = "bfloat16"
    max_cache_length: int = 2048
    batch_size: int = 1

    def __post_init__(self):
        for idx in self.attention_layer_indices:
            if idx >= self.num_hidden_layers:
                raise ValueError(
                    f"Attention layer index {idx} exceeds num_hidden_layers {self.num_hidden_layers}"
                )

    @classmethod
    def from_hf_config(cls, hf_config, **kwargs):
        """Create TTGraniteConfig from a HuggingFace model config."""
        cfg = {
            "hidden_size": hf_config.hidden_size,
            "intermediate_size": hf_config.intermediate_size,
            "num_hidden_layers": hf_config.num_hidden_layers,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_key_value_heads": hf_config.num_key_value_heads,
            "vocab_size": hf_config.vocab_size,
            "max_position_embeddings": hf_config.max_position_embeddings,
            "rms_norm_eps": hf_config.rms_norm_eps,
        }
        for attr in ("residual_multiplier", "attention_layer_indices",
                     "logits_scaling", "embedding_multiplier"):
            if hasattr(hf_config, attr):
                cfg[attr] = getattr(hf_config, attr)
        cfg.update(kwargs)
        return cls(**cfg)

    def get_ttnn_dtype(self):
        mapping = {"bfloat16": ttnn.bfloat16, "bfloat8": ttnn.bfloat8_b, "float32": ttnn.float32}
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return mapping[self.dtype]

    def print_summary(self):
        print(f"  hidden={self.hidden_size}  layers={self.num_hidden_layers}"
              f"  attn_layers={self.attention_layer_indices}"
              f"  dtype={self.dtype}  cache={self.max_cache_length}")
