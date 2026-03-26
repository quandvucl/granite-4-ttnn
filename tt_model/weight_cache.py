"""Model weight caching on Tenstorrent device."""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import ttnn

sys.path.append(str(Path(__file__).parent.parent))
from tt_ops.base import to_tt_tensor


class WeightCache:
    """
    Manages model weights on Tenstorrent device.

    Converts HuggingFace model weights to TTNN format and caches them
    on device in optimal layout (TILE_LAYOUT for matmuls, ROW_MAJOR for elementwise ops).
    """

    def __init__(self, device, dtype=ttnn.bfloat16):
        self.device = device
        self.dtype = dtype
        self.cache: Dict[str, ttnn.Tensor] = {}
        self.metadata: Dict[str, Dict] = {}

    def store(
        self,
        name: str,
        weight: torch.Tensor,
        layout: str = "TILE",
        transpose: bool = False,
    ) -> ttnn.Tensor:
        """
        Store a weight tensor in the cache.

        Args:
            name: Unique identifier for the weight
            weight: PyTorch weight tensor
            layout: "TILE" (for matmul) or "ROW_MAJOR" (for elementwise)
            transpose: Whether to transpose the weight (for linear layers)

        Returns:
            TTNN tensor stored in cache
        """
        if name in self.cache:
            return self.cache[name]

        # Transpose if needed (PyTorch linear weights are [out, in], matmul needs [in, out])
        if transpose:
            weight = weight.T.contiguous()

        # Convert to appropriate layout
        if layout == "TILE":
            tt_layout = ttnn.TILE_LAYOUT
        else:
            tt_layout = ttnn.ROW_MAJOR_LAYOUT

        # Convert to TTNN tensor
        weight_tt = to_tt_tensor(weight, self.device, self.dtype, layout=tt_layout)

        # Cache it
        self.cache[name] = weight_tt
        self.metadata[name] = {
            "shape": tuple(weight.shape),
            "layout": layout,
            "transpose": transpose,
            "dtype": str(self.dtype),
        }

        return weight_tt

    def get(self, name: str) -> Optional[ttnn.Tensor]:
        """
        Retrieve a weight tensor from cache.

        Args:
            name: Unique identifier for the weight

        Returns:
            Cached TTNN tensor or None if not found
        """
        return self.cache.get(name)

    def has(self, name: str) -> bool:
        """Check if a weight is cached."""
        return name in self.cache

    def clear(self):
        """Clear all cached weights."""
        self.cache.clear()
        self.metadata.clear()

    def get_memory_usage(self) -> int:
        """Estimate total memory usage in bytes."""
        total_bytes = 0
        for name, weight_tt in self.cache.items():
            meta = self.metadata[name]
            shape = meta["shape"]
            # bfloat16 = 2 bytes per element
            bytes_per_elem = 2 if "bfloat16" in meta["dtype"] else 4
            total_bytes += torch.prod(torch.tensor(shape)).item() * bytes_per_elem
        return total_bytes

    def print_summary(self):
        """Print summary of cached weights."""
        print(f"\n=== Weight Cache Summary ===")
        print(f"Total weights cached: {len(self.cache)}")
        print(f"Estimated memory usage: {self.get_memory_usage() / 1024**2:.2f} MB")
        print(f"\nWeights:")
        for name, meta in self.metadata.items():
            print(f"  {name}: shape={meta['shape']}, layout={meta['layout']}")


def convert_hf_weights_to_cache(
    hf_model, device, dtype=ttnn.bfloat16, verbose: bool = True
) -> WeightCache:
    """
    Convert HuggingFace model weights to TT weight cache.

    Args:
        hf_model: HuggingFace model instance
        device: TTNN device
        dtype: Target dtype
        verbose: Print progress

    Returns:
        WeightCache with all model weights
    """
    cache = WeightCache(device, dtype)

    if verbose:
        print("Converting HuggingFace weights to TTNN format...")

    # Count total parameters
    total_params = sum(p.numel() for p in hf_model.parameters())
    if verbose:
        print(f"Total parameters: {total_params / 1e6:.1f}M")

    # Convert embeddings (keep in ROW_MAJOR for indexing)
    if hasattr(hf_model.model, "embed_tokens"):
        cache.store(
            "embed_tokens.weight",
            hf_model.model.embed_tokens.weight,
            layout="ROW_MAJOR",
        )

    # Convert layer weights
    for layer_idx, layer in enumerate(hf_model.model.layers):
        prefix = f"layers.{layer_idx}"

        # Check if this is an attention layer or mamba layer
        is_attention = hasattr(layer, "self_attn") and layer.self_attn is not None
        is_mamba = hasattr(layer, "mamba") and layer.mamba is not None

        if is_attention:
            # Attention layer weights
            attn = layer.self_attn

            # QKV projections (linear layers - transpose for matmul)
            cache.store(
                f"{prefix}.self_attn.q_proj.weight",
                attn.q_proj.weight,
                layout="TILE",
                transpose=True,
            )
            cache.store(
                f"{prefix}.self_attn.k_proj.weight",
                attn.k_proj.weight,
                layout="TILE",
                transpose=True,
            )
            cache.store(
                f"{prefix}.self_attn.v_proj.weight",
                attn.v_proj.weight,
                layout="TILE",
                transpose=True,
            )
            cache.store(
                f"{prefix}.self_attn.o_proj.weight",
                attn.o_proj.weight,
                layout="TILE",
                transpose=True,
            )

        elif is_mamba:
            # Mamba layer weights stay on CPU - no need to cache
            pass

        # MLP weights (each layer has its own despite the "shared_mlp" name)
        if hasattr(layer, "shared_mlp") and layer.shared_mlp is not None:
            mlp = layer.shared_mlp
            cache.store(
                f"{prefix}.shared_mlp.input_linear.weight",
                mlp.input_linear.weight,
                layout="TILE",
                transpose=True,
            )
            cache.store(
                f"{prefix}.shared_mlp.output_linear.weight",
                mlp.output_linear.weight,
                layout="TILE",
                transpose=True,
            )

            if hasattr(mlp.input_linear, "bias") and mlp.input_linear.bias is not None:
                cache.store(
                    f"{prefix}.shared_mlp.input_linear.bias",
                    mlp.input_linear.bias,
                    layout="ROW_MAJOR",
                )
            if (
                hasattr(mlp.output_linear, "bias")
                and mlp.output_linear.bias is not None
            ):
                cache.store(
                    f"{prefix}.shared_mlp.output_linear.bias",
                    mlp.output_linear.bias,
                    layout="ROW_MAJOR",
                )

        # Layer norms
        if hasattr(layer, "input_layernorm"):
            cache.store(
                f"{prefix}.input_layernorm.weight",
                layer.input_layernorm.weight,
                layout="ROW_MAJOR",
            )

        if hasattr(layer, "post_attention_layernorm"):
            cache.store(
                f"{prefix}.post_attention_layernorm.weight",
                layer.post_attention_layernorm.weight,
                layout="ROW_MAJOR",
            )

    # Final norm
    if hasattr(hf_model.model, "norm"):
        cache.store("norm.weight", hf_model.model.norm.weight, layout="ROW_MAJOR")

    # LM head (keep on CPU for now - large vocab matmul)
    # We'll handle this separately during inference

    if verbose:
        cache.print_summary()

    return cache
