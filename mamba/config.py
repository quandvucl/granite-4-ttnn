"""Configuration classes for Mamba2 layers."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Mamba2Config:
    """Configuration for Mamba2 layer parameters."""

    # Model dimensions
    hidden_size: int
    intermediate_size: int
    num_heads: int
    head_dim: int
    ssm_state_size: int
    conv_dim: int
    conv_kernel: int
    chunk_size: int

    # Layer identification
    layer_idx: int

    # SSM parameters (time_step limits)
    time_step_min: float
    time_step_max: float

    # Additional parameters
    use_conv_bias: bool = True
    n_groups: int = 1

    @property
    def time_step_limit(self) -> Tuple[float, float]:
        """Time step limits for clamping."""
        return (self.time_step_min, self.time_step_max)

    @classmethod
    def from_hf_mamba(cls, hf_mamba) -> "Mamba2Config":
        """Create config from HuggingFace Granite Mamba layer."""
        return cls(
            hidden_size=hf_mamba.hidden_size,
            intermediate_size=hf_mamba.intermediate_size,
            num_heads=hf_mamba.num_heads,
            head_dim=hf_mamba.head_dim,
            ssm_state_size=hf_mamba.ssm_state_size,
            conv_dim=hf_mamba.conv_dim,
            conv_kernel=hf_mamba.conv_kernel_size,
            chunk_size=hf_mamba.chunk_size,
            layer_idx=hf_mamba.layer_idx,
            time_step_min=hf_mamba.time_step_min,
            time_step_max=hf_mamba.time_step_max,
            use_conv_bias=hf_mamba.use_conv_bias,
            n_groups=hf_mamba.n_groups,
        )


@dataclass
class MemoryConfig:
    """Memory configuration for TTNN operations."""

    # Default memory configs for different operation types
    default_layout: str = "TILE"
    matmul_memory_config: Optional[str] = None
    scan_memory_config: Optional[str] = None

    def get_config(self, shape: Tuple[int, ...], op_type: str):
        """Get memory config for specific operation."""
        # Placeholder - will be filled with actual logic
        return None
