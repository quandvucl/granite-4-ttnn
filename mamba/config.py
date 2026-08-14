"""Configuration for Mamba2 layers."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Mamba2Config:
    """Mamba2 layer parameters extracted from a HuggingFace Granite layer."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    head_dim: int
    ssm_state_size: int
    conv_dim: int
    conv_kernel: int
    chunk_size: int
    layer_idx: int
    time_step_min: float
    time_step_max: float
    use_conv_bias: bool = True
    n_groups: int = 1

    @property
    def time_step_limit(self) -> Tuple[float, float]:
        return (self.time_step_min, self.time_step_max)

    @classmethod
    def from_hf_mamba(cls, hf_mamba) -> "Mamba2Config":
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
