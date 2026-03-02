import ttnn
import torch
import math
from typing import Tuple
from .base import TTOperation, to_tt_tensor, to_tile_layout


class TTRotaryEmbedding(TTOperation):
    """
    Rotary Position Embeddings (RoPE) on TTNN.

    RoPE applies a rotation to query and key vectors based on their position,
    enabling the model to capture relative position information.
    """

    def __init__(
        self,
        device,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        dtype=ttnn.bfloat16
    ):
        super().__init__(device, dtype)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.inv_freq = inv_freq

        # Precompute cos and sin for all positions
        self._precompute_freqs_cis(max_position_embeddings)

    def _precompute_freqs_cis(self, max_seq_len: int):
        """Precompute cos and sin values for all positions."""
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [max_seq_len, dim//2]

        # Concatenate to match full dimension
        freqs = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]

        # Compute cos and sin
        self.cos_cached = torch.cos(freqs).to(torch.bfloat16)  # [max_seq_len, dim]
        self.sin_cached = torch.sin(freqs).to(torch.bfloat16)  # [max_seq_len, dim]

        # Convert to TT tensors (keep in ROW_MAJOR for indexing)
        self.cos_cached_tt = to_tt_tensor(
            self.cos_cached, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.sin_cached_tt = to_tt_tensor(
            self.sin_cached, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def get_cos_sin(
        self,
        position_ids: torch.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get cos and sin values for given position IDs.

        Args:
            position_ids: [batch_size, seq_len] or [batch_size]

        Returns:
            cos, sin tensors [batch_size, seq_len, dim] or [batch_size, 1, dim]
        """
        # Index into cached values
        cos = self.cos_cached[position_ids]  # [batch, seq, dim] or [batch, dim]
        sin = self.sin_cached[position_ids]

        # Add seq dimension if needed (for single token case)
        if cos.dim() == 2:
            cos = cos.unsqueeze(1)  # [batch, 1, dim]
            sin = sin.unsqueeze(1)

        # Convert to TT tensors
        cos_tt = to_tt_tensor(cos, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        sin_tt = to_tt_tensor(sin, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        return cos_tt, sin_tt

    def forward(
        self,
        q_tt: ttnn.Tensor,
        k_tt: ttnn.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q_tt: Query tensor [batch, n_heads, seq_len, head_dim] (TILE)
            k_tt: Key tensor [batch, n_kv_heads, seq_len, head_dim] (TILE)
            position_ids: Position IDs [batch, seq_len]

        Returns:
            Rotated query and key tensors
        """
        # Get cos and sin for positions
        cos_tt, sin_tt = self.get_cos_sin(position_ids)

        # Apply rotation to Q and K
        q_rotated = self._apply_rotation(q_tt, cos_tt, sin_tt)
        k_rotated = self._apply_rotation(k_tt, cos_tt, sin_tt)

        return q_rotated, k_rotated

    def _apply_rotation(
        self,
        x_tt: ttnn.Tensor,
        cos_tt: ttnn.Tensor,
        sin_tt: ttnn.Tensor
    ) -> ttnn.Tensor:
        """
        Apply rotation: x_rotated = x*cos + rotate_half(x)*sin

        Args:
            x_tt: Input tensor [batch, n_heads, seq_len, head_dim]
            cos_tt: Cosine values [batch, seq_len, head_dim]
            sin_tt: Sine values [batch, seq_len, head_dim]

        Returns:
            Rotated tensor [batch, n_heads, seq_len, head_dim]
        """
        # For simplicity, we'll do rotation on CPU for now
        # A full TTNN implementation would require complex tensor manipulation
        # TODO: Optimize this by implementing rotation directly in TTNN

        # Convert to torch
        x = x_tt.to_torch()
        cos = cos_tt.to_torch()
        sin = sin_tt.to_torch()

        # Reshape cos/sin to match x dimensions [batch, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Apply rotation
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)

        # Convert back to TT
        x_rotated_tt = to_tt_tensor(x_rotated, self.device, self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Convert to TILE if input was TILE
        if x_tt.layout == ttnn.TILE_LAYOUT:
            x_rotated_tt = to_tile_layout(x_rotated_tt)

        return x_rotated_tt

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the dimensions of x.

        For RoPE, we split the last dimension in half, negate the first half,
        and swap the two halves.
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class TTRotaryEmbeddingOptimized(TTOperation):
    """
    Optimized Rotary Position Embeddings for decode mode.

    For single token generation (seq_len=1), we can avoid complex
    tensor operations and just scale by cos/sin directly.
    """

    def __init__(
        self,
        device,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        dtype=ttnn.bfloat16
    ):
        super().__init__(device, dtype)
        self.base_rope = TTRotaryEmbedding(device, dim, max_position_embeddings, base, dtype)

    def forward_decode(
        self,
        q_tt: ttnn.Tensor,
        k_tt: ttnn.Tensor,
        position: int
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Apply rotary embeddings for a single token (decode mode).

        Args:
            q_tt: Query tensor [batch, n_heads, 1, head_dim]
            k_tt: Key tensor [batch, n_kv_heads, 1, head_dim]
            position: Current position (scalar)

        Returns:
            Rotated query and key tensors
        """
        # For single token, create position_ids [batch, 1]
        batch_size = q_tt.shape[0] if hasattr(q_tt, 'shape') else 1
        position_ids = torch.full((batch_size, 1), position, dtype=torch.long)

        return self.base_rope.forward(q_tt, k_tt, position_ids)
