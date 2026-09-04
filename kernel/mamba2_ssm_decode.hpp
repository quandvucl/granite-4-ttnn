#pragma once

#include <optional>
#include <utility>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn_granite
{

    /**
     * Fused Mamba2 SSM decode step.
     *
     * dt       = clip(softplus(dt_raw + dt_bias), dt_min, dt_max)
     * dA       = exp(dt_expanded * A)
     * dBx      = (dt * x) ⊙ B
     * new_state = dBx + dA * state_in
     * y        = sum(new_state ⊙ C, dim=-1) + D * x
     *
     * Returns {y [B,H,D], new_state [B,H,D,N]}.
     */
    std::pair<ttnn::Tensor, ttnn::Tensor> mamba2_ssm_decode(
        const ttnn::Tensor &dt_raw,   // [B, H, 1]
        const ttnn::Tensor &dt_bias,  // [1, H, 1]
        const ttnn::Tensor &x,        // [B, H, D]
        const ttnn::Tensor &B,        // [B, H, N]  (group-expanded)
        const ttnn::Tensor &C,        // [B, H, N]  (group-expanded)
        const ttnn::Tensor &A,        // [B, H, D, N] or broadcast-compatible
        const ttnn::Tensor &D,        // [1, H, D]
        const ttnn::Tensor &state_in, // [B, H, D, N]
        float dt_min, float dt_max,
        const std::optional<tt::tt_metal::MemoryConfig> &memory_config =
            std::nullopt);

} // namespace ttnn_granite
