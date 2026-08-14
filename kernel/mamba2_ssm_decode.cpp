//
// Fused Mamba2 SSM decode-step: collapses multiple Python-dispatched TTNN ops into
// a single C++ call, removing per-op Python dispatch overhead.
//
// Implements (see forward_decode in mamba_chunk_scan_parallel.py):
//   dt       = softplus(dt_raw + dt_bias, beta=1, threshold=20)
//   dt       = clip(dt, dt_min, dt_max)
//   dA       = exp(dt.unsqueeze(-1) * A)              [B,H,D,N]
//   dBx      = (dt * x).unsqueeze(-1) * B.unsqueeze(-2) [B,H,D,N]
//   new_state = addcmul(dBx, dA, state_in, 1.0)       [B,H,D,N]
//   y        = sum(new_state * C.unsqueeze(-2), dim=-1) [B,H,D]
//   y        = addcmul(y, D, x, 1.0)                   [B,H,D]
//
// B and C are [B, H, N] on entry (already group-expanded by caller).
// Returns {y [B,H,D], new_state [B,H,D,N]}.

#include "mamba2_ssm_decode.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

// Unary ops
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

// Binary ops
#include "ttnn/operations/eltwise/binary/binary.hpp"

// Ternary ops
#include "ttnn/operations/eltwise/ternary/ternary_composite_op.hpp"

// Reduction
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn_granite
{

    using Tensor = ttnn::Tensor;
    using MemCfg = tt::tt_metal::MemoryConfig;

    // unsqueeze via tt::tt_metal::Tensor::reshape() — always exported
    static Tensor unsqueeze_at(const Tensor &t, int dim)
    {
        auto shape = t.logical_shape();
        const uint32_t rank = shape.rank();
        int pos = (dim < 0) ? static_cast<int>(rank) + 1 + dim : dim;
        ttsl::SmallVector<uint32_t> sv;
        sv.reserve(rank + 1);
        for (uint32_t i = 0; i <= rank; ++i)
        {
            if (static_cast<int>(i) == pos)
                sv.push_back(1);
            if (i < rank)
                sv.push_back(shape[i]);
        }
        return t.reshape(tt::tt_metal::Shape(std::move(sv)));
    }

    std::pair<Tensor, Tensor> mamba2_ssm_decode(
        const Tensor &dt_raw,
        const Tensor &dt_bias,
        const Tensor &x,
        const Tensor &B,
        const Tensor &C,
        const Tensor &A,
        const Tensor &D,
        const Tensor &state_in,
        float dt_min,
        float dt_max,
        const std::optional<MemCfg> &memory_config)
    {
        const MemCfg mc = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

        // dt = clip(softplus(dt_raw + dt_bias), dt_min, dt_max)
        Tensor dt_sum = ttnn::add(dt_raw, dt_bias, std::nullopt, mc);
        Tensor dt_sp = ttnn::softplus(dt_sum, 1.0f, 20.0f, mc);
        dt_sum.deallocate(true);
        Tensor dt = ttnn::operations::unary::ExecuteUnaryCompositeClip::invoke(
            dt_sp, dt_min, dt_max, mc);
        dt_sp.deallocate(true);

        // dA = exp(unsqueeze(dt, -1) * A)
        Tensor dt_4d = unsqueeze_at(dt, -1);
        Tensor dA_log = ttnn::multiply(dt_4d, A, std::nullopt, mc);
        dt_4d.deallocate(true);
        Tensor dA = ttnn::exp(dA_log, false, mc);
        dA_log.deallocate(true);

        // dBx = unsqueeze(dt * x, -1) * unsqueeze(B, -2)
        Tensor dtx = ttnn::multiply(dt, x, std::nullopt, mc);
        dt.deallocate(true);
        Tensor dtx_4d = unsqueeze_at(dtx, -1);
        dtx.deallocate(true);
        Tensor B_4d = unsqueeze_at(B, -2);
        Tensor dBx = ttnn::multiply(dtx_4d, B_4d, std::nullopt, mc);
        dtx_4d.deallocate(true);
        B_4d.deallocate(true);

        // new_state = dBx + dA * state_in
        Tensor new_state = ttnn::operations::ternary::_addcmul(
            dBx, dA, state_in, 1.0f, mc);
        dA.deallocate(true);
        dBx.deallocate(true);

        // y = sum(new_state * unsqueeze(C, -2), dim=-1)
        Tensor C_4d = unsqueeze_at(C, -2);
        Tensor yC = ttnn::multiply(new_state, C_4d, std::nullopt, mc);
        C_4d.deallocate(true);
        Tensor y = ttnn::operations::reduction::sum(
            yC,
            std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>(-1),
            /*keepdim=*/true,
            mc);
        yC.deallocate(true);

        // y = y + D * x
        Tensor y_out = ttnn::operations::ternary::_addcmul(y, D, x, 1.0f, mc);
        y.deallocate(true);

        return {y_out, new_state};
    }

} // namespace ttnn_granite
