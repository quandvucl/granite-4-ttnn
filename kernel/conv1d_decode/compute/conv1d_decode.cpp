//
// Compute kernel: conv1d_decode
//
// Fused causal conv1d decode step for Mamba2.
//
// For each C-tile c:
//   1. Shift: new_cache[k] = old_cache[k+1]  k=0..K-2
//             new_cache[K-1] = xBC_new
//   2. Accumulate: acc = sum_{k=0}^{K-1} new_cache[k] * w[k]
//   3. Optionally add bias
//   4. Apply silu(acc)
//   5. Emit K updated cache tiles to cb_new_cache, 1 output tile to cb_out
//
// DST slot layout (16 total on Wormhole, K≤4 so 2*4+2=10 needed):
//   0..K-1       : new_cache[k]  (shifted cache + new input)
//   K..2K-1      : w[k] * new_cache[k]  (per-lag products, accumulated into slot K)
//   2K           : bias / final output after silu
//
// CBs from reader:
//   c_0  cb_cache  — K old-cache tiles (k=0 oldest)
//   c_1  cb_xBC    — 1 new input tile
//   c_2  cb_w      — K weight tiles
//   c_3  cb_bias   — 1 bias tile (only when BIAS_VALID=1)
//
// CBs to writer:
//   c_16 cb_new_cache — K updated cache tiles
//   c_17 cb_out       — 1 silu output tile
//
// Compile-time defines:
//   BIAS_VALID  — 0 or 1
//   CONV_K      — kernel size (e.g. 4)
//
// Runtime args:
//   0: num_c — C-tiles for this core

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

using namespace ckernel;

void kernel_main() {
    uint32_t num_c = get_arg_val<uint32_t>(0);

    constexpr auto cb_cache     = tt::CBIndex::c_0;
    constexpr auto cb_xBC       = tt::CBIndex::c_1;
    constexpr auto cb_w         = tt::CBIndex::c_2;
    constexpr auto cb_bias      = tt::CBIndex::c_3;
    constexpr auto cb_new_cache = tt::CBIndex::c_16;
    constexpr auto cb_out       = tt::CBIndex::c_17;

    constexpr uint32_t K = CONV_K;

    // DST slot assignments (K=4, max slot used = 2*4+1 = 9, well within 16)
    // new_cache[k]: DST slot k           (k=0..K-1)
    // product[k]:   DST slot K+k         (k=0..K-1, reused for accumulation)
    // bias/out:     DST slot 2*K

    for (uint32_t ci = 0; ci < num_c; ++ci) {
        cb_wait_front(cb_cache, K);
        cb_wait_front(cb_xBC, 1);
        cb_wait_front(cb_w, K);
#if BIAS_VALID
        cb_wait_front(cb_bias, 1);
#endif

        cb_reserve_back(cb_new_cache, K);
        cb_reserve_back(cb_out, 1);

        // ── Phase 1 + 2: shift cache, load weights, compute dot product ───
        init_sfpu(cb_cache, cb_new_cache);
        tile_regs_acquire();

        // Load old cache[k+1] into DST[k] for k=0..K-2 (shift-left)
        copy_tile_to_dst_init_short(cb_cache);
        for (uint32_t k = 0; k < K - 1; ++k) {
            copy_tile(cb_cache, k + 1, k);      // DST[k] = cache[k+1]
        }
        // Load xBC_new into DST[K-1]
        copy_tile_to_dst_init_short(cb_xBC);
        copy_tile(cb_xBC, 0, K - 1);            // DST[K-1] = xBC_new

        // Load weights into DST[K..2K-1]
        copy_tile_to_dst_init_short(cb_w);
        for (uint32_t k = 0; k < K; ++k) {
            copy_tile(cb_w, k, K + k);          // DST[K+k] = w[k]
        }

        // Compute products: DST[K+k] = new_cache[k] * w[k]
        mul_binary_tile_init();
        for (uint32_t k = 0; k < K; ++k) {
            mul_binary_tile(k, K + k, K + k);   // DST[K+k] = DST[k] * DST[K+k]
        }

        // Accumulate: DST[K] += DST[K+k] for k=1..K-1
        add_binary_tile_init();
        for (uint32_t k = 1; k < K; ++k) {
            add_binary_tile(K, K + k, K);       // DST[K] += DST[K+k]
        }
        // DST[K] = dot(new_cache, w)

        // ── Phase 3: optionally add bias into DST[K] ──────────────────────
#if BIAS_VALID
        // Load bias into DST[2*K]
        copy_tile_to_dst_init_short(cb_bias);
        copy_tile(cb_bias, 0, 2 * K);
        add_binary_tile_init();
        add_binary_tile(K, 2 * K, K);          // DST[K] += bias
#endif

        // ── Phase 4: silu on DST[K] ───────────────────────────────────────
        silu_tile_init();
        silu_tile(K);                           // DST[K] = silu(DST[K])

        tile_regs_commit();
        tile_regs_wait();

        // ── Pack outputs ───────────────────────────────────────────────────
        // new_cache tiles: DST[0..K-1]
        for (uint32_t k = 0; k < K; ++k) {
            pack_tile(k, cb_new_cache);
        }
        // output tile: DST[K]
        pack_tile(K, cb_out);

        tile_regs_release();

        cb_push_back(cb_new_cache, K);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_cache, K);
        cb_pop_front(cb_xBC, 1);
        cb_pop_front(cb_w, K);
#if BIAS_VALID
        cb_pop_front(cb_bias, 1);
#endif
    }
}
