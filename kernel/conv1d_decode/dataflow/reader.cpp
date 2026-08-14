//
// Reader for conv1d_decode kernel.
//
// Tensors (all [1, Ct] logical tiles, bfloat16 TILE_LAYOUT):
//   xBC_new    [1, 1, C, 1]  — new input column, Ct tiles (tile index c)
//   conv_cache [1, K, C, 1]  — shift register (K cols × Ct tiles each)
//                               k=0 oldest, k=K-1 newest-before-this-step
//   conv_w     [1, K, C, 1]  — weight columns (k=0 oldest lag, k=K-1 newest lag)
//   conv_bias  [1, 1, C, 1]  — optional bias, Ct tiles (bias_valid=1 enables)
//
// Per-core work: a contiguous range of C-tiles [start_c, start_c + num_c).
// For each c-tile:
//   Send: K cache tiles (k=0..K-1), new input tile, K weight tiles, 1 bias tile
//
// Runtime args:
//   0: xBC_new_addr
//   1: conv_cache_addr
//   2: conv_w_addr
//   3: conv_bias_addr    (ignored when bias_valid=0)
//   4: num_c             — number of C-tiles for this core
//   5: K                 — conv kernel size (compile-time constant duplicate, for loop bound)
//   6: Ct               — total C-tiles (for tile index computation)
//   7: start_c           — first C-tile index for this core
//   8: bias_valid        — 1 if bias present, 0 otherwise

#include <cstdint>
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t xBC_addr   = get_arg_val<uint32_t>(0);
    uint32_t cache_addr = get_arg_val<uint32_t>(1);
    uint32_t w_addr     = get_arg_val<uint32_t>(2);
    uint32_t bias_addr  = get_arg_val<uint32_t>(3);
    uint32_t num_c      = get_arg_val<uint32_t>(4);
    uint32_t K          = get_arg_val<uint32_t>(5);
    uint32_t Ct         = get_arg_val<uint32_t>(6);
    uint32_t start_c    = get_arg_val<uint32_t>(7);
    uint32_t bias_valid = get_arg_val<uint32_t>(8);

    // CBs
    constexpr auto cb_cache  = tt::CBIndex::c_0;  // K tiles (old cache cols, oldest first)
    constexpr auto cb_xBC    = tt::CBIndex::c_1;  // 1 tile  (new input)
    constexpr auto cb_w      = tt::CBIndex::c_2;  // K tiles (weights)
    constexpr auto cb_bias   = tt::CBIndex::c_3;  // 1 tile  (bias, may be zeros)

    const uint32_t tile_bytes = get_tile_size(cb_xBC);

    constexpr auto xBC_args   = TensorAccessorArgs<0>();
    const auto xBC_acc   = TensorAccessor(xBC_args,   xBC_addr,   tile_bytes);
    constexpr auto cache_args = TensorAccessorArgs<xBC_args.next_compile_time_args_offset()>();
    const auto cache_acc = TensorAccessor(cache_args, cache_addr, tile_bytes);
    constexpr auto w_args     = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    const auto w_acc     = TensorAccessor(w_args,     w_addr,     tile_bytes);
    constexpr auto bias_args  = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto bias_acc  = TensorAccessor(bias_args,  bias_addr,  tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer buf_cache(cb_cache);
    experimental::CircularBuffer buf_xBC(cb_xBC);
    experimental::CircularBuffer buf_w(cb_w);
    experimental::CircularBuffer buf_bias(cb_bias);

    for (uint32_t ci = 0; ci < num_c; ++ci) {
        uint32_t c = start_c + ci;

        // Send K old cache tiles (k=0 oldest .. k=K-1 newest-before-update)
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t cache_tile = k * Ct + c;
            buf_cache.reserve_back(1);
            noc.async_read(cache_acc, buf_cache, tile_bytes, {.page_id = cache_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            buf_cache.push_back(1);
        }

        // Send new input tile
        buf_xBC.reserve_back(1);
        noc.async_read(xBC_acc, buf_xBC, tile_bytes, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        buf_xBC.push_back(1);

        // Send K weight tiles (k=0..K-1)
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t w_tile = k * Ct + c;
            buf_w.reserve_back(1);
            noc.async_read(w_acc, buf_w, tile_bytes, {.page_id = w_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            buf_w.push_back(1);
        }

        // Send bias tile (or zeros if bias_valid=0 — compute kernel handles it)
        if (bias_valid) {
            buf_bias.reserve_back(1);
            noc.async_read(bias_acc, buf_bias, tile_bytes, {.page_id = c}, {.offset_bytes = 0});
            noc.async_read_barrier();
            buf_bias.push_back(1);
        }
    }
}
