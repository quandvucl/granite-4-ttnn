//
// Writer for conv1d_decode kernel.
//
// Writes:
//   new_cache [1, K, C, 1] — updated shift register (k=0 oldest, k=K-1 = new input)
//   conv_out  [1, 1, C, 1] — silu(dot(cache, w) + bias)
//
// Per core, for each C-tile [start_c, start_c+num_c):
//   Receive K updated cache tiles, then 1 output tile.
//
// Runtime args:
//   0: new_cache_addr
//   1: conv_out_addr
//   2: num_c
//   3: K
//   4: Ct
//   5: start_c

#include <cstdint>
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t new_cache_addr = get_arg_val<uint32_t>(0);
    uint32_t conv_out_addr  = get_arg_val<uint32_t>(1);
    uint32_t num_c          = get_arg_val<uint32_t>(2);
    uint32_t K              = get_arg_val<uint32_t>(3);
    uint32_t Ct             = get_arg_val<uint32_t>(4);
    uint32_t start_c        = get_arg_val<uint32_t>(5);

    constexpr auto cb_new_cache = tt::CBIndex::c_16;  // K tiles updated cache
    constexpr auto cb_out       = tt::CBIndex::c_17;  // 1 tile conv output

    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto cache_args = TensorAccessorArgs<0>();
    const auto cache_acc = TensorAccessor(cache_args, new_cache_addr, tile_bytes);
    constexpr auto out_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    const auto out_acc   = TensorAccessor(out_args,   conv_out_addr,  tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer buf_new_cache(cb_new_cache);
    experimental::CircularBuffer buf_out(cb_out);

    for (uint32_t ci = 0; ci < num_c; ++ci) {
        uint32_t c = start_c + ci;

        // Write K updated cache tiles
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t cache_tile = k * Ct + c;
            buf_new_cache.wait_front(1);
            noc.async_write(buf_new_cache, cache_acc, tile_bytes, {}, {.page_id = cache_tile});
            noc.async_write_barrier();
            buf_new_cache.pop_front(1);
        }

        // Write 1 output tile
        buf_out.wait_front(1);
        noc.async_write(buf_out, out_acc, tile_bytes, {}, {.page_id = c});
        noc.async_write_barrier();
        buf_out.pop_front(1);
    }
}
