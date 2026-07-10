// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer for ssm_update kernel.
//
// Writes h_out (Nt tiles per group) and y (1 tile per group, containing
// the row-reduced sums for 32 D-positions, stored in the REDUCE_ROW packing format).
//
// h_out tile index: grp * Nt + n
// y tile index:     grp          (1 tile per group — same as reduce_w output layout)
//
// Runtime args:
//   0: hout_addr
//   1: y_addr
//   2: num_groups
//   3: Nt
//   4: start_group

#include <cstdint>
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t hout_addr   = get_arg_val<uint32_t>(0);
    uint32_t y_addr      = get_arg_val<uint32_t>(1);
    uint32_t num_groups  = get_arg_val<uint32_t>(2);
    uint32_t Nt          = get_arg_val<uint32_t>(3);
    uint32_t start_group = get_arg_val<uint32_t>(4);

    constexpr auto cb_hout = tt::CBIndex::c_16;
    constexpr auto cb_y    = tt::CBIndex::c_17;

    const uint32_t tile_bytes = get_tile_size(cb_hout);

    constexpr auto hout_args = TensorAccessorArgs<0>();
    const auto hout_acc = TensorAccessor(hout_args, hout_addr, tile_bytes);
    constexpr auto y_args = TensorAccessorArgs<hout_args.next_compile_time_args_offset()>();
    const auto y_acc    = TensorAccessor(y_args,    y_addr, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer buf_hout(cb_hout);
    experimental::CircularBuffer buf_y(cb_y);

    for (uint32_t g = 0; g < num_groups; ++g) {
        uint32_t grp = start_group + g;

        // h_out: Nt tiles per group
        for (uint32_t n = 0; n < Nt; ++n) {
            buf_hout.wait_front(1);
            noc.async_write(buf_hout, hout_acc, tile_bytes, {}, {.page_id = grp * Nt + n});
            noc.async_write_barrier();
            buf_hout.pop_front(1);
        }

        // y: 1 tile per group
        buf_y.wait_front(1);
        noc.async_write(buf_y, y_acc, tile_bytes, {}, {.page_id = grp});
        noc.async_write_barrier();
        buf_y.pop_front(1);
    }
}
