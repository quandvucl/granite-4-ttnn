// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for ssm_update kernel.
//
// Reads dBx, dA, state (all [B,H,D,N]) and C ([B,H,N]) tiles, plus one all-ones scaler tile.
//
// C tile index:  (h / 32) * Nt + n   where h = grp / Dt
// 4D tile index: grp * Nt + n
//
// The reader sends C tiles after dBx/dA/state for each group, in N-tile order.
// Compute Phase 1 reads dBx/dA/state immediately.
// Compute Phase 2 reads C (after h_out staging is ready).
//
// Runtime args:
//   0: dBx_addr
//   1: dA_addr
//   2: state_addr
//   3: C_addr
//   4: scaler_addr  (pre-built all-ones bfloat16 tile)
//   5: num_groups
//   6: Nt           — N_pad / 32
//   7: Dt           — D_pad / 32
//   8: start_group

#include <cstdint>
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dBx_addr    = get_arg_val<uint32_t>(0);
    uint32_t dA_addr     = get_arg_val<uint32_t>(1);
    uint32_t state_addr  = get_arg_val<uint32_t>(2);
    uint32_t C_addr      = get_arg_val<uint32_t>(3);
    uint32_t scaler_addr = get_arg_val<uint32_t>(4);
    uint32_t num_groups  = get_arg_val<uint32_t>(5);
    uint32_t Nt          = get_arg_val<uint32_t>(6);
    uint32_t Dt          = get_arg_val<uint32_t>(7);
    uint32_t start_group = get_arg_val<uint32_t>(8);

    constexpr auto cb_dBx    = tt::CBIndex::c_0;
    constexpr auto cb_dA     = tt::CBIndex::c_1;
    constexpr auto cb_state  = tt::CBIndex::c_2;
    constexpr auto cb_C      = tt::CBIndex::c_3;
    constexpr auto cb_scaler = tt::CBIndex::c_4;

    const uint32_t tile_bytes = get_tile_size(cb_dBx);

    constexpr auto dBx_args   = TensorAccessorArgs<0>();
    const auto dBx_acc   = TensorAccessor(dBx_args,   dBx_addr,   tile_bytes);
    constexpr auto dA_args    = TensorAccessorArgs<dBx_args.next_compile_time_args_offset()>();
    const auto dA_acc    = TensorAccessor(dA_args,    dA_addr,    tile_bytes);
    constexpr auto state_args = TensorAccessorArgs<dA_args.next_compile_time_args_offset()>();
    const auto state_acc = TensorAccessor(state_args, state_addr, tile_bytes);
    constexpr auto C_args     = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    const auto C_acc     = TensorAccessor(C_args,     C_addr,     tile_bytes);
    constexpr auto sc_args    = TensorAccessorArgs<C_args.next_compile_time_args_offset()>();
    const auto sc_acc    = TensorAccessor(sc_args,    scaler_addr, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer buf_dBx(cb_dBx);
    experimental::CircularBuffer buf_dA(cb_dA);
    experimental::CircularBuffer buf_state(cb_state);
    experimental::CircularBuffer buf_C(cb_C);
    experimental::CircularBuffer buf_scaler(cb_scaler);

    // Push constant scaler tile once
    buf_scaler.reserve_back(1);
    noc.async_read(sc_acc, buf_scaler, tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    buf_scaler.push_back(1);

    for (uint32_t g = 0; g < num_groups; ++g) {
        uint32_t grp      = start_group + g;
        uint32_t h        = grp / Dt;            // h index in padded H
        uint32_t C_h_tile = h >> 5u;             // h / 32 — which C H-tile row
        uint32_t base_4d  = grp * Nt;
        uint32_t base_C   = C_h_tile * Nt;

        // Phase 1 tiles: dBx, dA, state (compute reads these first)
        for (uint32_t n = 0; n < Nt; ++n) {
            uint32_t tile_4d = base_4d + n;

            buf_dBx.reserve_back(1);
            buf_dA.reserve_back(1);
            buf_state.reserve_back(1);

            noc.async_read(dBx_acc,   buf_dBx,   tile_bytes, {.page_id = tile_4d}, {.offset_bytes = 0});
            noc.async_read(dA_acc,    buf_dA,    tile_bytes, {.page_id = tile_4d}, {.offset_bytes = 0});
            noc.async_read(state_acc, buf_state, tile_bytes, {.page_id = tile_4d}, {.offset_bytes = 0});
            noc.async_read_barrier();

            buf_dBx.push_back(1);
            buf_dA.push_back(1);
            buf_state.push_back(1);
        }

        // Phase 2 tiles: C (compute reads these after Phase 1 h_out is staged)
        for (uint32_t n = 0; n < Nt; ++n) {
            uint32_t tile_C = base_C + n;

            buf_C.reserve_back(1);
            noc.async_read(C_acc, buf_C, tile_bytes, {.page_id = tile_C}, {.offset_bytes = 0});
            noc.async_read_barrier();
            buf_C.push_back(1);
        }
    }
}
