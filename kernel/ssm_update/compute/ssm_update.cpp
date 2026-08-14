//
// Compute kernel: ssm_update
//
// Per group (b, h, d_tile), for each N-tile n:
//   h_out[d,n] = dBx[d,n] + dA[d,n] * state[d,n]
//   yc[d,n]    = h_out[d,n] * C[h%32, n]    (row h%32 of C tile broadcast over D)
//   y[d]       = sum_n(yc[d,n])              (REDUCE_ROW over Nt tiles)
//
// Three phases per group:
//   Phase 1 (SFPU)  : compute h_out, pack to cb_hout (writer) and cb_hout_s (staging)
//   Phase 2 (BCAST) : yc = h_out_s * C via mul_tiles_bcast_rows, pack to cb_yc
//   Phase 3 (REDUCE): y  = reduce_row(yc) over Nt tiles, pack to cb_y
//
// CBs:
//   c_0  dBx    [B,H,D,N]  input — 1 tile
//   c_1  dA     [B,H,D,N]  input — 1 tile
//   c_2  state  [B,H,D,N]  input — 1 tile
//   c_3  C      [B,H,N]    input — 1 tile (covers 32 H-rows; h%32 selects broadcast row)
//   c_4  scaler constant all-ones 32×32 tile for reduce
//   c_5  hout_s [Nt tiles] staging h_out for Phase 2
//   c_6  yc     [Nt tiles] staging h_out*C for Phase 3
//   c_16 hout   [B,H,D,N]  output — 1 tile
//   c_17 y      [B,H,D,1]  output — 1 tile per group
//
// Runtime args:
//   0: num_groups   — groups for this core
//   1: Nt           — N_pad / 32
//   2: Dt           — D_pad / 32
//   3: start_group  — global group offset

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"

using namespace ckernel;

void kernel_main()
{
    uint32_t num_groups = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_arg_val<uint32_t>(1);
    uint32_t Dt = get_arg_val<uint32_t>(2);
    uint32_t start_group = get_arg_val<uint32_t>(3);

    constexpr auto cb_dBx = tt::CBIndex::c_0;
    constexpr auto cb_dA = tt::CBIndex::c_1;
    constexpr auto cb_state = tt::CBIndex::c_2;
    constexpr auto cb_C = tt::CBIndex::c_3;
    constexpr auto cb_scaler = tt::CBIndex::c_4;
    constexpr auto cb_hout_s = tt::CBIndex::c_5;
    constexpr auto cb_yc = tt::CBIndex::c_6;
    constexpr auto cb_hout = tt::CBIndex::c_16;
    constexpr auto cb_y = tt::CBIndex::c_17;

    cb_wait_front(cb_scaler, 1);

    for (uint32_t g = 0; g < num_groups; ++g)
    {
        uint32_t grp = start_group + g;
        uint32_t h = grp / Dt;
        uint32_t h_row = h & 31u;

        // Phase 1: h_out = dBx + dA * state
        // Pack h_out to both cb_hout (writer output) and cb_hout_s (staging for Phase 2).
        init_sfpu(cb_dBx, cb_hout);

        for (uint32_t n = 0; n < Nt; ++n)
        {
            cb_wait_front(cb_dBx, 1);
            cb_wait_front(cb_dA, 1);
            cb_wait_front(cb_state, 1);
            cb_reserve_back(cb_hout, 1);
            cb_reserve_back(cb_hout_s, 1);

            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_dBx);
            copy_tile(cb_dBx, 0, 0);
            copy_tile_to_dst_init_short(cb_dA);
            copy_tile(cb_dA, 0, 1);
            copy_tile_to_dst_init_short(cb_state);
            copy_tile(cb_state, 0, 2);

            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1); // DST[1] = dA * state
            add_binary_tile_init();
            add_binary_tile(0, 1, 0); // DST[0] = h_out

            tile_regs_commit();
            tile_regs_wait();

            // Pack h_out to output CB and staging CB
            pack_tile(0, cb_hout);
            cb_push_back(cb_hout, 1);
            pack_tile(0, cb_hout_s);
            cb_push_back(cb_hout_s, 1);

            tile_regs_release();

            cb_pop_front(cb_dBx, 1);
            cb_pop_front(cb_dA, 1);
            cb_pop_front(cb_state, 1);
        }

        // Phase 2: yc = h_out * C[h_row, :] via row-broadcast multiply
        init_bcast<ELWMUL, BroadcastType::ROW>(cb_hout_s, cb_C, cb_yc);

        for (uint32_t n = 0; n < Nt; ++n)
        {
            cb_wait_front(cb_hout_s, 1);
            cb_wait_front(cb_C, 1);
            cb_reserve_back(cb_yc, 1);

            acquire_dst();
            mul_tiles_bcast_rows(cb_hout_s, cb_C, 0, 0, 0, h_row);
            pack_tile(0, cb_yc);
            cb_push_back(cb_yc, 1);
            release_dst();

            cb_pop_front(cb_hout_s, 1);
            cb_pop_front(cb_C, 1);
        }

        // Phase 3: y = sum_n(yc) via REDUCE_ROW
        cb_reserve_back(cb_y, 1);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_yc, cb_scaler, cb_y);
        acquire_dst();

        for (uint32_t n = 0; n < Nt; ++n)
        {
            cb_wait_front(cb_yc, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_yc, cb_scaler, 0, 0, 0);
            cb_pop_front(cb_yc, 1);
        }

        pack_tile(0, cb_y);
        cb_push_back(cb_y, 1);
        release_dst();
        reduce_uninit(cb_yc);
    }

    // Consume the scaler tile so CB4 semaphore returns to 0 — required so
    // reserve_back in the reader succeeds on the next kernel invocation.
    cb_pop_front(cb_scaler, 1);
}
