#!/usr/bin/env python3
"""
Micro-benchmark: TTNN trace vs no-trace for decode sub-operations.
Identifies which operations are dispatch-overhead-bound vs hardware-bound.

Run: python test_trace.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tt-metal"))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

import torch
import ttnn
from kernel.ssm_update.op import ssm_update as _ssm_update_kernel

WARMUP = 5
ITERS  = 30


def bench(label, fn, device, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {'no-trace':8s} {label:45s}: {ms:.3f} ms")
    return ms


def bench_trace(label, capture_fn, device, warmup=WARMUP, iters=ITERS):
    """capture_fn() must return (tid, execute_fn)."""
    # warmup capture
    tid, execute_fn = capture_fn()
    ttnn.release_trace(device, tid)
    ttnn.synchronize_device(device)

    tid, execute_fn = capture_fn()
    for _ in range(warmup):
        execute_fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        execute_fn()
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / iters * 1000
    ttnn.release_trace(device, tid)
    print(f"  {'trace':8s} {label:45s}: {ms:.3f} ms  ({ms:.3f}ms)")
    return ms


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def run_all(device):
    B, H, D, N = 1, 48, 64, 128  # tiny model

    # ── Pre-allocate all tensors ──────────────────────────────────────────────
    x_hd    = to_tt(torch.randn(B, H, D,    dtype=torch.bfloat16), device)   # [1,48,64]
    dt_h1   = to_tt(torch.randn(B, H, 1,    dtype=torch.bfloat16), device)   # [1,48,1]
    dt_bias = to_tt(torch.randn(1, H, 1,    dtype=torch.bfloat16), device)
    A_tt    = to_tt(torch.randn(1, H, D, N, dtype=torch.bfloat16) * -1, device)
    dBx_tt  = to_tt(torch.randn(B, H, D, N, dtype=torch.bfloat16), device)
    dA_tt   = to_tt(torch.randn(B, H, D, N, dtype=torch.bfloat16) * -1, device)
    state   = to_tt(torch.randn(B, H, D, N, dtype=torch.bfloat16), device)
    C_4d    = to_tt(torch.randn(B, H, 1, N, dtype=torch.bfloat16), device)
    x_hd_4d = to_tt(torch.randn(B, H, D, 1, dtype=torch.bfloat16), device)
    B_hdn1  = to_tt(torch.randn(B, H, 1, N, dtype=torch.bfloat16), device)
    D_tt    = to_tt(torch.randn(1, H, D,    dtype=torch.bfloat16), device)

    # Tiny model MLP
    H_m, F_m = 1536, 512
    x_mlp = to_tt(torch.randn(1, 1, 1, H_m, dtype=torch.bfloat16), device)
    w1 = to_tt(torch.randn(H_m, F_m, dtype=torch.bfloat16), device)
    w3 = to_tt(torch.randn(H_m, F_m, dtype=torch.bfloat16), device)
    w2 = to_tt(torch.randn(F_m, H_m, dtype=torch.bfloat16), device)

    # In-proj for mamba decode: [1536 → 2*inter + conv_dim + H]
    inter, conv_dim = 1536, 1536 + 48  # approx
    H_in = 1536
    W_out = 2 * inter + conv_dim + H
    x_hidden = to_tt(torch.randn(1, 1, 1, H_in, dtype=torch.bfloat16), device)
    w_in = to_tt(torch.randn(H_in, W_out, dtype=torch.bfloat16), device)

    print("=" * 70)
    print(f"Tiny model dims: H={H}, D={D}, N={N}")
    print("=" * 70)

    results = {}

    # ── 1. dt preprocessing ───────────────────────────────────────────────────
    label = "dt: add+softplus+clip+unsqueeze+mul+exp"
    def dt_ops():
        d = ttnn.add(dt_h1, dt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d = ttnn.softplus(d, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d = ttnn.clip(d, 0.001, 0.1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d4 = ttnn.unsqueeze(d, -1)
        return ttnn.exp(ttnn.mul(d4, A_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG))

    t_no = bench(label, dt_ops, device)

    def dt_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        d = ttnn.add(dt_h1, dt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d = ttnn.softplus(d, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d = ttnn.clip(d, 0.001, 0.1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d4 = ttnn.unsqueeze(d, -1)
        ttnn.exp(ttnn.mul(d4, A_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, dt_capture, device)
    results["dt"] = (t_no, t_tr)

    # ── 2. dBx construction: dtx=dt*x, unsqueeze, mul B ──────────────────────
    label = "dBx: mul+unsqueeze+unsqueeze+mul"
    def dbx_ops():
        dtx = ttnn.mul(dt_h1, x_hd, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dtx4 = ttnn.unsqueeze(dtx, -1)
        B4   = ttnn.unsqueeze(B_hdn1, -2)
        return ttnn.mul(dtx4, B4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t_no = bench(label, dbx_ops, device)

    def dbx_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        dtx = ttnn.mul(dt_h1, x_hd, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dtx4 = ttnn.unsqueeze(dtx, -1)
        B4   = ttnn.unsqueeze(B_hdn1, -2)
        ttnn.mul(dtx4, B4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, dbx_capture, device)
    results["dBx"] = (t_no, t_tr)

    # ── 3. SSM state update + y reduction ────────────────────────────────────
    label = "SSM: addcmul+mul+sum  [1,48,64,128]"
    def ssm_ops():
        new_state = ttnn.addcmul(dBx_tt, dA_tt, state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        yc = ttnn.mul(new_state, C_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.sum(yc, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t_no = bench(label, ssm_ops, device)

    def ssm_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        new_state = ttnn.addcmul(dBx_tt, dA_tt, state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        yc = ttnn.mul(new_state, C_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.sum(yc, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, ssm_capture, device)
    results["ssm"] = (t_no, t_tr)

    # ── 3b. SSM fused kernel (ssm_update Metal kernel) ───────────────────────
    label = "SSM fused kernel  [1,48,64,128]"
    C_3d = to_tt(torch.randn(B, H, N, dtype=torch.bfloat16), device)
    hout_pre = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, H, D, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        device, ttnn.DRAM_MEMORY_CONFIG,
    )
    y_pre = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, H, D, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        device, ttnn.DRAM_MEMORY_CONFIG,
    )

    def ssm_fused_ops():
        _ssm_update_kernel(dBx_tt, dA_tt, state, C_3d, device,
                           hout_tt=hout_pre, y_tt=y_pre)

    t_no = bench(label, ssm_fused_ops, device)

    def ssm_fused_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _ssm_update_kernel(dBx_tt, dA_tt, state, C_3d, device,
                           hout_tt=hout_pre, y_tt=y_pre)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, ssm_fused_capture, device)
    results["ssm_fused"] = (t_no, t_tr)

    # ── 3c. Conv1d: old shift-register (10 dispatches) vs stacked (3 dispatches) ─
    C_conv = 1584  # tiny model conv_dim (inter + n_g*N = 1536 + 48)
    K_conv = 4
    conv_cache_cols = [
        to_tt(torch.zeros(1, 1, C_conv, 1, dtype=torch.bfloat16), device)
        for _ in range(K_conv)
    ]
    conv_w_cols = [
        to_tt(torch.randn(1, 1, C_conv, 1, dtype=torch.bfloat16), device)
        for _ in range(K_conv)
    ]
    xbc_col = to_tt(torch.randn(1, 1, C_conv, 1, dtype=torch.bfloat16), device)

    label = f"conv1d old shift-reg 10-dispatch  [1,1,{C_conv},1]"
    def conv_old():
        for i in range(K_conv - 1):
            ttnn.copy(conv_cache_cols[i + 1], conv_cache_cols[i])
        ttnn.copy(xbc_col, conv_cache_cols[K_conv - 1])
        out = None
        for k in range(K_conv):
            term = ttnn.mul(conv_w_cols[k], conv_cache_cols[k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if out is None:
                out = term
            else:
                new_out = ttnn.add(out, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out.deallocate(True); term.deallocate(True)
                out = new_out

    t_no = bench(label, conv_old, device)

    def conv_old_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for i in range(K_conv - 1):
            ttnn.copy(conv_cache_cols[i + 1], conv_cache_cols[i])
        ttnn.copy(xbc_col, conv_cache_cols[K_conv - 1])
        out = None
        for k in range(K_conv):
            term = ttnn.mul(conv_w_cols[k], conv_cache_cols[k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if out is None:
                out = term
            else:
                new_out = ttnn.add(out, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out.deallocate(True); term.deallocate(True)
                out = new_out
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, conv_old_capture, device)
    results["conv_old"] = (t_no, t_tr)

    conv_cache_stacked = to_tt(torch.zeros(1, 1, C_conv, K_conv, dtype=torch.bfloat16), device)
    conv_w_stacked = to_tt(torch.randn(1, 1, C_conv, K_conv, dtype=torch.bfloat16), device)

    label = f"conv1d new stacked  3-dispatch  [1,1,{C_conv},{K_conv}]"
    def conv_new():
        ttnn.copy(conv_cache_stacked[:, :, :, 1:], conv_cache_stacked[:, :, :, :K_conv - 1])
        ttnn.copy(xbc_col, conv_cache_stacked[:, :, :, K_conv - 1:])
        ttnn.sum(ttnn.mul(conv_cache_stacked, conv_w_stacked, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                 dim=3, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t_no = bench(label, conv_new, device)

    def conv_new_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.copy(conv_cache_stacked[:, :, :, 1:], conv_cache_stacked[:, :, :, :K_conv - 1])
        ttnn.copy(xbc_col, conv_cache_stacked[:, :, :, K_conv - 1:])
        ttnn.sum(ttnn.mul(conv_cache_stacked, conv_w_stacked, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                 dim=3, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, conv_new_capture, device)
    results["conv_new"] = (t_no, t_tr)

    # ── 4. D-skip + gate ──────────────────────────────────────────────────────
    label = "y-out: addcmul(D-skip)+silu+mul"
    y_hd = to_tt(torch.randn(B, H, D, dtype=torch.bfloat16), device)
    gate = to_tt(torch.randn(1, 1, 1, H_m, dtype=torch.bfloat16), device)

    def yout_ops():
        y = ttnn.addcmul(y_hd, D_tt, x_hd, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t_no = bench(label, yout_ops, device)

    def yout_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.addcmul(y_hd, D_tt, x_hd, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, yout_capture, device)
    results["yout"] = (t_no, t_tr)

    # ── 5. MLP forward ────────────────────────────────────────────────────────
    label = "MLP: linear+silu+mul+linear  [1536,512]"
    def mlp_ops():
        g = ttnn.silu(ttnn.linear(x_mlp, w1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        m = ttnn.mul(g, ttnn.linear(x_mlp, w3, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return ttnn.linear(m, w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t_no = bench(label, mlp_ops, device)

    def mlp_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        g = ttnn.silu(ttnn.linear(x_mlp, w1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        m = ttnn.mul(g, ttnn.linear(x_mlp, w3, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.linear(m, w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, mlp_capture, device)
    results["mlp"] = (t_no, t_tr)

    # ── 6. In-proj linear (large) ─────────────────────────────────────────────
    label = f"in-proj linear  [1,1,1,{H_in}] × [{H_in},{W_out}]"
    def inproj_ops():
        return ttnn.linear(x_hidden, w_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t_no = bench(label, inproj_ops, device)

    def inproj_capture():
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.linear(x_hidden, w_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)

    t_tr = bench_trace(label, inproj_capture, device)
    results["inproj"] = (t_no, t_tr)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"{'Component':<42} {'No-trace':>9} {'Trace':>7} {'Speedup':>8} {'Save/layer':>10}")
    print("-" * 70)
    for k, (nt, tr) in results.items():
        save = nt - tr
        print(f"  {k:<40} {nt:>8.3f}ms {tr:>6.3f}ms {nt/tr:>7.1f}x  {save:>8.3f}ms")
    print()
    total_no  = sum(v[0] for v in results.values())
    total_tr  = sum(v[1] for v in results.values())
    print(f"  {'Total (these ops only)':<40} {total_no:>8.3f}ms {total_tr:>6.3f}ms {total_no/total_tr:>7.1f}x  {total_no-total_tr:>8.3f}ms")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        run_all(device)
    finally:
        ttnn.close_device(device)
