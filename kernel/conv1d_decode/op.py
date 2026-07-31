"""
Host-side launcher for conv1d_decode — fused Mamba2 causal conv1d decode step.

Computes in one kernel dispatch:
  new_cache[k] = cache[k+1]   k=0..K-2    (shift left)
  new_cache[K-1] = xBC_new                 (insert newest)
  out = silu(sum_k(new_cache[k] * w[k]) + bias)

Expected shapes (bfloat16, TILE_LAYOUT, on device):
  xBC_new_tt   : [1, 1, C, 1]   — new token's pre-conv values
  conv_cache_tt: [1, K, C, 1]   — shift-register cache (k=0 oldest, k=K-1 newest)
  conv_w_tt    : [1, K, C, 1]   — weight columns (k=0 oldest lag)
  conv_bias_tt : [1, 1, C, 1]   — optional bias (pass None to skip)

Returns:
  (new_cache_tt, conv_out_tt) — both same shapes as inputs; new_cache_tt written
  in-place to conv_cache_tt's buffer (via assign) for trace-safety.

Hot-path dispatch: plan cached after first call per (device, C, K).
"""

import math
import os
import torch
import ttnn

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

_plan_cache: dict = {}


class _Conv1dPlan:
    __slots__ = [
        'core_grid', 'core_list',
        'Ct', 'K',
        'bias_valid',
        'reader_ct', 'writer_ct',
        'cbs',
        'is_mesh', 'mesh_rows', 'mesh_cols',
        'reader_src', 'writer_src', 'compute_src',
    ]


def _build_plan(xBC_new_tt, conv_cache_tt, conv_w_tt, conv_bias_tt,
                device, new_cache_tt, conv_out_tt):
    sh = xBC_new_tt.shape   # [1, 1, C, 1]
    C = sh[2]
    K = conv_cache_tt.shape[1]

    Ct = math.ceil(C / 32)
    num_groups = Ct  # one work unit per C-tile

    _query_dev = device.get_device(0) if hasattr(device, "get_device") else device
    grid = _query_dev.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    all_cores_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    (_, core_grid, group1, group2, g1_work, g2_work) = ttnn.split_work_to_cores(
        all_cores_set, num_groups
    )

    core_list = []
    cur = 0
    for cg, wpc in [(group1, g1_work), (group2, g2_work)]:
        if wpc == 0:
            continue
        for cr in cg.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    core_list.append((cx, cy, wpc, cur))
                    cur += wpc

    is_mesh = hasattr(device, "get_num_devices")
    bias_valid = 1 if conv_bias_tt is not None else 0

    if is_mesh:
        xBC_d0     = ttnn.get_device_tensors(xBC_new_tt)[0]
        cache_d0   = ttnn.get_device_tensors(conv_cache_tt)[0]
        w_d0       = ttnn.get_device_tensors(conv_w_tt)[0]
        bias_d0    = ttnn.get_device_tensors(conv_bias_tt)[0] if bias_valid else xBC_d0
        ncache_d0  = ttnn.get_device_tensors(new_cache_tt)[0]
        out_d0     = ttnn.get_device_tensors(conv_out_tt)[0]
    else:
        xBC_d0 = xBC_new_tt; cache_d0 = conv_cache_tt; w_d0 = conv_w_tt
        bias_d0 = conv_bias_tt if bias_valid else xBC_new_tt
        ncache_d0 = new_cache_tt; out_d0 = conv_out_tt

    reader_ct = []
    reader_ct.extend(ttnn.TensorAccessorArgs(xBC_d0).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(cache_d0).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(w_d0).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(bias_d0).get_compile_time_args())

    writer_ct = []
    writer_ct.extend(ttnn.TensorAccessorArgs(ncache_d0).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(out_d0).get_compile_time_args())

    tile_bytes = 2 * 32 * 32

    def make_cb(idx, n_tiles=1):
        return ttnn.CBDescriptor(
            total_size=n_tiles * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=idx,
                data_format=ttnn.bfloat16,
                page_size=tile_bytes,
            )],
        )

    cbs = [
        make_cb(0, K),   # cb_cache
        make_cb(1, 1),   # cb_xBC
        make_cb(2, K),   # cb_w
        make_cb(3, 1),   # cb_bias
        make_cb(16, K),  # cb_new_cache
        make_cb(17, 1),  # cb_out
    ]

    plan = _Conv1dPlan()
    plan.core_grid   = core_grid
    plan.core_list   = core_list
    plan.Ct          = Ct
    plan.K           = K
    plan.bias_valid  = bias_valid
    plan.reader_ct   = reader_ct
    plan.writer_ct   = writer_ct
    plan.cbs         = cbs
    plan.is_mesh     = is_mesh
    plan.mesh_rows   = device.shape[0] if is_mesh else 1
    plan.mesh_cols   = device.shape[1] if is_mesh else 1
    plan.reader_src  = os.path.join(KERNEL_DIR, "dataflow", "reader.cpp")
    plan.writer_src  = os.path.join(KERNEL_DIR, "dataflow", "writer.cpp")
    plan.compute_src = os.path.join(KERNEL_DIR, "compute", "conv1d_decode.cpp")
    return plan


def _make_runtime_args(plan, xBC_addr, cache_addr, w_addr, bias_addr,
                       new_cache_addr, out_addr):
    reader_rt  = ttnn.RuntimeArgs()
    writer_rt  = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    Ct = plan.Ct
    K  = plan.K
    for cx, cy, wpc, cur in plan.core_list:
        reader_rt[cx][cy]  = [xBC_addr, cache_addr, w_addr, bias_addr,
                               wpc, K, Ct, cur, plan.bias_valid]
        writer_rt[cx][cy]  = [new_cache_addr, out_addr, wpc, K, Ct, cur]
        compute_rt[cx][cy] = [wpc]
    return reader_rt, writer_rt, compute_rt


def _make_program(plan, reader_rt, writer_rt, compute_rt):
    reader_kern = ttnn.KernelDescriptor(
        kernel_source=plan.reader_src,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=plan.core_grid,
        compile_time_args=plan.reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kern = ttnn.KernelDescriptor(
        kernel_source=plan.writer_src,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=plan.core_grid,
        compile_time_args=plan.writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kern = ttnn.KernelDescriptor(
        kernel_source=plan.compute_src,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=plan.core_grid,
        compile_time_args=[],
        defines=[
            ("BIAS_VALID", str(plan.bias_valid)),
            ("CONV_K",     str(plan.K)),
        ],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(
        kernels=[reader_kern, writer_kern, compute_kern],
        semaphores=[],
        cbs=plan.cbs,
    )


def conv1d_decode(xBC_new_tt, conv_cache_tt, conv_w_tt, conv_bias_tt,
                  device, new_cache_tt=None, conv_out_tt=None):
    """
    Args:
        xBC_new_tt   : [1, 1, C, 1]  bfloat16 tile-layout  — new token pre-conv
        conv_cache_tt: [1, K, C, 1]  bfloat16 tile-layout  — shift-register (fixed addr)
        conv_w_tt    : [1, K, C, 1]  bfloat16 tile-layout  — weights (constant)
        conv_bias_tt : [1, 1, C, 1]  bfloat16 tile-layout  — bias (or None)
        device       : single TT device or MeshDevice
        new_cache_tt : optional pre-allocated [1, K, C, 1] output (trace-safe)
        conv_out_tt  : optional pre-allocated [1, 1, C, 1] output (trace-safe)

    Returns:
        (new_cache_tt, conv_out_tt)
    """
    sh = xBC_new_tt.shape
    C = sh[2]
    K = conv_cache_tt.shape[1]

    if new_cache_tt is None:
        new_cache_tt = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, K, C, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
            device, ttnn.DRAM_MEMORY_CONFIG,
        )
    if conv_out_tt is None:
        conv_out_tt = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, C, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
            device, ttnn.DRAM_MEMORY_CONFIG,
        )

    plan_key = (id(device), C, K, conv_bias_tt is not None)
    if plan_key not in _plan_cache:
        _plan_cache[plan_key] = _build_plan(
            xBC_new_tt, conv_cache_tt, conv_w_tt, conv_bias_tt,
            device, new_cache_tt, conv_out_tt,
        )
    plan = _plan_cache[plan_key]

    # Use xBC address as dummy bias address when no bias (reader won't read it)
    bias_addr_fn = (lambda d: d.buffer_address()) if not plan.is_mesh else None

    if plan.is_mesh:
        xBC_devs    = ttnn.get_device_tensors(xBC_new_tt)
        cache_devs  = ttnn.get_device_tensors(conv_cache_tt)
        w_devs      = ttnn.get_device_tensors(conv_w_tt)
        bias_devs   = ttnn.get_device_tensors(conv_bias_tt) if plan.bias_valid else None
        ncache_devs = ttnn.get_device_tensors(new_cache_tt)
        out_devs    = ttnn.get_device_tensors(conv_out_tt)

        mesh_prog_desc = ttnn.MeshProgramDescriptor()
        for row in range(plan.mesh_rows):
            for col in range(plan.mesh_cols):
                di    = row * plan.mesh_cols + col
                coord = ttnn.MeshCoordinate(row, col)
                bias_addr = bias_devs[di].buffer_address() if plan.bias_valid \
                            else xBC_devs[di].buffer_address()
                r, w, c = _make_runtime_args(
                    plan,
                    xBC_devs[di].buffer_address(),
                    cache_devs[di].buffer_address(),
                    w_devs[di].buffer_address(),
                    bias_addr,
                    ncache_devs[di].buffer_address(),
                    out_devs[di].buffer_address(),
                )
                mesh_prog_desc[ttnn.MeshCoordinateRange(coord, coord)] = \
                    _make_program(plan, r, w, c)

        input_tensors = [xBC_new_tt, conv_cache_tt, conv_w_tt]
        if plan.bias_valid:
            input_tensors.append(conv_bias_tt)
        ttnn.generic_op(input_tensors + [new_cache_tt, conv_out_tt], mesh_prog_desc)
    else:
        bias_addr = conv_bias_tt.buffer_address() if plan.bias_valid \
                    else xBC_new_tt.buffer_address()
        r, w, c = _make_runtime_args(
            plan,
            xBC_new_tt.buffer_address(),
            conv_cache_tt.buffer_address(),
            conv_w_tt.buffer_address(),
            bias_addr,
            new_cache_tt.buffer_address(),
            conv_out_tt.buffer_address(),
        )
        input_tensors = [xBC_new_tt, conv_cache_tt, conv_w_tt]
        if plan.bias_valid:
            input_tensors.append(conv_bias_tt)
        ttnn.generic_op(
            input_tensors + [new_cache_tt, conv_out_tt],
            _make_program(plan, r, w, c),
        )

    return new_cache_tt, conv_out_tt
