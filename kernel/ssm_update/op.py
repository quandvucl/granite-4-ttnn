"""Host-side launcher for the ssm_update kernel (fused Mamba2 SSM decode step)."""

import math
import os

import torch
import ttnn

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

_scaler_cache: dict = {}
_plan_cache: dict = {}


def _get_scaler(device):
    """All-ones bfloat16 single-tile on device, cached per device."""
    key = id(device)
    if key not in _scaler_cache:
        ones = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
        mapper = (
            ttnn.ReplicateTensorToMesh(device)
            if hasattr(device, "get_num_devices")
            else None
        )
        _scaler_cache[key] = ttnn.from_torch(
            ones,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
    return _scaler_cache[key]


class _SsmPlan:
    """Cached static dispatch data for one (device, B, H, D, N, cache_bust_id) combination.

    core_list: [(cx, cy, wpc, cur), ...] - pre-computed per-core static args.
               Only addresses change per call; these are constant.
    """

    __slots__ = [
        "core_grid",
        "core_list",
        "Nt",
        "Dt",
        "reader_ct",
        "writer_ct",
        "cbs",
        "is_mesh",
        "mesh_rows",
        "mesh_cols",
        "scaler_tt",
        "reader_src",
        "writer_src",
        "compute_src",
    ]


def _build_plan(
    dBx_tt, dA_tt, state_tt, C_tt, device, hout_tt, y_tt, scaler_tt, cache_bust_id
):
    """Build static dispatch plan: grid split, CB descriptors, compile-time args."""
    sh = dBx_tt.shape
    B, H, D, N = sh[0], sh[1], sh[2], sh[3]

    D_pad = math.ceil(D / 32) * 32
    N_pad = math.ceil(N / 32) * 32
    Dt = D_pad // 32
    Nt = N_pad // 32
    num_groups = B * H * Dt

    _query_dev = device.get_device(0) if hasattr(device, "get_device") else device
    grid = _query_dev.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    all_cores_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    _, core_grid, group1, group2, g1_work, g2_work = ttnn.split_work_to_cores(
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

    if is_mesh:
        dBx_d0 = ttnn.get_device_tensors(dBx_tt)[0]
        state_d0 = ttnn.get_device_tensors(state_tt)[0]
        C_d0 = ttnn.get_device_tensors(C_tt)[0]
        sc_d0 = ttnn.get_device_tensors(scaler_tt)[0]
        hout_d0 = ttnn.get_device_tensors(hout_tt)[0]
        y_d0 = ttnn.get_device_tensors(y_tt)[0]
    else:
        dBx_d0 = dBx_tt
        state_d0 = state_tt
        C_d0 = C_tt
        sc_d0 = scaler_tt
        hout_d0 = hout_tt
        y_d0 = y_tt

    reader_ct = []
    reader_ct.extend(ttnn.TensorAccessorArgs(dBx_d0).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(dBx_d0).get_compile_time_args()
    )  # dA same layout
    reader_ct.extend(ttnn.TensorAccessorArgs(state_d0).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(C_d0).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(sc_d0).get_compile_time_args())

    writer_ct = []
    writer_ct.extend(ttnn.TensorAccessorArgs(hout_d0).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(y_d0).get_compile_time_args())

    tile_bytes = 2 * 32 * 32

    def make_cb(idx, n_tiles=1):
        return ttnn.CBDescriptor(
            total_size=n_tiles * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=idx,
                    data_format=ttnn.bfloat16,
                    page_size=tile_bytes,
                )
            ],
        )

    cbs = [
        make_cb(0),
        make_cb(1),
        make_cb(2),
        make_cb(3),
        make_cb(4),
        make_cb(5, Nt),
        make_cb(6, Nt),
        make_cb(16),
        make_cb(17),
        # CB7 size encodes cache_bust_id so each Mamba layer gets a unique program hash,
        # preventing cross-layer generic_op cache collisions with stale L1 CB addresses.
        ttnn.CBDescriptor(
            total_size=(cache_bust_id + 1) * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=7,
                    data_format=ttnn.bfloat16,
                    page_size=tile_bytes,
                )
            ],
        ),
    ]

    plan = _SsmPlan()
    plan.core_grid = core_grid
    plan.core_list = core_list
    plan.Nt = Nt
    plan.Dt = Dt
    plan.reader_ct = reader_ct
    plan.writer_ct = writer_ct
    plan.cbs = cbs
    plan.is_mesh = is_mesh
    plan.mesh_rows = device.shape[0] if is_mesh else 1
    plan.mesh_cols = device.shape[1] if is_mesh else 1
    plan.scaler_tt = scaler_tt
    plan.reader_src = os.path.join(KERNEL_DIR, "dataflow", "reader.cpp")
    plan.writer_src = os.path.join(KERNEL_DIR, "dataflow", "writer.cpp")
    plan.compute_src = os.path.join(KERNEL_DIR, "compute", "ssm_update.cpp")
    return plan


def _make_runtime_args(
    plan, dBx_addr, dA_addr, state_addr, C_addr, scaler_addr, hout_addr, y_addr
):
    """Fill per-core address slots into RuntimeArgs structures."""
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    Nt = plan.Nt
    Dt = plan.Dt
    for cx, cy, wpc, cur in plan.core_list:
        reader_rt[cx][cy] = [
            dBx_addr,
            dA_addr,
            state_addr,
            C_addr,
            scaler_addr,
            wpc,
            Nt,
            Dt,
            cur,
        ]
        writer_rt[cx][cy] = [hout_addr, y_addr, wpc, Nt, cur]
        compute_rt[cx][cy] = [wpc, Nt, Dt, cur]
    return reader_rt, writer_rt, compute_rt


def _make_program(plan, reader_rt, writer_rt, compute_rt):
    """Assemble a ProgramDescriptor from the cached plan and fresh runtime args."""
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
            ("REDUCE_OP", "PoolType::SUM"),
            ("REDUCE_DIM", "ReduceDim::REDUCE_ROW"),
        ],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(
        kernels=[reader_kern, writer_kern, compute_kern],
        semaphores=[],
        cbs=plan.cbs,
    )


def ssm_update(
    dBx_tt, dA_tt, state_tt, C_tt, device, hout_tt=None, y_tt=None, cache_bust_id=0
):
    """
    Args:
        dBx_tt, dA_tt, state_tt : [1, H, D, N]  bfloat16 tile-layout
        C_tt                     : [1, H, N]     bfloat16 tile-layout
        device                   : single TT device or MeshDevice
        hout_tt                  : optional pre-allocated [1, H, D, N] output (trace-safe)
        y_tt                     : optional pre-allocated [1, H, D, 1] output (trace-safe)
        cache_bust_id            : per-layer integer that differentiates the generic_op
                                   program hash across Mamba layers, preventing cross-layer
                                   cache collisions with stale L1 CB addresses.

    Returns:
        (hout_tt, y_tt) where
          hout_tt : [1, H, D, N]
          y_tt    : [1, H, D, 1]  (reduce_w output format, W padded to 32)
    """
    sh = dBx_tt.shape
    B, H, D, N = sh[0], sh[1], sh[2], sh[3]

    scaler_tt = _get_scaler(device)

    if hout_tt is None:
        hout_tt = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, H, D, N]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
    if y_tt is None:
        y_tt = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, H, D, 1]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    plan_key = (id(device), B, H, D, N, cache_bust_id)
    if plan_key not in _plan_cache:
        _plan_cache[plan_key] = _build_plan(
            dBx_tt,
            dA_tt,
            state_tt,
            C_tt,
            device,
            hout_tt,
            y_tt,
            scaler_tt,
            cache_bust_id,
        )
    plan = _plan_cache[plan_key]

    if plan.is_mesh:
        dBx_devs = ttnn.get_device_tensors(dBx_tt)
        dA_devs = ttnn.get_device_tensors(dA_tt)
        state_devs = ttnn.get_device_tensors(state_tt)
        C_devs = ttnn.get_device_tensors(C_tt)
        scaler_devs = ttnn.get_device_tensors(scaler_tt)
        hout_devs = ttnn.get_device_tensors(hout_tt)
        y_devs = ttnn.get_device_tensors(y_tt)

        mesh_prog_desc = ttnn.MeshProgramDescriptor()
        for row in range(plan.mesh_rows):
            for col in range(plan.mesh_cols):
                di = row * plan.mesh_cols + col
                coord = ttnn.MeshCoordinate(row, col)
                r, w, c = _make_runtime_args(
                    plan,
                    dBx_devs[di].buffer_address(),
                    dA_devs[di].buffer_address(),
                    state_devs[di].buffer_address(),
                    C_devs[di].buffer_address(),
                    scaler_devs[di].buffer_address(),
                    hout_devs[di].buffer_address(),
                    y_devs[di].buffer_address(),
                )
                mesh_prog_desc[ttnn.MeshCoordinateRange(coord, coord)] = _make_program(
                    plan, r, w, c
                )

        ttnn.generic_op(
            [dBx_tt, dA_tt, state_tt, C_tt, scaler_tt, hout_tt, y_tt], mesh_prog_desc
        )
    else:
        r, w, c = _make_runtime_args(
            plan,
            dBx_tt.buffer_address(),
            dA_tt.buffer_address(),
            state_tt.buffer_address(),
            C_tt.buffer_address(),
            scaler_tt.buffer_address(),
            hout_tt.buffer_address(),
            y_tt.buffer_address(),
        )
        ttnn.generic_op(
            [dBx_tt, dA_tt, state_tt, C_tt, scaler_tt, hout_tt, y_tt],
            _make_program(plan, r, w, c),
        )

    return hout_tt, y_tt
