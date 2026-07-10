"""
Fix: allocate each layer's mutable buffers AFTER the previous layer's trace is captured.
This ensures each layer's buffers get addresses above all prior trace intermediates.
"""
import os, sys, torch, ttnn
sys.path.insert(0, "/work/tt-metal"); sys.path.insert(0, "/work/tt-granite")
os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto")

def tt(t, device, mapper):
    return ttnn.from_torch(t.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mapper)

def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=268435456)
    device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]
    mapper = ttnn.ReplicateTensorToMesh(device)

    H, D = 48, 64
    hidden = 3072
    inter = 3072
    conv_dim = 3328
    n_heads = 48
    N = 128
    n_g = 1
    proj_full = inter + conv_dim + n_heads
    proj_per_dev = proj_full // 4
    K = 4

    # Shared read-only weights (allocated first)
    w_inproj = tt(torch.randn(hidden, proj_per_dev) * 0.01, device, mapper)
    w_out    = tt(torch.randn(inter, hidden) * 0.01, device, mapper)
    A        = tt(-torch.rand(H, D, N), device, mapper)
    dt_bias  = tt(torch.zeros(H, 1), device, mapper)
    D_tt     = tt(torch.ones(H, D), device, mapper)

    def trace_body(trace_in, state_in):
        gate_tt     = trace_in[:, :, :, :inter]
        conv_out_tt = trace_in[:, :, :, inter:inter + conv_dim]
        dt_tt       = trace_in[:, :, :, inter + conv_dim:]
        conv_out_4d = ttnn.reshape(conv_out_tt, [1, 1, conv_dim, 1])
        conv_out_tt.deallocate(False)
        x_tt     = conv_out_4d[:, :, :inter, :]
        B_raw_tt = conv_out_4d[:, :, inter:inter + n_g * N, :]
        C_raw_tt = conv_out_4d[:, :, inter + n_g * N:, :]
        conv_out_4d.deallocate(False)
        x_tt = ttnn.reshape(x_tt, [1, H, D])
        B_tt = ttnn.reshape(B_raw_tt, [1, n_g, N])
        C_tt = ttnn.reshape(C_raw_tt, [1, n_g, N])
        B_raw_tt.deallocate(False); C_raw_tt.deallocate(False)
        dt_tt = ttnn.reshape(dt_tt, [1, H, 1])
        dt_tt = ttnn.add(dt_tt, dt_bias)
        dt_tt = ttnn.softplus(dt_tt, beta=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_exp = ttnn.unsqueeze(dt_tt, -1)
        dA_tt = ttnn.exp(ttnn.mul(dt_exp, A))
        dt_exp.deallocate(True)
        dtx_tt = ttnn.mul(dt_tt, x_tt)
        dt_tt.deallocate(True)
        dtx_tt = ttnn.unsqueeze(dtx_tt, -1)
        B_tt = ttnn.unsqueeze(B_tt, -2)
        dBx_tt = ttnn.mul(dtx_tt, B_tt)
        dtx_tt.deallocate(True); B_tt.deallocate(True)
        C_tt = ttnn.unsqueeze(C_tt, -2)
        new_state = ttnn.addcmul(dBx_tt, dA_tt, state_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dA_tt.deallocate(True); dBx_tt.deallocate(True)
        y_unred = ttnn.mul(new_state, C_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        C_tt.deallocate(True)
        y_tt = ttnn.sum(y_unred, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y_unred.deallocate(True)
        y_tt = ttnn.addcmul(y_tt, D_tt, x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_tt.deallocate(False)
        y3d = ttnn.reshape(y_tt, [1, 1, H * D])
        y_tt.deallocate(True)
        gate_tt = ttnn.reshape(gate_tt, [1, 1, inter])
        silu_gate = ttnn.silu(gate_tt)
        gate_tt.deallocate(False)
        gated = ttnn.mul(y3d, silu_gate)
        y3d.deallocate(True); silu_gate.deallocate(True)
        out = ttnn.linear(gated, w_out, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated.deallocate(True)
        out4d = ttnn.reshape(out, [1, 1, 1, hidden])
        out.deallocate(True)
        return out4d, new_state

    def setup_layer(w_conv, prev_cap_in=None):
        """Alloc, compile, capture a single layer. Alloc happens AFTER prev trace captured."""
        cap_in = tt(torch.zeros(1, 1, 1, proj_full), device, mapper)
        state  = tt(torch.zeros(1, H, D, N), device, mapper)
        conv   = [tt(torch.zeros(1, 1, conv_dim, 1), device, mapper) for _ in range(K)]
        print(f"  cap_in={hex(cap_in.buffer_address())}", flush=True)

        # Compile
        c = ttnn.to_memory_config(cap_in, ttnn.DRAM_MEMORY_CONFIG)
        o, s = trace_body(c, state); o.deallocate(True); s.deallocate(True)
        ttnn.synchronize_device(device)

        # Capture
        p = ttnn.to_memory_config(cap_in, ttnn.DRAM_MEMORY_CONFIG)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        trace_out, trace_state = trace_body(p, state)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        print(f"  trace_out={hex(trace_out.buffer_address())}", flush=True)
        return cap_in, state, conv, tid, trace_out, trace_state

    print("Setting up L0...", flush=True)
    w_conv0 = [tt(torch.randn(1, 1, conv_dim, 1) * 0.01, device, mapper) for _ in range(K)]
    cap_in0, state0, conv0, tid0, trace_out0, trace_state0 = setup_layer(w_conv0)

    print("Setting up L1...", flush=True)
    w_conv1 = [tt(torch.randn(1, 1, conv_dim, 1) * 0.01, device, mapper) for _ in range(K)]
    cap_in1, state1, conv1, tid1, trace_out1, trace_state1 = setup_layer(w_conv1)

    hidden_tt = tt(torch.randn(1, 1, 1, hidden) * 0.1, device, mapper)

    def forward_layer(layer_id, cap_in, tid, trace_out, trace_state, st, conv, w_conv):
        print(f"L{layer_id}: in_proj+gather+conv", flush=True)
        proj = ttnn.linear(hidden_tt, w_inproj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        proj_g = ttnn.all_gather(proj, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        proj.deallocate(True)

        gate_tt = proj_g[:, :, :, :inter]
        xBC_tt  = proj_g[:, :, :, inter:inter + conv_dim]
        dt_tt   = proj_g[:, :, :, inter + conv_dim:]

        xBC_4d = ttnn.reshape(xBC_tt, [1, 1, conv_dim, 1])
        xBC_tt.deallocate(False)
        for k in range(K - 1):
            ttnn.copy(conv[k + 1], conv[k])
        ttnn.copy(xBC_4d, conv[K - 1])
        xBC_4d.deallocate(False)

        conv_acc = None
        for k in range(K):
            term = ttnn.mul(w_conv[k], conv[k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if conv_acc is None:
                conv_acc = term
            else:
                new_acc = ttnn.add(conv_acc, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                conv_acc.deallocate(True); term.deallocate(True)
                conv_acc = new_acc
        conv_silu = ttnn.silu(conv_acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_acc.deallocate(True)
        conv_flat = ttnn.reshape(conv_silu, [1, 1, 1, conv_dim])
        conv_silu.deallocate(False)

        print(f"L{layer_id}: pack+copy to trace_in", flush=True)
        packed = ttnn.concat([gate_tt, conv_flat, dt_tt], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_tt.deallocate(False); conv_flat.deallocate(False); dt_tt.deallocate(False)
        proj_g.deallocate(True)
        print(f"L{layer_id}: packed={hex(packed.buffer_address())}, cap_in={hex(cap_in.buffer_address())}", flush=True)
        # Use add(packed, zeros) → result at NEW address, then copy to cap_in
        zeros = ttnn.zeros_like(packed)
        tmp_copy = ttnn.add(packed, zeros, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        zeros.deallocate(True)
        packed.deallocate(True)
        print(f"L{layer_id}: tmp_copy={hex(tmp_copy.buffer_address())}", flush=True)
        ttnn.copy(tmp_copy, cap_in)
        tmp_copy.deallocate(True)

        print(f"L{layer_id}: execute_trace", flush=True)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)

        print(f"L{layer_id}: post-trace ops", flush=True)
        ttnn.copy(trace_state, st)
        out_g = ttnn.all_gather(trace_out, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_g.deallocate(True)
        ag_in = ttnn.reshape(hidden_tt, [1, 4, 1, hidden // 4])
        ag_out = ttnn.all_gather(ag_in, dim=1, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ag_in.deallocate(True)
        ag_sum = ttnn.sum(ag_out, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ag_out.deallocate(True); ag_sum.deallocate(True)
        print(f"L{layer_id}: done", flush=True)

    forward_layer(1, cap_in1, tid1, trace_out1, trace_state1, state1, conv1, w_conv1)
    forward_layer(0, cap_in0, tid0, trace_out0, trace_state0, state0, conv0, w_conv0)
    print("PASS", flush=True)

    ttnn.release_trace(device, tid0); ttnn.release_trace(device, tid1)
    hidden_tt.deallocate(True)
    ttnn.close_mesh_device(device); ttnn.close_mesh_device(full_mesh)

if __name__ == "__main__":
    main()
