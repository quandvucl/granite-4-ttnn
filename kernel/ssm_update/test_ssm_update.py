#!/usr/bin/env python3
"""
Standalone correctness test for the ssm_update Metal kernel.

Compares against the reference Python implementation:
  h_out = dBx + dA * state
  y     = sum_n(h_out * C, dim=-1)

Run from /work/tt-granite/ with the virtual environment active:
  source env/bin/activate
  cd /work/tt-granite
  python kernel/ssm_update/test_ssm_update.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tt-metal"))

import torch
import ttnn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from kernel.ssm_update.op import ssm_update

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_test(device, B=1, H=48, D=64, N=128):
    print(f"Testing B={B} H={H} D={D} N={N}")

    torch.manual_seed(42)
    dBx_np   = torch.randn(B, H, D, N, dtype=torch.bfloat16)
    dA_np    = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1 - 1.0  # negative
    state_np = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1
    C_np     = torch.randn(B, H, N,    dtype=torch.bfloat16)

    # Reference CPU computation — use bfloat16 throughout to match hardware precision
    h_out_ref = (dBx_np + dA_np * state_np)
    yc_ref    = h_out_ref * C_np.unsqueeze(-2)   # [B,H,1,N] broadcast over D
    y_ref     = yc_ref.sum(dim=-1)               # [B,H,D]
    # bfloat16 reference (accumulate in bf16 like hardware does)
    y_ref_bf16 = yc_ref.to(torch.bfloat16).sum(dim=-1).float()

    # Move to device
    def to_tt(t, shape=None):
        if shape is not None:
            t = t.reshape(shape)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    dBx_tt   = to_tt(dBx_np)
    dA_tt    = to_tt(dA_np)
    state_tt = to_tt(state_np)
    C_tt     = to_tt(C_np)

    # TTNN reference: compute y using standard ops on device
    hout_ref_tt = ttnn.add(dBx_tt, ttnn.multiply(dA_tt, state_tt))
    C_4d_tt = ttnn.unsqueeze(C_tt, -2)           # [B,H,1,N]
    yc_ref_tt = ttnn.multiply(hout_ref_tt, C_4d_tt)
    y_ref_tt = ttnn.sum(yc_ref_tt, dim=-1)       # [B,H,D]
    ttnn.synchronize_device(device)
    y_ttnn_ref = ttnn.to_torch(y_ref_tt).float()[:, :H, :D]

    # Run kernel
    hout_tt, y_raw_tt = ssm_update(dBx_tt, dA_tt, state_tt, C_tt, device)
    ttnn.synchronize_device(device)

    # Read back
    hout_got = ttnn.to_torch(hout_tt).float()
    y_raw_got = ttnn.to_torch(y_raw_tt).float()  # [B, H, D, 1]

    # Try reshaping [B,H,D,1] → [B,H,D] via ttnn
    y_reshaped_tt = ttnn.reshape(y_raw_tt, ttnn.Shape([B, H, D]))
    ttnn.synchronize_device(device)
    y_got_reshaped = ttnn.to_torch(y_reshaped_tt).float()[:, :H, :D]

    # Also extract directly from padded 4D tensor (column 0)
    y_got = y_raw_got[:, :H, :D, 0]  # trim padding, take column 0

    print(f"  y_raw shape: {y_raw_tt.shape}, y_reshaped shape: {y_reshaped_tt.shape}")

    # PCC
    pcc_hout       = pcc(hout_got[:, :H, :D, :N], h_out_ref)
    pcc_y          = pcc(y_got, y_ref)
    pcc_y_bf16     = pcc(y_got, y_ref_bf16)
    pcc_y_vs_ttnn  = pcc(y_got, y_ttnn_ref)

    print(f"  h_out PCC       = {pcc_hout:.6f}  (target > 0.999)")
    print(f"  y     PCC       = {pcc_y:.6f}  (vs float ref)")
    print(f"  y     PCC       = {pcc_y_bf16:.6f}  (vs bf16 ref)")
    print(f"  y     PCC       = {pcc_y_vs_ttnn:.6f}  (vs TTNN ops ref)")

    if pcc_hout > 0.999 and pcc_y_vs_ttnn > 0.999:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        # Debug: check intermediate values
        print(f"  h_out ref range [{h_out_ref.min():.3f}, {h_out_ref.max():.3f}]")
        print(f"  h_out got range [{hout_got[:,:H,:D,:N].min():.3f}, {hout_got[:,:H,:D,:N].max():.3f}]")
        print(f"  y ref range [{y_ref.min():.3f}, {y_ref.max():.3f}]")
        print(f"  y got range [{y_got.min():.3f}, {y_got.max():.3f}]")
        return False


def run_test_mesh(mesh_device, B=1, H=48, D=64, N=128):
    """Run correctness test on a MeshDevice with replicated inputs.
    Compares against TTNN standard ops on the same device (not CPU bf16 sum,
    which has different reduction order and gives lower PCC).
    """
    print(f"Testing MeshDevice {mesh_device.shape} B={B} H={H} D={D} N={N}")

    torch.manual_seed(42)
    dBx_np   = torch.randn(B, H, D, N, dtype=torch.bfloat16)
    dA_np    = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1 - 1.0
    state_np = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1
    C_np     = torch.randn(B, H, N, dtype=torch.bfloat16)

    h_out_ref = (dBx_np + dA_np * state_np)

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_tt_mesh(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=mapper)

    dBx_tt   = to_tt_mesh(dBx_np)
    dA_tt    = to_tt_mesh(dA_np)
    state_tt = to_tt_mesh(state_np)
    C_tt     = to_tt_mesh(C_np)

    # TTNN standard ops reference using the SAME input tensors (reused after ops, as in single-device test).
    # The ops create new output tensors; inputs are not modified.
    hout_ref_tt = ttnn.add(dBx_tt, ttnn.multiply(dA_tt, state_tt))
    C_4d_tt = ttnn.unsqueeze(C_tt, -2)
    yc_ref_tt = ttnn.multiply(hout_ref_tt, C_4d_tt)
    y_ref_tt = ttnn.sum(yc_ref_tt, dim=-1)
    ttnn.synchronize_device(mesh_device)
    y_ttnn_ref = ttnn.to_torch(ttnn.get_device_tensors(y_ref_tt)[0]).float()[:, :H, :D]

    # Run kernel on same inputs
    hout_tt, y_raw_tt = ssm_update(dBx_tt, dA_tt, state_tt, C_tt, mesh_device)
    ttnn.synchronize_device(mesh_device)

    hout_got = ttnn.to_torch(ttnn.get_device_tensors(hout_tt)[0]).float()
    y_raw_got = ttnn.to_torch(ttnn.get_device_tensors(y_raw_tt)[0]).float()
    y_got = y_raw_got[:, :H, :D, 0]

    pcc_hout      = pcc(hout_got[:, :H, :D, :N], h_out_ref)
    pcc_y_vs_ttnn = pcc(y_got, y_ttnn_ref)

    print(f"  h_out PCC = {pcc_hout:.6f}  (target > 0.999)")
    print(f"  y     PCC = {pcc_y_vs_ttnn:.6f}  (vs TTNN ops, target > 0.999)")

    if pcc_hout > 0.999 and pcc_y_vs_ttnn > 0.999:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", action="store_true", help="Test on 1x4 MeshDevice")
    args = parser.parse_args()

    if args.mesh:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
        try:
            ok = run_test_mesh(mesh_device, B=1, H=48, D=64, N=128)
            sys.exit(0 if ok else 1)
        finally:
            ttnn.close_mesh_device(mesh_device)
    else:
        device = ttnn.open_device(device_id=0)
        try:
            ok = run_test(device, B=1, H=48, D=64, N=128)
            sys.exit(0 if ok else 1)
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
