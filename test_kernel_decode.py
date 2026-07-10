"""
Minimal test: verify ssm_update kernel works with decode-path tensor shapes and
from_torch mesh tensors (the same way mamba_chunk_scan_parallel.py creates them).

Run after: tt-smi -r 0
  source env/bin/activate && python test_kernel_decode.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tt-metal"))
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

import torch
import ttnn
from kernel.ssm_update.op import ssm_update as _ssm_update_kernel

# Tiny model dims (granite-4.0-h-tiny)
B, H, D, N = 1, 48, 64, 128

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
mapper = ttnn.ReplicateTensorToMesh(device)

def tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mapper)

print("Allocating buffers...")
state_tt  = tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16))
hout_tt   = tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16))
y_tt      = tt(torch.zeros(B, H, D, 1,  dtype=torch.bfloat16))

print("Building decode inputs...")
dBx_tt = tt(torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1)
dA_tt  = tt(torch.ones(B, H, D, N,  dtype=torch.bfloat16) * 0.99)
C_tt   = tt(torch.randn(B, H, N,    dtype=torch.bfloat16))  # 3D — as in forward_decode

print("Calling kernel (step 1)...")
_ssm_update_kernel(dBx_tt, dA_tt, state_tt, C_tt, device, hout_tt=hout_tt, y_tt=y_tt)
ttnn.synchronize_device(device)
print("Step 1 done. Swapping buffers...")

state_tt, hout_tt = hout_tt, state_tt

# Reshape y as forward_decode does
y_reshaped = ttnn.reshape(y_tt, [B, H, D])
print(f"y shape after reshape: {y_reshaped.shape}")

print("Calling kernel (step 2)...")
dBx_tt2 = tt(torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1)
dA_tt2  = tt(torch.ones(B, H, D, N,  dtype=torch.bfloat16) * 0.99)
C_tt2   = tt(torch.randn(B, H, N,    dtype=torch.bfloat16))
_ssm_update_kernel(dBx_tt2, dA_tt2, state_tt, C_tt2, device, hout_tt=hout_tt, y_tt=y_tt)
ttnn.synchronize_device(device)
print("Step 2 done.")

ttnn.close_mesh_device(device)
print("PASS")
