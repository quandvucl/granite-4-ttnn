"""
Test the submesh scenario: full 8x4 mesh opened, 1x4 submesh used (as test_bench does).
Verify kernel + reshape on submesh.

Run: source env/bin/activate && python test_decode_step.py
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

B, H, D, N = 1, 48, 64, 128

print("Opening full 8x4 mesh...")
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    print("  fabric enabled")
except Exception as e:
    print(f"  fabric failed ({e}), opening without fabric")
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))

print("Creating 1x4 submesh...")
device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]
mapper = ttnn.ReplicateTensorToMesh(device)

def tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mapper)

def rand_tt(shape):
    return tt(torch.randn(*shape, dtype=torch.bfloat16) * 0.1)

print("Allocating state buffers...")
state_tt = tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16))
hout_tt  = tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16))
y_tt     = tt(torch.zeros(B, H, D, 1,  dtype=torch.bfloat16))
print(f"  state: {state_tt.shape}, hout: {hout_tt.shape}, y: {y_tt.shape}")

print("Decode step 1 on submesh...")
dBx = rand_tt([B, H, D, N])
dA  = rand_tt([B, H, D, N])
C   = rand_tt([B, H, N])
_ssm_update_kernel(dBx, dA, state_tt, C, device, hout_tt=hout_tt, y_tt=y_tt)
ttnn.synchronize_device(device)
print("  kernel ok")

dA.deallocate(True); dBx.deallocate(True); C.deallocate(True)
state_tt, hout_tt = hout_tt, state_tt

y_reshaped = ttnn.reshape(y_tt, [B, H, D])
print(f"  y reshaped: {y_reshaped.shape}")

print("Decode step 2...")
dBx2 = rand_tt([B, H, D, N])
dA2  = rand_tt([B, H, D, N])
C2   = rand_tt([B, H, N])
_ssm_update_kernel(dBx2, dA2, state_tt, C2, device, hout_tt=hout_tt, y_tt=y_tt)
ttnn.synchronize_device(device)
print("  kernel ok")

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
except Exception:
    pass
print("PASS")
