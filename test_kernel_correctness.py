"""
Numerical correctness test for the ssm_update Metal kernel.

Reference formula (per element):
  h_out[b,h,d,n] = dA[b,h,d,n] * state[b,h,d,n] + dBx[b,h,d,n]
  y[b,h,d]       = sum_n( h_out[b,h,d,n] * C[b, h//32*32 : h//32*32+32, n][h%32] )
                 = sum_n( h_out[b,h,d,n] * C[b,h,n] )
                 (C has shape [B,H,N]; h%32 selects row within the 32-row tile,
                  but since C is [B,H,N] each h has its own row — no ambiguity)

Tests:
  1. Single-step correctness: randn inputs, compare kernel vs torch reference
  2. Multi-step state accumulation: 8 steps, state carries forward correctly
  3. Multi-layer (cache_bust_id): two layers interleaved, no cross-layer corruption

Run: source env/bin/activate && python test_kernel_correctness.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tt-metal"))
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

import torch
import ttnn
from kernel.ssm_update.op import ssm_update as _kernel

# granite-4.0-h-tiny dims
B, H, D, N = 1, 48, 64, 128

# ── helpers ──────────────────────────────────────────────────────────────────

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
mapper = ttnn.ReplicateTensorToMesh(device)

def to_tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mapper)

def from_tt(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()

def ref_ssm_step(dBx, dA, state, C):
    """Pure-PyTorch reference for one decode step."""
    h = dA * state + dBx                        # [B,H,D,N]
    y = (h * C.unsqueeze(2)).sum(-1)            # [B,H,D]
    return h, y

def alloc_state():
    return torch.zeros(B, H, D, N, dtype=torch.bfloat16)

def alloc_outputs():
    hout_tt = to_tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16))
    y_tt    = to_tt(torch.zeros(B, H, D, 1,  dtype=torch.bfloat16))
    return hout_tt, y_tt

def check(name, got, ref, atol=None, rtol=0.02):
    """Pass if max_abs < atol OR max_rel < rtol (element-wise OR).
    atol default: 2× bfloat16 epsilon (0.008) for element ops;
                  caller raises it for reduction outputs.
    """
    if atol is None:
        atol = 0.008
    diff = (got - ref).abs()
    abs_ = diff.max().item()
    scale = ref.abs().clamp(min=1e-6)
    rel = (diff / scale).max().item()
    ok = abs_ < atol or rel < rtol
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name:40s}  max_rel={rel:.4f}  max_abs={abs_:.4f}  atol={atol}")
    return ok

# ── Test 1: single-step correctness ──────────────────────────────────────────
print("\n=== Test 1: single-step correctness ===")

torch.manual_seed(42)
dBx   = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1
dA    = torch.sigmoid(torch.randn(B, H, D, N, dtype=torch.bfloat16)) * 0.99 + 0.005
state = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.3
C     = torch.randn(B, H, N, dtype=torch.bfloat16)

ref_h, ref_y = ref_ssm_step(dBx.float(), dA.float(), state.float(), C.float())

state_tt = to_tt(state)
hout_tt, y_tt = alloc_outputs()

_kernel(to_tt(dBx), to_tt(dA), state_tt, to_tt(C), device,
        hout_tt=hout_tt, y_tt=y_tt, cache_bust_id=0)
ttnn.synchronize_device(device)

got_h = from_tt(hout_tt)
got_y = from_tt(ttnn.reshape(y_tt, [B, H, D]))

all_pass = True
all_pass &= check("h_out (state update)", got_h, ref_h)
all_pass &= check("y     (output)",       got_y, ref_y, atol=0.2)

# ── Test 2: multi-step state accumulation ────────────────────────────────────
print("\n=== Test 2: multi-step state accumulation (8 steps) ===")

torch.manual_seed(7)
state_pt  = torch.zeros(B, H, D, N, dtype=torch.float32)
state_tt  = to_tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16))
hout_tt, y_tt = alloc_outputs()

for step in range(8):
    dBx_t = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1
    dA_t  = torch.sigmoid(torch.randn(B, H, D, N, dtype=torch.bfloat16)) * 0.99 + 0.005
    C_t   = torch.randn(B, H, N, dtype=torch.bfloat16)

    # PyTorch reference
    state_pt, ref_y = ref_ssm_step(dBx_t.float(), dA_t.float(), state_pt, C_t.float())

    # Kernel
    _kernel(to_tt(dBx_t), to_tt(dA_t), state_tt, to_tt(C_t), device,
            hout_tt=hout_tt, y_tt=y_tt, cache_bust_id=0)
    ttnn.synchronize_device(device)
    state_tt, hout_tt = hout_tt, state_tt   # swap, just like forward_decode

    got_y = from_tt(ttnn.reshape(y_tt, [B, H, D]))
    ok = check(f"step {step} y", got_y, ref_y, atol=0.2)
    all_pass &= ok

# Also compare final state
got_state = from_tt(state_tt)
all_pass &= check("final state", got_state, state_pt, atol=0.02)

# ── Test 3: multi-layer (cache_bust_id isolation) ────────────────────────────
print("\n=== Test 3: multi-layer cache isolation (2 layers × 4 steps) ===")

torch.manual_seed(99)
# Two independent SSM state buffers (simulating two Mamba layers)
states_pt  = [torch.zeros(B, H, D, N, dtype=torch.float32) for _ in range(2)]
states_tt  = [to_tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16)) for _ in range(2)]
houts_tt   = [to_tt(torch.zeros(B, H, D, N, dtype=torch.bfloat16)) for _ in range(2)]
ys_tt      = [to_tt(torch.zeros(B, H, D, 1, dtype=torch.bfloat16)) for _ in range(2)]

for step in range(4):
    for layer in range(2):
        dBx_t = torch.randn(B, H, D, N, dtype=torch.bfloat16) * 0.1
        dA_t  = torch.sigmoid(torch.randn(B, H, D, N, dtype=torch.bfloat16)) * 0.99 + 0.005
        C_t   = torch.randn(B, H, N, dtype=torch.bfloat16)

        states_pt[layer], ref_y = ref_ssm_step(
            dBx_t.float(), dA_t.float(), states_pt[layer], C_t.float())

        _kernel(to_tt(dBx_t), to_tt(dA_t), states_tt[layer], to_tt(C_t), device,
                hout_tt=houts_tt[layer], y_tt=ys_tt[layer], cache_bust_id=layer)
        ttnn.synchronize_device(device)
        states_tt[layer], houts_tt[layer] = houts_tt[layer], states_tt[layer]

        got_y = from_tt(ttnn.reshape(ys_tt[layer], [B, H, D]))
        ok = check(f"layer {layer} step {step} y", got_y, ref_y, atol=0.2)
        all_pass &= ok

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
print("=" * 60)

ttnn.close_mesh_device(device)
