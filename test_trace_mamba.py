"""
Watchdog-wrapped test: load the full model, run prefill + 2 decode steps.
If it hangs for >60s, dumps the Python stack and exits — no tt-smi needed.

Because TTNN itself may hang in C++, the watchdog uses os._exit() for a hard kill.
After the hard exit, the TT device is in a bad state — run tt-smi -r 0 once
to reset before the NEXT run. But within a session, reruns after clean exit are fine.

Run: source env/bin/activate && python test_trace_mamba.py
"""
import os, sys, threading, traceback, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tt-metal"))
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

# ── Watchdog ─────────────────────────────────────────────────────────────────
_watchdog_deadline = [time.time() + 600]  # updated after model load + prefill

def _watchdog():
    while True:
        time.sleep(2)
        if time.time() > _watchdog_deadline[0]:
            print(f"\n=== WATCHDOG FIRED — dumping stacks ===", flush=True)
            for tid, frame in sys._current_frames().items():
                print(f"\n--- Thread {tid} ---")
                traceback.print_stack(frame)
            print("=== END WATCHDOG ===", flush=True)
            os._exit(1)

threading.Thread(target=_watchdog, daemon=True).start()
# ─────────────────────────────────────────────────────────────────────────────

import torch
import ttnn
from transformers import AutoTokenizer

print("Opening mesh (no fabric)...", flush=True)
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
except Exception:
    pass
full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
print("  mesh ok", flush=True)

device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]

print("Loading model...", flush=True)
from granite.model import TTGraniteMoeHybridForCausalLM
model_id = "ibm-granite/granite-4.0-h-tiny"
tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
    model_id, device,
    verbose=False,
    use_tt_attention=True,
    use_tt_mamba=True,
    use_tt_moe=True,
    mamba_chunk_size=256,
    max_cache_length=512,
    moe_weight_dtype=ttnn.bfloat8_b,
    moe_use_all_gather=False,  # disable fabric all_gather to test NOC conflict hypothesis
)
print("Model loaded.", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]

print("Prefill...", flush=True)
tt_model.reset_cache()
logits = tt_model.forward(input_ids)
ttnn.synchronize_device(device)
print("Prefill done.", flush=True)
_watchdog_deadline[0] = time.time() + 30  # 30s for each decode step

from utils import to_torch_tensor
if isinstance(logits, ttnn.Tensor):
    last = logits[0, 0, -1, :]
    if last.dtype == ttnn.bfloat8_b:
        last = ttnn.typecast(last, ttnn.bfloat16)
    next_id = to_torch_tensor(last).float().argmax().item()
else:
    next_id = logits[0, -1, :].float().argmax().item()

next_tensor = torch.zeros((1, 1), dtype=input_ids.dtype)
next_tensor[0, 0] = next_id

print("Decode step 1...", flush=True)
_watchdog_deadline[0] = time.time() + 600  # first decode compiles all 36 kernels
logits2 = tt_model.forward(next_tensor)
ttnn.synchronize_device(device)
print("Decode step 1 done.", flush=True)
_watchdog_deadline[0] = time.time() + 120

if isinstance(logits2, ttnn.Tensor):
    last2 = logits2[0, 0, -1, :]
    if last2.dtype == ttnn.bfloat8_b:
        last2 = ttnn.typecast(last2, ttnn.bfloat16)
    next_id2 = to_torch_tensor(last2).float().argmax().item()
else:
    next_id2 = logits2[0, -1, :].float().argmax().item()
next_tensor[0, 0] = next_id2

print("Decode step 2...", flush=True)
_watchdog_deadline[0] = time.time() + 120  # kernels compiled, 120s is ample
logits3 = tt_model.forward(next_tensor)
ttnn.synchronize_device(device)
print("Decode step 2 done — PASS", flush=True)
_watchdog_deadline[0] = time.time() + 600

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
except Exception:
    pass
