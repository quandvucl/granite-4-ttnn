"""
Debug: compare trace vs non-trace logits token by token.
Runs 6 decode steps: steps 1-2 non-trace (compile+warmup), step 3 trace capture,
steps 4-6 trace replay. Prints top-5 token indices for each step.
"""
import os, sys
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)
import torch, ttnn
from transformers import AutoTokenizer
from granite.model import TTGraniteMoeHybridForCausalLM
from utils import to_torch_tensor

print("Opening mesh...", flush=True)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=268435456)
device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]

print("Loading model...", flush=True)
model_id = "ibm-granite/granite-4.0-h-tiny"
tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
    model_id, device, verbose=False,
    use_tt_attention=True, use_tt_mamba=True, use_tt_moe=True,
    mamba_chunk_size=256, max_cache_length=512,
    moe_weight_dtype=ttnn.bfloat8_b, moe_use_all_gather=True,
)
tt_model._trace_debug = True  # enable per-step top-5 logging

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]

tt_model.reset_cache()
print("Prefill...", flush=True)
logits = tt_model.forward(input_ids)
ttnn.synchronize_device(device)

if isinstance(logits, ttnn.Tensor):
    last = logits[0, 0, -1, :]
    if last.dtype == ttnn.bfloat8_b:
        last = ttnn.typecast(last, ttnn.bfloat16)
    next_id = to_torch_tensor(last).float().argmax().item()
else:
    next_id = logits[0, -1, :].float().argmax().item()

print(f"First token: {next_id} = {tokenizer.decode([next_id])!r}", flush=True)

next_tensor = torch.zeros((1, 1), dtype=input_ids.dtype)
for step in range(6):
    next_tensor[0, 0] = next_id
    print(f"\n--- Decode step {step+1} (input token {next_id} = {tokenizer.decode([next_id])!r}) ---", flush=True)
    logits = tt_model.forward(next_tensor)
    ttnn.synchronize_device(device)
    if isinstance(logits, ttnn.Tensor):
        last = logits[0, 0, -1, :]
        if last.dtype == ttnn.bfloat8_b: last = ttnn.typecast(last, ttnn.bfloat16)
        top5 = to_torch_tensor(last).float().topk(5)
    else:
        top5 = logits[0, -1].float().topk(5)
    print(f"  top5 ids:    {top5.indices.tolist()}", flush=True)
    print(f"  top5 tokens: {[tokenizer.decode([i]) for i in top5.indices.tolist()]}", flush=True)
    next_id = top5.indices[0].item()

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
