"""Compare trace vs no-trace logits at each decode step using argmax only."""
import os, sys, time
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)
import torch, ttnn
from transformers import AutoTokenizer
from granite.model import TTGraniteMoeHybridForCausalLM

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=268435456)
device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]

model_id = "ibm-granite/granite-4.0-h-tiny"
tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
    model_id, device, verbose=False,
    use_tt_attention=True, use_tt_mamba=True, use_tt_moe=True,
    mamba_chunk_size=256, max_cache_length=512,
    moe_weight_dtype=ttnn.bfloat8_b, moe_use_all_gather=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]

def run(use_trace):
    tt_model._disable_trace = not use_trace
    tt_model.reset_cache()
    logits = tt_model.forward(input_ids)
    ttnn.synchronize_device(device)
    next_id = logits[0, -1, :].float().argmax().item()
    gen = [next_id]
    next_tensor = torch.zeros((1, 1), dtype=input_ids.dtype)
    for step in range(8):
        next_tensor[0, 0] = next_id
        logits = tt_model.forward(next_tensor)
        ttnn.synchronize_device(device)
        top3 = logits[0, -1, :].float().topk(3)
        toks = [tokenizer.decode([i]) for i in top3.indices.tolist()]
        print(f"  step={step+1} trace_id={tt_model._decode_trace_id is not None} "
              f"input={tokenizer.decode([next_id])!r:8s} top3={toks}", flush=True)
        next_id = top3.indices[0].item()
        gen.append(next_id)
        # After 2 warmup steps, capture trace for subsequent steps
        if step == 1 and use_trace:
            tt_model.capture_decode_trace()
    print("  result:", tokenizer.decode(gen, skip_special_tokens=True), flush=True)

print("=== WITH TRACE ===", flush=True)
run(use_trace=True)
print("\n=== NO TRACE ===", flush=True)
run(use_trace=False)

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
