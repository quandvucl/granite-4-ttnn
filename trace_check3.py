"""Trace with use_tt_moe=False — isolate whether MoE is causing trace corruption."""
import os, sys, time
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)
import torch, ttnn
from transformers import AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from granite.model import TTGraniteMoeHybridForCausalLM

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=268435456)
device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]

model_id = "ibm-granite/granite-4.0-h-tiny"
# use_tt_moe=False forces HF CPU MoE — trace will NOT be used (moe_use_all_gather is irrelevant)
tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
    model_id, device, verbose=False,
    use_tt_attention=True, use_tt_mamba=True, use_tt_moe=False,
    mamba_chunk_size=256, max_cache_length=512,
    moe_weight_dtype=ttnn.bfloat8_b, moe_use_all_gather=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
rep_proc = RepetitionPenaltyLogitsProcessor(penalty=1.3)

def decode_20(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    tt_model.reset_cache()
    logits = tt_model.forward(input_ids)
    ttnn.synchronize_device(device)
    context = input_ids[0].tolist()
    scores = logits[0, -1].float().unsqueeze(0)
    scores = rep_proc(torch.tensor([context], dtype=torch.long), scores)
    next_id = scores[0].argmax().item()
    gen = [next_id]
    next_tensor = torch.zeros((1,1), dtype=input_ids.dtype)
    for _ in range(19):
        next_tensor[0,0] = next_id
        logits = tt_model.forward(next_tensor)
        ttnn.synchronize_device(device)
        scores = logits[0,-1].float().unsqueeze(0)
        scores = rep_proc(torch.tensor([context+gen], dtype=torch.long), scores)
        next_id = scores[0].argmax().item()
        gen.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(gen, skip_special_tokens=True)

print(f"_trace_supported would be: {tt_model.tt_ccl is not None and tt_model.moe_use_all_gather}")
for prompt in [
    "The capital of France is",
    "The largest planet in our solar system is",
]:
    print(f"\n[{prompt}]")
    print(decode_20(prompt))

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
