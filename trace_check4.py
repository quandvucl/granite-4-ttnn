"""Verify _cur_pos_tt updates correctly across trace replays."""
import os, sys, time
os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)
import torch, ttnn
from transformers import AutoTokenizer
from granite.model import TTGraniteMoeHybridForCausalLM
from utils import to_torch_tensor

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
prefill_len = input_ids.shape[1]
print(f"Prefill length: {prefill_len}", flush=True)

tt_model.reset_cache()
logits = tt_model.forward(input_ids)
ttnn.synchronize_device(device)
next_id = logits[0, -1, :].float().argmax().item()

next_tensor = torch.zeros((1, 1), dtype=input_ids.dtype)
attn_layers = [l for l in tt_model.layers if l.is_attention_layer and l.simple_attention is not None]
first_attn = attn_layers[0]

for step in range(7):
    next_tensor[0, 0] = next_id
    # Log _cur_pos_tt BEFORE forward
    pos_before = ttnn.to_torch(first_attn.simple_attention._cur_pos_tt)[0].item() if not tt_model._decode_trace_id else -1
    t0 = time.time()
    logits = tt_model.forward(next_tensor)
    ttnn.synchronize_device(device)
    ms = (time.time()-t0)*1000
    # Log _cur_pos_tt AFTER forward (only safe outside trace)
    pos_after_raw = first_attn.simple_attention._cur_pos_tt
    try:
        pos_after = ttnn.to_torch(pos_after_raw)[0].item()
    except:
        pos_after = "ERR"
    tok = tokenizer.decode([next_id])
    has_trace = tt_model._decode_trace_id is not None
    print(f"step={step+1} has_trace={has_trace} pos_before={pos_before} pos_after={pos_after} {ms:.1f}ms tok={tok!r}", flush=True)
    next_id = logits[0, -1, :].float().argmax().item()

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
