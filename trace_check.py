"""Log _decode_trace_input after capture to verify correct embedding is stored."""
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
embed = tt_model.embed_tokens  # CPU embed_tokens for reference

input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
tt_model.reset_cache()
logits = tt_model.forward(input_ids)
ttnn.synchronize_device(device)
next_id = logits[0, -1, :].float().argmax().item()

next_tensor = torch.zeros((1, 1), dtype=input_ids.dtype)
for step in range(6):
    next_tensor[0, 0] = next_id
    tok = tokenizer.decode([next_id])
    logits = tt_model.forward(next_tensor)
    ttnn.synchronize_device(device)

    # After capture step, read back _decode_trace_input and compare to expected embedding
    if tt_model._decode_trace_id is not None and tt_model._decode_trace_input is not None:
        trace_input_cpu = ttnn.to_torch(
            tt_model._decode_trace_input,
            mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
        )[0:1]  # [1, 1, 1, H]
        expected_emb = embed(next_tensor).float().to(torch.bfloat16)
        if tt_model.config.embedding_multiplier != 1.0:
            expected_emb = expected_emb * tt_model.config.embedding_multiplier
        expected_emb_4d = expected_emb.reshape(1, 1, 1, -1)
        diff = (trace_input_cpu.float() - expected_emb_4d.float()).abs().max().item()
        print(f"step={step+1} tok={tok!r} trace_id_set={tt_model._decode_trace_id is not None} "
              f"trace_input vs expected embedding: max_diff={diff:.6f}", flush=True)
    else:
        print(f"step={step+1} tok={tok!r} trace_id=None (no trace yet)", flush=True)

    next_id = logits[0, -1, :].float().argmax().item()

ttnn.close_mesh_device(device)
for sm in full_mesh.get_submeshes():
    ttnn.close_mesh_device(sm)
ttnn.close_mesh_device(full_mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
