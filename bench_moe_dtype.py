#!/usr/bin/env python3
"""
Sweep MoE expert weight dtype (bfloat8_b vs bfloat16) and measure
prefill throughput, decode throughput, and response quality.
"""
import os, time, torch, ttnn

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

from transformers import AutoTokenizer
from granite.model import TTGraniteMoeHybridForCausalLM

MODEL_ID     = "ibm-granite/granite-4.0-h-tiny"
PROMPT       = "The largest planet in our solar system is"
DECODE_TOKENS = 10
DTYPES = [
    ("bfloat16", ttnn.bfloat16),
    ("bfloat8_b", ttnn.bfloat8_b),
]

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
print(f"Prompt tokens: {input_ids.shape[1]}\n")

results = []
for dtype_name, dtype in DTYPES:
    print(f"=== MoE weight dtype: {dtype_name} ===")
    model = TTGraniteMoeHybridForCausalLM.from_pretrained(
        MODEL_ID, device,
        verbose=False,
        use_tt_attention=True, use_tt_mamba=True, use_tt_moe=True,
        mamba_chunk_size=256, max_cache_length=512,
        moe_weight_dtype=dtype,
    )

    model.reset_cache()
    t0 = time.time()
    logits = model.forward(input_ids)
    ttnn.synchronize_device(device)
    prefill_tps = input_ids.shape[1] / (time.time() - t0)

    next_id = torch.argmax(logits[0, -1]).reshape(1, 1)
    tokens  = [next_id.item()]
    t0 = time.time()
    for _ in range(DECODE_TOKENS - 1):
        logits  = model.forward(next_id)
        next_id = torch.argmax(logits[0, -1]).reshape(1, 1)
        tokens.append(next_id.item())
    ttnn.synchronize_device(device)
    decode_tps = (DECODE_TOKENS - 1) / (time.time() - t0)

    response = tokenizer.decode(tokens)
    print(f"  Prefill: {prefill_tps:.1f} tok/s  |  Decode: {decode_tps:.2f} tok/s")
    print(f"  Response: {response}\n")
    results.append((dtype_name, prefill_tps, decode_tps, response))

    del model

ttnn.close_mesh_device(device)
ttnn.close_mesh_device(full_mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

print("=== SUMMARY ===")
print(f"{'dtype':<12} {'prefill tok/s':>14} {'decode tok/s':>13}")
for name, pre, dec, _ in results:
    print(f"{name:<12} {pre:>14.1f} {dec:>13.2f}")
