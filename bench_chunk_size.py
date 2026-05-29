#!/usr/bin/env python3
"""
Profile prefill throughput vs mamba_chunk_size for tiny and small models.
Runs one prompt at several chunk sizes and reports tok/s.
"""
import os, time, argparse, json
import torch, ttnn

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

from transformers import AutoTokenizer
from granite.model import TTGraniteMoeHybridForCausalLM

MODELS = {
    "tiny":  ("ibm-granite/granite-4.0-h-tiny",  ttnn.MeshShape(1, 4)),
    "small": ("ibm-granite/granite-4.0-h-small", ttnn.MeshShape(2, 4)),
}

PROMPTS = {
    "short":  "The capital of France is",
    "medium": (
        "Artificial intelligence is transforming many industries. "
        "Machine learning models can now perform tasks that previously required human expertise, "
        "such as image recognition"
    ),
    "long": (
        "Large language models have become increasingly capable over recent years. "
        "These models are trained on vast amounts of text data and can generate coherent, "
        "contextually appropriate responses across a wide range of topics. "
        "They are used in applications ranging from customer service chatbots to code generation tools, "
        "scientific research assistants, and creative writing aids. "
        "The architecture underlying most of these models is the transformer, "
        "which uses self-attention mechanisms to capture long-range dependencies in text. "
        "Recent hybrid architectures combine transformers with state-space models"
    ),
}

# Chunk sizes to sweep. Must be multiples of 32 (tile size).
CHUNK_SIZES = [32, 64, 128, 256, 512]


def run(model_name, chunk_size, device, prompt_label, prompt_text, tokenizer):
    model = TTGraniteMoeHybridForCausalLM.from_pretrained(
        MODELS[model_name][0], device,
        verbose=False,
        use_tt_attention=True, use_tt_mamba=True, use_tt_moe=True,
        mamba_chunk_size=chunk_size, max_cache_length=512,
    )

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    n_tokens  = input_ids.shape[1]

    # warmup
    model.reset_cache()
    model.forward(input_ids)
    ttnn.synchronize_device(device)

    # measure prefill
    model.reset_cache()
    t0 = time.time()
    model.forward(input_ids)
    ttnn.synchronize_device(device)
    prefill_s = time.time() - t0
    prefill_tps = n_tokens / prefill_s

    # measure decode (5 steps)
    next_id = torch.argmax(model.forward(input_ids)[0, -1]).reshape(1, 1)
    model.reset_cache()
    model.forward(input_ids)
    t0 = time.time()
    for _ in range(5):
        logits  = model.forward(next_id)
        next_id = torch.argmax(logits[0, -1]).reshape(1, 1)
    ttnn.synchronize_device(device)
    decode_tps = 5 / (time.time() - t0)

    del model
    return n_tokens, prefill_tps, decode_tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tiny", "small"], default="tiny")
    args = parser.parse_args()

    model_name = args.model
    model_id, mesh_shape = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    device = full_mesh.create_submeshes(mesh_shape)[0]

    results = {}
    try:
        for prompt_label, prompt_text in PROMPTS.items():
            results[prompt_label] = {}
            print(f"\n=== {model_name} / {prompt_label} ===")
            print(f"  {'chunk_size':>12}  {'tokens':>7}  {'prefill tok/s':>14}  {'decode tok/s':>13}")
            for cs in CHUNK_SIZES:
                n_tok, pre, dec = run(model_name, cs, device, prompt_label, prompt_text, tokenizer)
                results[prompt_label][cs] = {"tokens": n_tok, "prefill_tps": round(pre, 2), "decode_tps": round(dec, 2)}
                print(f"  {cs:>12}  {n_tok:>7}  {pre:>14.1f}  {dec:>13.2f}")
    finally:
        ttnn.close_mesh_device(device)
        ttnn.close_mesh_device(full_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    out_file = f"bench_chunk_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
