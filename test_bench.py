#!/usr/bin/env python3
"""
Benchmark Granite models: model load time + prefill + decode across prompt lengths.

Usage:
  python test_bench.py                        # tiny(4) + small(8), all prompts
  python test_bench.py --model tiny           # tiny only
  python test_bench.py --model small          # small only
  python test_bench.py --decode-tokens 30     # more decode steps
"""
import argparse
import json
import os
import time
import sys
import torch
import ttnn

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)
from transformers import AutoTokenizer, AutoConfig
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

MODELS = {
    "tiny":  ("ibm-granite/granite-4.0-h-tiny",  4),
    "small": ("ibm-granite/granite-4.0-h-small",  8),
}

PROMPTS = {
    "short_8":   "The capital of France is",
    "short_10":  "The largest planet in our solar system is",
    "medium_32": (
        "Artificial intelligence is transforming many industries. "
        "Machine learning models can now perform tasks that previously required human expertise, "
        "such as image recognition"
    ),
    "long_128": (
        "Large language models have become increasingly capable over recent years. "
        "These models are trained on vast amounts of text data and can generate coherent, "
        "contextually appropriate responses across a wide range of topics. "
        "They are used in applications ranging from customer service chatbots to code generation tools, "
        "scientific research assistants, and creative writing aids. "
        "The architecture underlying most of these models is the transformer, "
        "which uses self-attention mechanisms to capture long-range dependencies in text. "
        "Recent hybrid architectures combine transformers with state-space models"
    ),
    "long_256": (
        "The field of natural language processing has undergone a remarkable transformation "
        "with the advent of large-scale pretrained language models. These systems, trained on "
        "hundreds of billions of tokens of text from the internet and other sources, have "
        "demonstrated surprising emergent capabilities including multi-step reasoning, "
        "in-context learning, and instruction following. The scaling laws governing these "
        "models suggest that performance on many tasks continues to improve predictably with "
        "increases in model size, dataset size, and compute budget. This has motivated the "
        "development of increasingly large models, from hundreds of millions to hundreds of "
        "billions of parameters. However, serving these models efficiently at inference time "
        "presents significant engineering challenges, particularly for applications requiring "
        "low latency responses. Speculative decoding, quantization, and hardware-specific "
        "optimizations are among the techniques used to address these challenges. "
        "Hybrid architectures that combine attention layers with state-space models like Mamba "
        "offer promising tradeoffs between computational efficiency and modeling capability"
    ),
}

DECODE_TOKENS = 20
REPETITION_PENALTY = 1.3


MESH_SHAPE_MAP = {
    1: ttnn.MeshShape(1, 1),
    2: ttnn.MeshShape(1, 2),
    4: ttnn.MeshShape(1, 4),
    8: ttnn.MeshShape(2, 4),
}


def run_bench(model_name, full_mesh, decode_tokens=DECODE_TOKENS):
    from granite.model import TTGraniteMoeHybridForCausalLM
    from utils import to_torch_tensor

    model_id, requested_devices = MODELS[model_name]

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_kv_heads = hf_config.num_key_value_heads
    num_devices = min(requested_devices, num_kv_heads)

    print(f"\n{'='*70}")
    print(f"  Model : {model_id}")
    print(f"  Devices: {num_devices}  (requested {requested_devices}, KV heads {num_kv_heads})")
    print(f"  Decode : {decode_tokens} tokens per prompt")
    print(f"{'='*70}")

    full_mesh_devices = full_mesh.get_num_devices()
    if num_devices == full_mesh_devices:
        device = full_mesh  # can't create a submesh covering the entire parent mesh
    else:
        device = full_mesh.create_submeshes(MESH_SHAPE_MAP[num_devices])[0]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Model load ──────────────────────────────────────────────────────
        t_load0 = time.time()
        tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
            model_id, device,
            verbose=False,
            use_tt_attention=True,
            use_tt_mamba=True,
            use_tt_moe=True,
            mamba_chunk_size=128 if num_devices >= 4 else None,
            max_cache_length=512,
        )
        load_s = time.time() - t_load0
        print(f"\nModel load: {load_s:.1f} s\n")

        results = []

        for prompt_label, prompt_text in PROMPTS.items():
            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            actual_len = input_ids.shape[1]

            print(f"── {prompt_label}  (actual tokens: {actual_len}) ──────────────────")

            tt_model.reset_cache()

            # Prefill
            t0 = time.time()
            logits = tt_model.forward(input_ids)
            ttnn.synchronize_device(device)
            prefill_ms = (time.time() - t0) * 1000

            rep_processor = RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY)

            def pick_next(logits_raw, seen_ids):
                if isinstance(logits_raw, ttnn.Tensor):
                    last = logits_raw[0, 0, -1, :]
                    if last.dtype == ttnn.bfloat8_b:
                        last = ttnn.typecast(last, ttnn.bfloat16)
                    scores = to_torch_tensor(last).float().reshape(1, -1)
                else:
                    scores = logits_raw[0, -1, :].float().unsqueeze(0)
                vocab_size = scores.shape[1]
                valid = [tid for tid in seen_ids if tid < vocab_size]
                if valid:
                    seen = torch.tensor([valid], dtype=torch.long)
                    scores = rep_processor(seen, scores)
                return scores[0].argmax().item()

            context_ids = input_ids[0].tolist()
            next_id = pick_next(logits, context_ids)

            # Decode loop
            decode_times = []
            generated_ids = [next_id]
            next_tensor = torch.zeros((1, 1), dtype=input_ids.dtype)
            for step in range(decode_tokens):
                next_tensor[0, 0] = next_id
                t0 = time.time()
                logits = tt_model.forward(next_tensor)
                ttnn.synchronize_device(device)
                step_ms = (time.time() - t0) * 1000
                decode_times.append(step_ms)

                next_id = pick_next(logits, context_ids + generated_ids)

                generated_ids.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    break

            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  Prompt   : {prompt_text}")
            print(f"  Response : {response}")

            # Stats — skip warmup (first decode step)
            steady = decode_times[1:] if len(decode_times) > 1 else decode_times
            avg_ms = sum(steady) / len(steady)
            tok_s  = 1000.0 / avg_ms

            print(f"  Prefill: {actual_len*1000/prefill_ms:.1f} tok/s  |  Decode: {tok_s:.2f} tok/s")

            results.append({
                "prompt": prompt_label,
                "tokens": actual_len,
                "prefill_toks": actual_len * 1000 / prefill_ms,
                "decode_toks": tok_s,
                "prompt_text": prompt_text,
                "response": response,
            })

        # ── Summary table ───────────────────────────────────────────────────
        print(f"{'='*70}")
        print(f"  SUMMARY: {model_name.upper()} on {num_devices} devices")
        print(f"  Model load: {load_s:.1f} s")
        print(f"{'─'*70}")
        print(f"  {'Prompt':<12} {'Tokens':>6}  {'Prefill tok/s':>14}  {'Decode tok/s':>13}")
        print(f"{'─'*70}")
        for r in results:
            print(f"  {r['prompt']:<12} {r['tokens']:>6}  "
                  f"{r['prefill_toks']:>14.1f}  {r['decode_toks']:>13.2f}")
        print(f"{'='*70}\n")

        return {"model": model_name, "load_s": load_s, "num_devices": num_devices, "results": results}

    finally:
        if device is not full_mesh:
            ttnn.close_mesh_device(device)  # release submesh devices back to full_mesh


def main():
    parser = argparse.ArgumentParser(description="Benchmark Granite models")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    try:
        all_results = [run_bench(m, full_mesh, args.decode_tokens) for m in models]
    finally:
        ttnn.close_mesh_device(full_mesh)

    for result in all_results:
        out_path = f"bench_results_{result['model']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
