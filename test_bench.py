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
    "small": ("ibm-granite/granite-4.0-h-small", 8),
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
    1:  ttnn.MeshShape(1, 1),
    2:  ttnn.MeshShape(1, 2),
    4:  ttnn.MeshShape(1, 4),
    8:  ttnn.MeshShape(2, 4),
    16: ttnn.MeshShape(4, 4),
    32: ttnn.MeshShape(4, 8),
}


def run_bench(model_name, full_mesh, decode_tokens=DECODE_TOKENS, use_all_gather=True):
    from granite.model import TTGraniteMoeHybridForCausalLM
    from utils import to_torch_tensor

    model_id, requested_devices = MODELS[model_name]

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_kv_heads = hf_config.num_key_value_heads
    num_experts  = getattr(hf_config, "num_local_experts", 1)
    full_mesh_size = full_mesh.get_num_devices()

    # Device count is limited to the largest power-of-2 (or supported mesh shape)
    # that fits in the available mesh AND divides the expert count for EP sharding.
    # Attention runs replicated when num_devices > num_kv_heads — no KV sharding needed.
    # MoE effective_devices logic in moe_tt.py handles non-divisor counts gracefully.
    num_devices = min(requested_devices, full_mesh_size)
    # Snap to largest entry in MESH_SHAPE_MAP that does not exceed num_devices
    num_devices = max(k for k in MESH_SHAPE_MAP if k <= num_devices)

    print(f"\n{'='*70}")
    print(f"  Model  : {model_id}")
    print(f"  Devices: {num_devices}  (requested {requested_devices}, KV heads {num_kv_heads}, experts {num_experts})")
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
            mamba_chunk_size=256 if num_devices >= 4 else None,
            max_cache_length=512,
            # tiny: bfloat8_b gives +5-7% throughput at fast load (small weights).
            # small: bfloat16 avoids 375s CPU quantization of 2.9 GB expert weights.
            moe_weight_dtype=ttnn.bfloat8_b if num_devices <= 4 else ttnn.bfloat16,
            moe_use_all_gather=use_all_gather,
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

            # Decode loop — steps 1-2 are warmup (compile + kernel upload),
            # then capture trace; steps 3+ replay trace at full speed.
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

                # After 2 warmup steps, capture trace for subsequent steps
                if step == 1:
                    tt_model.capture_decode_trace()

                if next_id == tokenizer.eos_token_id:
                    break

            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  Prompt   : {prompt_text}")
            print(f"  Response : {response}")

            # Stats — skip steps 1-2 (warmup) and step 3 (first trace replay, slower due to alloc)
            steady = decode_times[3:] if len(decode_times) > 3 else decode_times
            avg_ms = sum(steady) / len(steady)
            tok_s  = 1000.0 / avg_ms

            print(f"  Prefill: {actual_len*1000/prefill_ms:.1f} tok/s  |  Decode: {tok_s:.2f} tok/s")
            t = getattr(tt_model, "last_layer_family_timing", None)
            if t and t.get("seq_len") == 1:
                total_ms = t["layer_total"] * 1000
                print(f"    timing → attn={t['attention_total']*1000:.1f}ms  "
                      f"mamba={t['mamba_decode_total']*1000:.1f}ms  "
                      f"mlp={t['mlp_total']*1000:.1f}ms  "
                      f"layers={total_ms:.1f}ms")

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
    parser.add_argument("--mesh", default="8x4", help="Mesh shape, e.g. 8x4 (default) or 1x1")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    rows, cols = (int(x) for x in args.mesh.split("x"))
    use_fabric = rows * cols > 1
    fabric_ok = False
    # Galaxy fabric requires the full 8x4 mesh to be opened — fabric sync is system-wide.
    # Always open full mesh, then carve out the requested submesh inside run_bench.
    GALAXY_SHAPE = ttnn.MeshShape(8, 4)
    full_mesh = None
    if use_fabric:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            full_mesh = ttnn.open_mesh_device(mesh_shape=GALAXY_SHAPE, trace_region_size=268435456)
            fabric_ok = True
            print("[info] fabric enabled")
        except Exception as e:
            print(f"[warn] fabric open failed ({type(e).__name__}), retrying without fabric")
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            full_mesh = ttnn.open_mesh_device(mesh_shape=GALAXY_SHAPE)
    if full_mesh is None:
        full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    all_results = []
    try:
        all_results = [run_bench(m, full_mesh, args.decode_tokens, use_all_gather=fabric_ok) for m in models]
        ttnn.synchronize_device(full_mesh)
    finally:
        for submesh in full_mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(full_mesh)
        if use_fabric:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
        del full_mesh

    for result in all_results:
        out_path = f"bench_results_{result['model']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
