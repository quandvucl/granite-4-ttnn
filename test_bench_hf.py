#!/usr/bin/env python3
"""
Benchmark Granite models on CPU/HuggingFace: model load time + prefill + decode.

Usage:
  python test_bench_hf.py                        # tiny + small, all prompts
  python test_bench_hf.py --model tiny           # tiny only
  python test_bench_hf.py --model small          # small only
  python test_bench_hf.py --decode-tokens 30     # more decode steps
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

MODELS = {
    "tiny": "ibm-granite/granite-4.0-h-tiny",
    "small": "ibm-granite/granite-4.0-h-small",
}

PROMPTS = {
    "short_8": "The capital of France is",
    "short_10": "The largest planet in our solar system is",
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


def load_model(model_id, device, compile=True):
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    load_kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
    if device == "cuda":
        load_kwargs["device_map"] = "cuda"
        load_kwargs["attn_implementation"] = "sdpa"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()

    if device == "cuda" and compile:
        model = torch.compile(model, dynamic=True, backend="inductor", options={
            "max_autotune": True,
            "triton.cudagraphs": False,
            "max_autotune_gemm": True,
            "coordinate_descent_tuning": True,
            "shape_padding": True,
        })
        torch.cuda.empty_cache()

    load_s = time.time() - t0
    return model, load_s


def _warmup_one(model, ids, decode_steps, device, HybridMambaAttentionDynamicCache):
    cache = HybridMambaAttentionDynamicCache(
        model.config, batch_size=1, dtype=model.dtype, device=device
    )
    pos = torch.arange(ids.shape[1], device=device)
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True,
                    cache_position=pos, position_ids=pos.unsqueeze(0))
        next_id = out.logits[0, -1, :].argmax().item()
        for step in range(decode_steps):
            t = torch.tensor([[next_id]], device=device)
            cp = torch.tensor([ids.shape[1] + step], device=device)
            out = model(t, past_key_values=cache, use_cache=True,
                        cache_position=cp, position_ids=cp.unsqueeze(0))
            next_id = out.logits[0, -1, :].argmax().item()
    del cache, out


def warmup_cuda(model, tokenizer, device, HybridMambaAttentionDynamicCache, decode_steps=20):
    """Compile prefill + decode paths across short and longer prompt shapes."""
    for prompt in ["Hi", "Once upon a time in a land far away"]:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        _warmup_one(model, ids, decode_steps, device, HybridMambaAttentionDynamicCache)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def pick_next(logits_raw, seen_ids, rep_processor, device):
    scores = logits_raw[0, -1, :].float().unsqueeze(0)
    vocab_size = scores.shape[1]
    valid = [tid for tid in seen_ids if tid < vocab_size]
    if valid:
        seen = torch.tensor([valid], dtype=torch.long, device=device)
        scores = rep_processor(seen, scores)
    return scores[0].argmax().item()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench_prompt(model, input_ids, decode_tokens, rep_processor, tokenizer, device, HybridMambaAttentionDynamicCache):
    actual_len = input_ids.shape[1]
    cache = HybridMambaAttentionDynamicCache(
        model.config, batch_size=1, dtype=model.dtype, device=device
    )
    cache_position = torch.arange(actual_len, device=device)

    with torch.no_grad():
        sync(device)
        t0 = time.time()
        out = model(input_ids, past_key_values=cache, use_cache=True,
                    cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
        sync(device)
        prefill_ms = (time.time() - t0) * 1000

    all_ids = input_ids[0].tolist()
    next_id = pick_next(out.logits, all_ids, rep_processor, device)
    all_ids.append(next_id)

    decode_times = []
    pos = actual_len
    next_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
    cache_position = torch.zeros(1, dtype=torch.long, device=device)

    for _ in range(decode_tokens):
        next_tensor[0, 0] = next_id
        cache_position[0] = pos
        with torch.no_grad():
            sync(device)
            t0 = time.time()
            out = model(next_tensor, past_key_values=cache, use_cache=True,
                        cache_position=cache_position, position_ids=cache_position.unsqueeze(0))
            sync(device)
            decode_times.append((time.time() - t0) * 1000)
        next_id = pick_next(out.logits, all_ids, rep_processor, device)
        all_ids.append(next_id)
        pos += 1
        if next_id == tokenizer.eos_token_id:
            break

    generated_ids = all_ids[actual_len:]

    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg_ms = sum(steady) / len(steady)
    return prefill_ms, 1000.0 / avg_ms, generated_ids


def print_summary(model_name, device, load_s, results):
    print("=" * 70)
    print(f"  SUMMARY: {model_name.upper()} (HuggingFace {device.upper()})")
    print(f"  Model load: {load_s:.1f} s")
    print("-" * 70)
    print("  Prompt         Tokens  TTFT (ms)  Decode tok/s")
    print("-" * 70)
    for r in results:
        print(f"  {r['prompt']}  tokens={r['tokens']}  ttft={r['ttft_ms']:.1f}ms  decode={r['decode_toks']:.2f} tok/s")
    print("=" * 70)


def run_bench(model_name, decode_tokens=DECODE_TOKENS, device="cpu", compile=True):
    model_id = MODELS[model_name]

    print(f"\n{'='*70}")
    print(f"  Model : {model_id}")
    print(f"  Device: {device.upper()}")
    print(f"  Decode : {decode_tokens} tokens per prompt")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, load_s = load_model(model_id, device, compile=compile)
    print(f"\nModel load: {load_s:.1f} s\n")

    # Deferred import: module is registered dynamically via trust_remote_code
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (  # noqa: PLC0415
        HybridMambaAttentionDynamicCache,
    )

    if device == "cuda":
        warmup_cuda(model, tokenizer, device, HybridMambaAttentionDynamicCache)

    rep_processor = RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY)
    results = []

    for prompt_label, prompt_text in PROMPTS.items():
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        actual_len = input_ids.shape[1]
        print(f"── {prompt_label}  (actual tokens: {actual_len}) ──────────────────")

        prefill_ms, tok_s, generated_ids = bench_prompt(
            model, input_ids, decode_tokens, rep_processor, tokenizer, device,
            HybridMambaAttentionDynamicCache,
        )
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"  Prompt   : {prompt_text}")
        print(f"  Response : {response}")
        print(f"  TTFT: {prefill_ms:.1f} ms  |  Decode: {tok_s:.2f} tok/s")

        results.append({
            "prompt": prompt_label,
            "tokens": actual_len,
            "ttft_ms": prefill_ms,
            "decode_toks": tok_s,
            "prompt_text": prompt_text,
            "response": response,
        })

    print_summary(model_name, device, load_s, results)
    return {"model": model_name, "backend": f"hf_{device}", "load_s": load_s, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Granite HuggingFace models")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU found")

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    for m in models:
        result = run_bench(m, args.decode_tokens, args.device, compile=not args.no_compile)
        out_path = f"bench_results_hf_{result['model']}_{args.device}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
