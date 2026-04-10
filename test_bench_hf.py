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
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

MODELS = {
    "tiny":  "ibm-granite/granite-4.0-h-tiny",
    "small": "ibm-granite/granite-4.0-h-small",
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


def run_bench(model_name, decode_tokens=DECODE_TOKENS, device="cpu"):
    model_id = MODELS[model_name]

    print(f"\n{'='*70}")
    print(f"  Model : {model_id}")
    print(f"  Device: {device.upper()}")
    print(f"  Decode : {decode_tokens} tokens per prompt")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t_load0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()
    load_s = time.time() - t_load0
    print(f"\nModel load: {load_s:.1f} s\n")

    rep_processor = RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY)
    results = []

    for prompt_label, prompt_text in PROMPTS.items():
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        actual_len = input_ids.shape[1]

        print(f"── {prompt_label}  (actual tokens: {actual_len}) ──────────────────")

        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            HybridMambaAttentionDynamicCache,
        )
        past_key_values = HybridMambaAttentionDynamicCache(
            model.config, batch_size=1, dtype=model.dtype, device=device
        )

        cache_position = torch.arange(actual_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        with torch.no_grad():
            # Prefill
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(input_ids, past_key_values=past_key_values,
                        use_cache=True, cache_position=cache_position,
                        position_ids=position_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            prefill_ms = (time.time() - t0) * 1000
            logits = out.logits

        def pick_next(logits_raw, seen_ids):
            scores = logits_raw[0, -1, :].float().unsqueeze(0)
            vocab_size = scores.shape[1]
            valid = [tid for tid in seen_ids if tid < vocab_size]
            if valid:
                seen = torch.tensor([valid], dtype=torch.long, device=device)
                scores = rep_processor(seen, scores)
            return scores[0].argmax().item()

        context_ids = input_ids[0].tolist()
        next_id = pick_next(logits, context_ids)

        decode_times = []
        generated_ids = [next_id]
        pos = actual_len
        for _ in range(decode_tokens):
            next_tensor = torch.tensor([[next_id]], device=device)
            cache_position = torch.tensor([pos], device=device)
            position_ids = cache_position.unsqueeze(0)
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                out = model(next_tensor, past_key_values=past_key_values,
                            use_cache=True, cache_position=cache_position,
                            position_ids=position_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                step_ms = (time.time() - t0) * 1000
            decode_times.append(step_ms)
            next_id = pick_next(out.logits, context_ids + generated_ids)
            generated_ids.append(next_id)
            pos += 1
            if next_id == tokenizer.eos_token_id:
                break

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"  Prompt   : {prompt_text}")
        print(f"  Response : {response}")

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

    print(f"{'='*70}")
    print(f"  SUMMARY: {model_name.upper()} (HuggingFace CPU)")
    print(f"  Model load: {load_s:.1f} s")
    print(f"{'─'*70}")
    print(f"  {'Prompt':<12} {'Tokens':>6}  {'Prefill tok/s':>14}  {'Decode tok/s':>13}")
    print(f"{'─'*70}")
    for r in results:
        print(f"  {r['prompt']:<12} {r['tokens']:>6}  "
              f"{r['prefill_toks']:>14.1f}  {r['decode_toks']:>13.2f}")
    print(f"{'='*70}\n")

    return {"model": model_name, "backend": f"hf_{device}", "load_s": load_s, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Granite HuggingFace models")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU found")

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    for m in models:
        result = run_bench(m, args.decode_tokens, args.device)
        out_path = f"bench_results_hf_{result['model']}_{args.device}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
