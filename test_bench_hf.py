#!/usr/bin/env python3
"""
Benchmark Granite models on CPU/HuggingFace: model load time + prefill + decode.

Usage:
  python generate_with_warmup.py                        # tiny + small, all prompts
  python generate_with_warmup.py --model tiny           # tiny only
  python generate_with_warmup.py --model small          # small only
  python generate_with_warmup.py --decode-tokens 30     # more decode steps
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

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


@dataclass
class BenchRunner:
    model: Any
    tokenizer: Any
    device: str
    cache_class: Any  # HybridMambaAttentionDynamicCache
    rep_processor: RepetitionPenaltyLogitsProcessor


def load_model(model_id: str, device: str, compile: bool = True):
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    load_kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
    if device == "cuda":
        load_kwargs["device_map"] = "cuda"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()

    if compile:
        if device == "cuda":
            model = torch.compile(model, dynamic=True, backend="inductor", options={
                "max_autotune": True,
                "triton.cudagraphs": False,
                "max_autotune_gemm": True,
                "coordinate_descent_tuning": True,
                "shape_padding": True,
            })
            torch.cuda.empty_cache()
        else:
            model = torch.compile(model, dynamic=True, backend="inductor", options={
                "cpp_wrapper": True,
            })

    return model, time.time() - t0


def _warmup_one(runner: BenchRunner, ids: torch.Tensor, decode_steps: int):
    cache = runner.cache_class(
        runner.model.config, batch_size=1, dtype=runner.model.dtype, device=runner.device
    )
    pos = torch.arange(ids.shape[1], device=runner.device)
    with torch.no_grad():
        out = runner.model(ids, past_key_values=cache, use_cache=True,
                           cache_position=pos, position_ids=pos.unsqueeze(0))
        next_id = out.logits[0, -1, :].argmax().item()
        for step in range(decode_steps):
            t = torch.tensor([[next_id]], device=runner.device)
            cp = torch.tensor([ids.shape[1] + step], device=runner.device)
            out = runner.model(t, past_key_values=cache, use_cache=True,
                               cache_position=cp, position_ids=cp.unsqueeze(0))
            next_id = out.logits[0, -1, :].argmax().item()
    del cache, out


def warmup_cuda(runner: BenchRunner, decode_steps: int = 20):
    """Compile prefill + decode paths across short and longer prompt shapes."""
    for prompt in ["Hi", "Once upon a time in a land far away"]:
        ids = runner.tokenizer(prompt, return_tensors="pt")["input_ids"].to(runner.device)
        _warmup_one(runner, ids, decode_steps)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def warmup_cpu(runner: BenchRunner, decode_steps: int = 20):
    """Trigger torch.compile JIT before benchmarking so prompt TTFTs are not inflated."""
    for prompt in ["Hi", "Once upon a time in a land far away"]:
        ids = runner.tokenizer(prompt, return_tensors="pt")["input_ids"]
        _warmup_one(runner, ids, decode_steps)


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def bench_prompt(runner: BenchRunner, input_ids: torch.Tensor, decode_tokens: int):
    actual_len = input_ids.shape[1]
    cache = runner.cache_class(
        runner.model.config, batch_size=1, dtype=runner.model.dtype, device=runner.device
    )
    cache_position = torch.arange(actual_len, device=runner.device)

    # Prefill
    sync(runner.device)
    t0 = time.time()
    with torch.no_grad():
        out = runner.model(input_ids, past_key_values=cache, use_cache=True,
                           cache_position=cache_position,
                           position_ids=cache_position.unsqueeze(0))
    sync(runner.device)
    prefill_ms = (time.time() - t0) * 1000

    # Pre-allocate seen_ids buffer — avoids list→tensor each step
    seen_buf = torch.empty(1, actual_len + decode_tokens + 1, dtype=torch.long, device=runner.device)
    seen_buf[0, :actual_len] = input_ids[0]
    seen_len = actual_len

    scores = runner.rep_processor(seen_buf[:, :seen_len],
                                  out.logits[0, -1, :].float().unsqueeze(0))
    next_token = scores[0].argmax().view(1, 1)
    seen_buf[0, seen_len] = next_token[0, 0]
    seen_len += 1

    cache_position = torch.zeros(1, dtype=torch.long, device=runner.device)

    # Per-step timing, skip first step (matches ttnn methodology)
    decode_times = []
    with torch.no_grad():
        for step in range(decode_tokens):
            cache_position[0] = actual_len + step
            sync(runner.device)
            t0 = time.time()
            out = runner.model(next_token, past_key_values=cache, use_cache=True,
                               cache_position=cache_position,
                               position_ids=cache_position.unsqueeze(0))
            sync(runner.device)
            decode_times.append((time.time() - t0) * 1000)
            scores = runner.rep_processor(seen_buf[:, :seen_len],
                                          out.logits[0, -1, :].float().unsqueeze(0))
            next_token = scores[0].argmax().view(1, 1)
            seen_buf[0, seen_len] = next_token[0, 0]
            seen_len += 1

    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    tok_s = 1000.0 / (sum(steady) / len(steady))

    generated_ids = seen_buf[0, actual_len:seen_len].tolist()
    if runner.tokenizer.eos_token_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(runner.tokenizer.eos_token_id)]

    return prefill_ms, tok_s, generated_ids


def print_summary(model_name: str, device: str, load_s: float, results: list):
    print(f"{'='*70}")
    print(f"  SUMMARY: {model_name.upper()} (HuggingFace {device.upper()})")
    print(f"  Model load: {load_s:.1f} s")
    print(f"{'─'*70}")
    print("  Prompt        Tokens  TTFT (ms)  Decode tok/s")
    print(f"{'─'*70}")
    for r in results:
        print(f"  {r['prompt']}  tokens={r['tokens']}  ttft={r['ttft_ms']:.1f}ms  decode={r['decode_toks']:.2f} tok/s")
    print(f"{'='*70}\n")


def run_bench(model_name: str, decode_tokens: int = DECODE_TOKENS, device: str = "cpu", compile: bool = True):
    model_id = MODELS[model_name]

    print(f"\n{'='*70}")
    print(f"  Model : {model_id}")
    print(f"  Device: {device.upper()}")
    print(f"  Decode: {decode_tokens} tokens per prompt")
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

    runner = BenchRunner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cache_class=HybridMambaAttentionDynamicCache,
        rep_processor=RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY),
    )

    if device == "cuda":
        warmup_cuda(runner)
    elif compile:
        warmup_cpu(runner)

    results = []
    for prompt_label, prompt_text in PROMPTS.items():
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        actual_len = input_ids.shape[1]
        print(f"── {prompt_label}  (actual tokens: {actual_len}) ──────────────────")

        prefill_ms, tok_s, generated_ids = bench_prompt(runner, input_ids, decode_tokens)
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
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Enable torch.compile (default: off)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU found")

    compile = args.compile

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    for m in models:
        result = run_bench(m, args.decode_tokens, args.device, compile=compile)
        compile_suffix = "_compile" if compile else ""
        out_path = f"bench_results_hf_{result['model']}_{args.device}{compile_suffix}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
