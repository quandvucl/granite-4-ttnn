#!/usr/bin/env python3
"""
TT-Granite text generation script.

Usage:
    # Single prompt
    python generate.py --model hf                          # Run HuggingFace model
    python generate.py --model tt                          # Run TT-optimized model
    python generate.py --compare                           # Run both and compare

    # Batch processing
    python generate.py --compare --batch-size 4            # Compare with batch size 4
    python generate.py --compare --batch-size 8 --max-tokens 20

    # Custom prompts
    python generate.py --compare --prompt "Once upon a time"
    python generate.py --compare --prompts "Hello" "How are you?" "What is AI?" --batch-size 3
"""

import argparse
import torch
import ttnn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


def generate_hf(prompts, max_tokens=10, batch_size=1):
    """Generate text with HuggingFace model."""
    print("\n" + "="*70)
    print("HUGGINGFACE MODEL")
    print("="*70)

    if isinstance(prompts, str):
        prompts = [prompts]
    if len(prompts) < batch_size:
        prompts = (prompts * batch_size)[:batch_size]

    tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-h-1b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        'ibm-granite/granite-4.0-h-1b',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    print(f"\nBatch size: {batch_size}")
    print(f"Prompts: {len(prompts)}")
    print(f"Prompt: \"{prompts[0]}\"")
    print(f"Generating {max_tokens} tokens per prompt...")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed_time = time.time() - start_time

    texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    all_ids = [output.tolist() for output in outputs]

    total_tokens_generated = sum(
        len(output) - len(inputs['input_ids'][i])
        for i, output in enumerate(outputs)
    )
    tokens_per_sec = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0

    print(f"\nOutput (first sample):\n{texts[0]}")
    if batch_size > 1:
        print(f"\n... ({batch_size - 1} more samples)")
    print(f"\nPerformance:")
    print(f"  Time: {elapsed_time:.3f}s")
    print(f"  Total tokens: {total_tokens_generated}")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"  Per-sample: {tokens_per_sec/batch_size:.2f} tokens/sec")
    print("\n" + "="*70)

    return texts, all_ids, elapsed_time


# Full system mesh shape — must match physical topology (32 cards = 4x8)
_SYSTEM_MESH_SHAPE = ttnn.MeshShape(4, 8)
_SYSTEM_NUM_DEVICES = 32

def _open_device(num_devices: int):
    """
    Open TT device(s).

    When using fabric (num_devices > 1), ttnn requires opening the FULL system
    mesh first, then creating a submesh of the desired size. Opening a subset
    directly is not supported when fabric is active.

    For single device, open just a 1x1 mesh (no fabric needed).
    """
    mesh_shape_map = {
        1:  ttnn.MeshShape(1, 1),
        2:  ttnn.MeshShape(1, 2),
        4:  ttnn.MeshShape(2, 2),
        8:  ttnn.MeshShape(2, 4),
        16: ttnn.MeshShape(4, 4),
        32: ttnn.MeshShape(4, 8),
    }
    target_shape = mesh_shape_map.get(num_devices, ttnn.MeshShape(1, num_devices))

    if num_devices == 1:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        print(f"Opened 1 device as MeshDevice(1x1)")
        return device, None
    else:
        # Must open full system mesh first when fabric is active
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        full_mesh = ttnn.open_mesh_device(mesh_shape=_SYSTEM_MESH_SHAPE)
        print(f"Opened full system mesh ({_SYSTEM_MESH_SHAPE})")
        if num_devices == _SYSTEM_NUM_DEVICES:
            # Using all devices — no submesh needed
            return full_mesh, full_mesh
        else:
            # Create submesh of desired size
            submeshes = full_mesh.create_submeshes(target_shape)
            device = submeshes[0]
            print(f"Created submesh of shape {target_shape} ({num_devices} devices)")
            return device, full_mesh


def _close_device(device, all_devices=None):
    """Close device(s) opened by _open_device."""
    if all_devices is not None and all_devices is not device:
        # submesh case — close the full mesh (submesh is released automatically)
        ttnn.close_mesh_device(all_devices)
    elif all_devices is device:
        # full mesh was used directly
        ttnn.close_mesh_device(device)
    else:
        # single device 1x1 mesh
        ttnn.close_mesh_device(device)


def _warmup(tt_model, tokenizer, device):
    """
    Run one short forward pass to trigger kernel compilation before timing.
    Kernel compilation is a one-time cost per process but will corrupt benchmark
    numbers if included in the timed region.
    """
    print("Warming up (kernel compilation)...", end=" ", flush=True)
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    _ = tt_model.forward(warmup_ids)
    tt_model.reset_cache()
    print("done")


def generate_tt(prompts, max_tokens=10, batch_size=1, num_devices=1):
    """
    Generate text with TT-optimized model.

    Key correctness fix vs original:
    - Prefill: pass full prompt once → get first generated token
    - Decode:  pass ONLY the single new token each step (not the growing sequence)

    This is critical for performance. Passing the full sequence every step causes
    the model to re-run prefill (O(seq_len) work) instead of decode (O(1) work).
    """
    print("\n" + "="*70)
    print("TT-OPTIMIZED MODEL")
    print("="*70)

    if isinstance(prompts, str):
        prompts = [prompts]
    if len(prompts) < batch_size:
        prompts = (prompts * batch_size)[:batch_size]

    device, all_devices = _open_device(num_devices)

    try:
        from tt_model.model import TTGraniteMoeHybridForCausalLM

        tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-h-1b')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("\nLoading TT model...")
        load_start = time.time()
        tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
            'ibm-granite/granite-4.0-h-1b',
            device,
            verbose=False
        )
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")

        # Warmup: trigger kernel compilation before timing
        _warmup(tt_model, tokenizer, device)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']  # [batch, prompt_len]

        print(f"\nBatch size: {batch_size}")
        print(f"Prompts: {len(prompts)}")
        print(f"Prompt: \"{prompts[0]}\"")
        print(f"Generating {max_tokens} tokens per prompt...")

        all_generated_ids = []
        all_texts = []
        total_tokens_generated = 0

        gen_start = time.time()

        # NOTE: Sequential batch processing is a bottleneck for multi-device setups.
        # TODO: Implement data parallelism to process different samples on different devices.
        # With 32 devices, we could process up to 32 samples in parallel!
        for batch_idx in range(batch_size):
            prompt_ids = input_ids[batch_idx:batch_idx+1]  # [1, prompt_len]
            generated_ids = input_ids[batch_idx].tolist()

            # ── PREFILL ───────────────────────────────────────────────────
            # Pass the full prompt; cache is populated; we get logits for all
            # positions but only use the last one to pick the first new token.
            with torch.no_grad():
                logits = tt_model.forward(prompt_ids)          # [1, prompt_len, vocab]

            next_token = logits[0, -1, :].argmax().item()
            generated_ids.append(next_token)
            total_tokens_generated += 1

            # ── DECODE ────────────────────────────────────────────────────
            # Pass ONLY the single new token each step.
            # The model's cache_manager already holds KV state from prefill;
            # cache_manager.get_position() advances by 1 each call so positional
            # embeddings are computed correctly.
            for _ in range(max_tokens - 1):
                if next_token == tokenizer.eos_token_id:
                    break

                decode_ids = torch.tensor([[next_token]], dtype=torch.long)  # [1, 1]
                with torch.no_grad():
                    logits = tt_model.forward(decode_ids)      # [1, 1, vocab]

                next_token = logits[0, -1, :].argmax().item()
                generated_ids.append(next_token)
                total_tokens_generated += 1

            # Reset cache before processing next sample
            if batch_idx < batch_size - 1:
                tt_model.reset_cache()

            all_generated_ids.append(generated_ids)
            all_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

        gen_time = time.time() - gen_start
        elapsed_time = load_time + gen_time

        # Only count decode tokens for throughput (prefill tokens are amortised)
        decode_tokens = total_tokens_generated
        tokens_per_sec = decode_tokens / gen_time if gen_time > 0 else 0

        print(f"\nOutput (first sample):\n{all_texts[0]}")
        if batch_size > 1:
            print(f"\n... ({batch_size - 1} more samples)")

        print(f"\nPerformance:")
        print(f"  Total time:  {elapsed_time:.3f}s")
        print(f"    - Load:    {load_time:.3f}s")
        print(f"    - Generate:{gen_time:.3f}s")
        print(f"  Total tokens generated: {total_tokens_generated}")
        print(f"  Throughput:  {tokens_per_sec:.2f} tokens/sec")
        print(f"  Per-sample:  {tokens_per_sec/batch_size:.2f} tokens/sec")
        print("\n" + "="*70)

        return all_texts, all_generated_ids, gen_time

    finally:
        _close_device(device, all_devices)


def compare(prompts, max_tokens=10, batch_size=1, num_devices=1):
    """Run both models and compare outputs."""
    print("\n" + "="*70)
    print("COMPARING HF vs TT")
    print("="*70)

    hf_texts, hf_ids_list, hf_time = generate_hf(prompts, max_tokens, batch_size)
    tt_texts, tt_ids_list, tt_time = generate_tt(prompts, max_tokens, batch_size, num_devices)

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    print(f"\nFirst sample comparison:")
    print(f"HF output: \"{hf_texts[0]}\"")
    print(f"TT output: \"{tt_texts[0]}\"")

    if hf_ids_list[0] == tt_ids_list[0]:
        print("\n✓ IDENTICAL! Both models generated the same tokens")
    else:
        print("\n✗ DIFFERENT! Token-by-token comparison:")
        max_len = max(len(hf_ids_list[0]), len(tt_ids_list[0]))
        for i in range(min(max_len, 20)):
            hf_tok = hf_ids_list[0][i] if i < len(hf_ids_list[0]) else None
            tt_tok = tt_ids_list[0][i] if i < len(tt_ids_list[0]) else None
            match = "✓" if hf_tok == tt_tok else "✗"
            print(f"  Position {i}: HF={hf_tok}, TT={tt_tok} {match}")
        if max_len > 20:
            print(f"  ... ({max_len - 20} more positions)")

    if batch_size > 1:
        identical_count = sum(1 for h, t in zip(hf_ids_list, tt_ids_list) if h == t)
        print(f"\nBatch results: {identical_count}/{batch_size} samples identical")

    print(f"\nPerformance Comparison:")
    print(f"  HF time: {hf_time:.3f}s")
    print(f"  TT time: {tt_time:.3f}s")

    if tt_time < hf_time:
        speedup = hf_time / tt_time
        print(f"  Speedup: {speedup:.2f}x faster ({((hf_time-tt_time)/hf_time)*100:.1f}% improvement)")
    elif tt_time > hf_time:
        slowdown = tt_time / hf_time
        print(f"  Slowdown: {slowdown:.2f}x slower ({((tt_time-hf_time)/hf_time)*100:.1f}% regression)")
    else:
        print(f"  No difference")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="TT-Granite text generation")
    parser.add_argument('--model', choices=['hf', 'tt'])
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--prompt', type=str, default='The future of AI is')
    parser.add_argument('--prompts', type=str, nargs='+')
    parser.add_argument('--max-tokens', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-devices', type=int, default=1)

    args = parser.parse_args()

    if not args.model and not args.compare:
        parser.error("Must specify either --model or --compare")

    prompts = args.prompts if args.prompts else args.prompt

    if args.compare:
        compare(prompts, args.max_tokens, args.batch_size, args.num_devices)
    elif args.model == 'hf':
        generate_hf(prompts, args.max_tokens, args.batch_size)
    elif args.model == 'tt':
        generate_tt(prompts, args.max_tokens, args.batch_size, args.num_devices)


if __name__ == "__main__":
    main()