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
    """Generate text with HuggingFace model.

    Args:
        prompts: Single prompt string or list of prompts
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Batch size for processing
    """
    print("\n" + "="*70)
    print("HUGGINGFACE MODEL")
    print("="*70)

    # Handle single prompt or list
    if isinstance(prompts, str):
        prompts = [prompts]

    # Repeat prompts to match batch size if needed
    if len(prompts) < batch_size:
        prompts = prompts * batch_size
        prompts = prompts[:batch_size]

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
    if batch_size == 1:
        print(f"Prompt: \"{prompts[0]}\"")
    else:
        print(f"First prompt: \"{prompts[0]}\"")
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

    # Decode all outputs
    texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    all_ids = [output.tolist() for output in outputs]

    total_tokens_generated = sum(len(output) - len(inputs['input_ids'][i])
                                  for i, output in enumerate(outputs))
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


def generate_tt(prompts, max_tokens=10, batch_size=1):
    """Generate text with TT-optimized model.

    Args:
        prompts: Single prompt string or list of prompts
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Batch size for processing
    """
    print("\n" + "="*70)
    print("TT-OPTIMIZED MODEL")
    print("="*70)

    # Handle single prompt or list
    if isinstance(prompts, str):
        prompts = [prompts]

    # Repeat prompts to match batch size if needed
    if len(prompts) < batch_size:
        prompts = prompts * batch_size
        prompts = prompts[:batch_size]

    # Initialize TTNN device
    device = ttnn.open_device(device_id=0)

    try:
        from tt_model.model import TTGraniteMoeHybridForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-h-1b')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("\nLoading TT model...")
        start_time = time.time()
        tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
            'ibm-granite/granite-4.0-h-1b',
            device,
            verbose=False
        )

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']

        print(f"\nBatch size: {batch_size}")
        print(f"Prompts: {len(prompts)}")
        if batch_size == 1:
            print(f"Prompt: \"{prompts[0]}\"")
        else:
            print(f"First prompt: \"{prompts[0]}\"")
        print(f"Generating {max_tokens} tokens per prompt...")

        # Timing breakdown
        load_time = time.time() - start_time

        gen_start = time.time()

        # Process each sample in batch sequentially (model currently batch_size=1)
        all_generated_ids = []
        all_texts = []
        total_tokens_generated = 0

        for batch_idx in range(batch_size):
            # Generate tokens for this sample
            generated_ids = input_ids[batch_idx].tolist()
            initial_length = len(generated_ids)

            for i in range(max_tokens):
                current_ids = torch.tensor([generated_ids], dtype=torch.long)

                # Forward pass
                logits = tt_model.forward(current_ids)

                # Get next token (greedy)
                next_token = logits[0, -1, :].argmax().item()
                generated_ids.append(next_token)

                # Stop if EOS
                if next_token == tokenizer.eos_token_id:
                    break

            # Reset cache for next sample
            if batch_idx < batch_size - 1:
                tt_model.reset_cache()

            all_generated_ids.append(generated_ids)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_texts.append(text)
            total_tokens_generated += len(generated_ids) - initial_length

        gen_time = time.time() - gen_start
        elapsed_time = time.time() - start_time

        tokens_per_sec = total_tokens_generated / gen_time if gen_time > 0 else 0

        print(f"\nOutput (first sample):\n{all_texts[0]}")
        if batch_size > 1:
            print(f"\n... ({batch_size - 1} more samples)")

        print(f"\nPerformance:")
        print(f"  Total time: {elapsed_time:.3f}s")
        print(f"    - Model load: {load_time:.3f}s")
        print(f"    - Generation: {gen_time:.3f}s")
        print(f"  Total tokens: {total_tokens_generated}")
        print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
        print(f"  Per-sample: {tokens_per_sec/batch_size:.2f} tokens/sec")
        print("\n" + "="*70)

        return all_texts, all_generated_ids, gen_time

    finally:
        ttnn.close_device(device)


def compare(prompts, max_tokens=10, batch_size=1):
    """Run both models and compare outputs.

    Args:
        prompts: Single prompt string or list of prompts
        max_tokens: Maximum tokens to generate per prompt
        batch_size: Batch size for processing
    """
    print("\n" + "="*70)
    print("COMPARING HF vs TT")
    print("="*70)

    # Run HF
    hf_texts, hf_ids_list, hf_time = generate_hf(prompts, max_tokens, batch_size)

    # Run TT
    tt_texts, tt_ids_list, tt_time = generate_tt(prompts, max_tokens, batch_size)

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    # Compare first sample in detail
    print(f"\nFirst sample comparison:")
    print(f"HF output: \"{hf_texts[0]}\"")
    print(f"TT output: \"{tt_texts[0]}\"")

    if hf_ids_list[0] == tt_ids_list[0]:
        print("\n✓ IDENTICAL! Both models generated the same tokens")
    else:
        print("\n✗ DIFFERENT! Token-by-token comparison:")
        max_len = max(len(hf_ids_list[0]), len(tt_ids_list[0]))
        for i in range(min(max_len, 20)):  # Show first 20 tokens
            hf_tok = hf_ids_list[0][i] if i < len(hf_ids_list[0]) else None
            tt_tok = tt_ids_list[0][i] if i < len(tt_ids_list[0]) else None
            match = "✓" if hf_tok == tt_tok else "✗"
            print(f"  Position {i}: HF={hf_tok}, TT={tt_tok} {match}")
        if max_len > 20:
            print(f"  ... ({max_len - 20} more positions)")

    # Check all samples
    if batch_size > 1:
        all_identical = all(hf_ids == tt_ids for hf_ids, tt_ids in zip(hf_ids_list, tt_ids_list))
        identical_count = sum(1 for hf_ids, tt_ids in zip(hf_ids_list, tt_ids_list) if hf_ids == tt_ids)
        print(f"\nBatch results: {identical_count}/{batch_size} samples identical")

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  HF time:  {hf_time:.3f}s")
    print(f"  TT time:  {tt_time:.3f}s")

    if tt_time < hf_time:
        speedup = hf_time / tt_time
        improvement = ((hf_time - tt_time) / hf_time) * 100
        print(f"  Speedup:  {speedup:.2f}x faster ({improvement:.1f}% improvement)")
    elif tt_time > hf_time:
        slowdown = tt_time / hf_time
        regression = ((tt_time - hf_time) / hf_time) * 100
        print(f"  Slowdown: {slowdown:.2f}x slower ({regression:.1f}% slower)")
    else:
        print(f"  No difference")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="TT-Granite text generation")
    parser.add_argument(
        '--model',
        choices=['hf', 'tt'],
        help='Model to use: hf (HuggingFace) or tt (TT-optimized)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run both models and compare outputs'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='The future of AI is',
        help='Input prompt (default: "The future of AI is")'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        help='Multiple prompts for batch processing'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=10,
        help='Maximum tokens to generate (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.compare:
        parser.error("Must specify either --model or --compare")

    # Determine prompts to use
    prompts = args.prompts if args.prompts else args.prompt

    if args.compare:
        compare(prompts, args.max_tokens, args.batch_size)
    elif args.model == 'hf':
        generate_hf(prompts, args.max_tokens, args.batch_size)
    elif args.model == 'tt':
        generate_tt(prompts, args.max_tokens, args.batch_size)


if __name__ == "__main__":
    main()
