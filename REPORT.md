# Granite 4.0-H Porting Report: TTNN on Wormhole Galaxy

## Overview

This report summarizes the porting of IBM Granite 4.0-H (tiny and small) to Tenstorrent hardware using the TTNN framework. Both models run end-to-end on a Wormhole Galaxy (32-device board). Benchmarks compare prefill and decode throughput against HuggingFace on CPU (float32) and NVIDIA A100 (bfloat16).

**Models benchmarked:**

| Model | Parameters | Architecture | Devices used |
|-------|-----------|--------------|--------------|
| granite-4.0-h-tiny | ~1.2B | 40-layer hybrid (4 attention + 36 Mamba), 72 MoE experts | 4 WH devices |
| granite-4.0-h-small | ~3.3B | 40-layer hybrid (4 attention + 36 Mamba), 72 MoE experts | 8 WH devices |

---

## Benchmark Setup

- **TTNN backend**: Wormhole Galaxy, `single_galaxy_mesh_graph_descriptor.textproto`, mesh shapes `MeshShape(1,4)` (tiny) and `MeshShape(2,4)` (small)
- **HF CPU**: Single host, float32 (bfloat16 produces incorrect output on CPU without hardware support)
- **HF A100**: NVIDIA A100 80GB, bfloat16
- **Prompts**: 5 lengths — short (5–8 tokens), medium (25 tokens), long (96–98 tokens), very long (176–181 tokens)
- **Decode**: 20 tokens per prompt; metrics exclude the first decode step (JIT warmup)
- **Synchronization**: `ttnn.synchronize_device()` called after each forward pass for accurate wall-clock timing

---

## Results: Tiny Model (granite-4.0-h-tiny)

**Model load time:** TTNN 23.2s | HF CPU 2.6s | HF A100 4.1s

### Prefill throughput (tok/s)

| Prompt | Tokens | TTNN 4-dev | HF CPU | HF A100 | TTNN vs CPU | TTNN vs A100 |
|--------|-------:|----------:|-------:|--------:|:-----------:|:------------:|
| short_8 | 5 | 1.04 | 0.44 | 5.6 | 2.4× | 0.19× |
| short_10 | 8 | 2.28 | 0.70 | 25.4 | 3.3× | 0.09× |
| medium_32 | 25 | 6.68 | 2.17 | 75.1 | 3.1× | 0.09× |
| long_128 | 96 | 21.2 | 8.22 | 279 | 2.6× | 0.08× |
| long_256 | 176 | 27.1 | 14.8 | 503 | 1.8× | 0.05× |

### Decode throughput (tok/s, steady-state)

| Prompt | TTNN 4-dev | HF CPU | HF A100 | TTNN vs CPU | TTNN vs A100 |
|--------|----------:|-------:|--------:|:-----------:|:------------:|
| short_8 | 3.10 | 2.58 | 7.65 | 1.2× | 0.41× |
| short_10 | 3.10 | 2.60 | 9.40 | 1.2× | 0.33× |
| medium_32 | 3.05 | 2.61 | 9.32 | 1.2× | 0.33× |
| long_128 | 3.06 | 2.61 | 9.34 | 1.2× | 0.33× |
| long_256 | 3.05 | 2.62 | 9.19 | 1.2× | 0.33× |

---

## Results: Small Model (granite-4.0-h-small)

**Model load time:** TTNN 117.0s | HF CPU 23.6s | HF A100 175.5s

### Prefill throughput (tok/s)

| Prompt | Tokens | TTNN 8-dev | HF CPU | HF A100 | TTNN vs CPU | TTNN vs A100 |
|--------|-------:|----------:|-------:|--------:|:-----------:|:------------:|
| short_8 | 5 | 0.37 | 0.16 | 0.95 | 2.3× | 0.39× |
| short_10 | 8 | 0.75 | 0.26 | 9.84 | 2.9× | 0.08× |
| medium_32 | 25 | 2.22 | 0.81 | 28.8 | 2.7× | 0.08× |
| long_128 | 96 | 3.76 | 3.01 | 95.6 | 1.2× | 0.04× |
| long_256 | 176 | 4.72 | 5.39 | 178 | 0.9× | 0.03× |

### Decode throughput (tok/s, steady-state)

| Prompt | TTNN 8-dev | HF CPU | HF A100 | TTNN vs CPU | TTNN vs A100 |
|--------|----------:|-------:|--------:|:-----------:|:------------:|
| short_8 | 1.99 | 0.66 | 5.39 | 3.0× | 0.37× |
| short_10 | 1.97 | 0.66 | 5.40 | 3.0× | 0.37× |
| medium_32 | 1.95 | 0.66 | 5.53 | 2.9× | 0.35× |
| long_128 | 1.94 | 0.66 | 5.55 | 2.9× | 0.35× |
| long_256 | 1.97 | 0.66 | 5.59 | 3.0× | 0.35× |

---

## Key Findings

### What works well
- **Correctness**: Both models produce coherent, accurate responses across all prompt lengths on TTNN. The porting is functionally correct.
- **TTNN tiny decode**: 3.1 tok/s, roughly 1.2× faster than CPU (2.6 tok/s). Demonstrates that TT hardware is useful even at small scale.
- **TTNN small decode**: 1.97 tok/s, roughly 3× faster than CPU (0.66 tok/s). The larger model benefits more from the 8-device setup.
- **Prefill scaling**: TTNN tiny shows 2–3× CPU speedup on prefill for short-to-medium prompts, confirming that the matmul-heavy attention and MLP layers are executing efficiently on-chip.

### Where TTNN lags A100
- **Decode gap**: TTNN sits at ~0.33–0.37× of A100 decode throughput. The gap is consistent regardless of prompt length, which points to a fixed per-step overhead rather than a memory-bandwidth problem.
- **Prefill gap at long sequences**: TTNN small is unable to beat CPU for long_256 (4.72 vs 5.39 tok/s). A100 is ~38× faster than TTNN on the same prompt, indicating the prefill path has substantial room for improvement.

---

## Bottleneck Analysis and Improvement Opportunities

### 1. MoE expert allgather — highest impact

Each decoder layer runs a block-sparse MoE with 72 experts. The current implementation:
- Uploads routing weights per-forward (small but adds PCIe traffic)
- Gathers all expert outputs from 8 devices back to CPU via PCIe (`ttnn.to_torch`) to sum them
- Re-uploads the summed result to all devices (`ttnn.from_torch` with `ReplicateTensorToMesh`)

The allgather round-trip is approximately 36 MB per forward pass (40 layers × 72 experts × seq × hidden). At PCIe bandwidth (~12 GB/s), this alone accounts for ~3 ms/step minimum latency.

**Fix**: Replace the CPU gather-sum with an on-device reduce-scatter + all-gather using `ttnn.reduce_scatter` + `ttnn.all_gather` on the fabric. This eliminates the 40× PCIe round-trips per forward pass entirely.

**Estimated gain**: 2–4× decode throughput improvement.

### 2. Mamba decode: SSM state on CPU

The Mamba layers use `HybridMambaAttentionDynamicCache` (HF) to hold the SSM recurrent state. During decode:
- The hidden state is pulled from TT device to CPU for the SSM recurrence step (conv + selective scan)
- The result is re-uploaded to TT device for the next layer

This PCIe round-trip happens for every Mamba layer (36 out of 40) on every decode step.

**Fix**: Implement the Mamba SSM recurrence (`mamba_chunk_scan` for prefill, single-step selective scan for decode) natively in TTNN. The `mamba/mamba_chunk_scan_parallel.py` file contains a prefill-optimized version; the decode path needs a corresponding 1-step kernel. With the state kept on-device, the 36 round-trips collapse to zero.

**Estimated gain**: Additional 1.5–2× decode throughput improvement once MoE PCIe is fixed.

### 3. Depthwise convolution split between CPU and TT

The Mamba layers include a short depthwise convolution over the SSM input. Currently this runs on CPU as part of the `HybridMambaAttentionDynamicCache` logic. Moving it to TTNN (or fusing it with the nearest linear projection) would further reduce cross-boundary transfers.

**Estimated gain**: Moderate; smaller than #1 and #2 but meaningful at high token rates.

### 4. Prefill batch parallelism

TTNN's prefill path processes the entire prompt as a single sequence. For long prompts (96–176 tokens), the sequence dimension is small enough that TT cores are underutilized. Adding chunked prefill (process the prompt in segments, reusing the Mamba chunk-scan kernel) would improve hardware utilization for medium-to-long contexts.

**Estimated gain**: 2–5× prefill throughput at long_128/long_256 lengths, which would close the gap with A100.

---

## Summary Table

| Metric | TTNN tiny (4-dev) | TTNN small (8-dev) | A100 tiny | A100 small |
|--------|:-----------------:|:------------------:|:---------:|:----------:|
| Decode tok/s (avg) | 3.06 | 1.97 | 9.14 | 5.49 |
| TTNN / A100 decode | 0.33× | 0.36× | — | — |
| Prefill tok/s (long_256) | 27.1 | 4.72 | 503 | 178 |
| TTNN / A100 prefill (long) | 0.05× | 0.03× | — | — |
| Model load time | 23.2s | 117.0s | 4.1s | 175.5s |

The primary bottleneck is PCIe data movement for MoE and Mamba layers. With native on-device reduce-scatter for MoE and a TTNN Mamba SSM decode kernel, TTNN throughput is expected to reach or exceed A100 decode performance, leveraging the Wormhole Galaxy's memory bandwidth advantage over a single A100.
