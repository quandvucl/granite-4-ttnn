# Performance Improvement Plan: Granite 4H on Wormhole Galaxy

## Baselines

### Original baseline (pre-optimisation)
| Model | Decode (tok/s) | Prefill long_256 (tok/s) |
|-------|---------------:|-------------------------:|
| tiny (4-dev) | 3.06 | 27.1 |
| small (8-dev) | 1.97 | 4.72 |

### Current baseline (2026-07-05, fresh fabric session, 4 devices, fabric enabled)
| Prompt | Tokens | Prefill tok/s | Decode tok/s |
|--------|-------:|--------------:|-------------:|
| short_8 | 5 | 2.6 | 8.17 |
| short_10 | 8 | 10.7 | 8.14 |
| medium_32 | 25 | 27.9 | 8.17 |
| long_128 | 96 | 83.1 | 8.02 |
| long_256 | 176 | 105.1 | 8.07 |

Model load: 97.7s. Note: 7.0 tok/s observed after multiple resets in same session — hardware session variance, not a code regression.

### Measured decode breakdown (per token, 4 devices, tiny model, fresh session)
```
Attention total:    ~2.4 ms   (20 layers, replicated — only 2% of decode!)
Mamba total:        ~49 ms   (20 layers, already TP — 41%)
MLP total:          ~60 ms   (20 layers, REPLICATED — 50%)
Other/overhead:     ~10 ms
Total:             ~119 ms → ~8.1 tok/s
```

### Parallelism status (tiny, 4 devices)
| Component | Parallelism | Status |
|-----------|-------------|--------|
| Embedding (1536-dim lookup) | Replicated | 1× |
| Attention (20 layers, GQA 12H/4KV) | Replicated | 1× |
| Mamba (20 layers) | Column-parallel + all_gather | 4× |
| MoE experts (64 experts, top-6) | Expert-parallel + all_gather | 4× |
| Shared MLP (512 intermediate) | **Replicated** | 1× |
| LM head (1536 → vocab) | Replicated | 1× |

---

## Completed Work

### ✅ Chunk Size Tuning (2026-07-04)

**Conclusion**: `mamba_chunk_size=256` is optimal for Wormhole.

Sweep results (tiny, 4 devices):

| chunk_size | long_128 (96 tok) prefill | long_256 (176 tok) prefill | avg prefill | decode |
|-----------|:-------------------------:|:--------------------------:|:-----------:|:------:|
| 192 | 41.4 tok/s | 96.2 tok/s | 68.8 | ~5 tok/s |
| **256** | **140.6 tok/s** | **143.5 tok/s** | **142.1** | 7.3 tok/s |
| 320 | 139.3 tok/s | 144.6 tok/s | 141.9 | 7.2 tok/s |
| 384 | 128.3 tok/s | 145.3 tok/s | 136.8 | 7.1 tok/s |
| 512 | 135.5 tok/s | 145.3 tok/s | 140.4 | 7.1 tok/s |

`chunk_size=256` already set as default in `test_bench.py`. No config change needed.
`chunk_size=192` anomaly on long_128: 96 < 192 so Mamba uses one small padded chunk.

**Key learning**: chunk_size must be ≥ longest prompt or attention splits across chunks, hitting the multi-chunk `is_causal=False` SDPA path which hangs on WH. Sweep sizes must all exceed max prompt length.

---

### ✅ Fusion 2 — Batched Conv1d (tested, NOT applied)

> **TESTED AND NOT APPLIED** (2026-06-17). Benchmarked on `test_trace.py`:
> - Old shift-register 10-dispatch: **0.429ms no-trace → 0.061ms traced** (7.0×)
> - New stacked 3-dispatch:         **0.645ms no-trace → 0.330ms traced** (2.0×)
>
> The stacked approach is **5.4× slower under trace**. The 10 separate `[1,1,C,1]`
> ops (6 KB each) are fully dispatch-overhead-bound — trace collapses them to near-zero.
> **Skip. The current shift-register traces optimally.**

---

### ✅ Fusion 4 — Gated SiLU Fused Elementwise (reverted)

> **REVERTED** (2026-06-18). `input_tensor_b_activations` computes `SiLU(y * gate)`
> (post-multiply) rather than the required `y * SiLU(gate)` (pre-multiply on gate).
> Bench output became nonsensical. **Do not attempt without a unit test.**

---

### ✅ Split-trace attempt (abandoned, 2026-06-18)

Attempted to trace the inner decode body between the two `all_gather` calls.
- Result: 6.3–6.5 tok/s with trace vs 8.0 tok/s baseline
- Root cause: 3 extra `ttnn.copy` calls per conv layer × 36 layers = 108 extra dispatches.
  Trace savings (~15ms on Mamba) do not offset extra device work.
- **Recommended path**: wait for `all_gather`-in-trace support (tt-metal#26649), then the
  entire `forward_decode` can be traced without any extra device work.

---

## Root Cause Analysis

### Wormhole Memory Hierarchy
```
SFPU registers   — in-core, zero traffic, ~64 bf16 per lane
L1 CBs           — within one kernel, 0 DRAM traffic for intermediates
L1 SRAM (pinned) — 1.5 MB/core, 108 MB total per chip, survives across dispatches
DRAM             — 12 GB @ 256 GB/s per chip
Ethernet/NOC2    — 16 × 100 Gbps = 1.6 Tbps chip-to-chip (all_gather/reduce_scatter)
PCIe             — 12 GB/s to host, ~100 µs per op dispatch (the bottleneck)
```

### Theoretical vs actual
**Theoretical DRAM minimum: 2.6 ms/token. Actual: 119 ms/token. 97% of decode is PCIe dispatch overhead.**

Weight data per device per token (with 4-device split):
- Shared MLP (replicated): 9.4 MB → 0.04 ms theoretical DRAM time
- Mamba projections (col-parallel): 7.2 MB/device → 0.03 ms
- MoE active experts (EP4): 7.1 MB/device → 0.03 ms
- LM head: 77 MB/device → 0.30 ms
- **Total DRAM: ~2.6 ms. PCIe dispatch overhead: ~116 ms.**

Each TTNN op dispatch costs ~100–150 µs (Python→TTNN call, kernel cache lookup, runtime arg write over PCIe). ~1000 dispatches per token = ~100–150 ms.

### What this means for TP / more devices
- **TP MLP / TP Attention**: saves DRAM reads already < 0.05 ms per layer. The `all_gather` dispatch cost erases any gain. Does not help.
- **More devices (32)**: dispatch overhead is fixed per op regardless of device count. DRAM is not the bottleneck. No gain for batch=1 decode.
- **Pipeline parallelism** (Gemini's proposal): requires large batch of independent sequences. At batch=1 decode, chips sit idle waiting for the previous layer. Zero gain.

### What does help
- **`_ssm_state_tt` in L1**: currently allocated in DRAM. SSM state `[H=12,D=64,N=128]` = 192 KB per layer × 36 layers = 6.8 MB total. Fits in 6% of chip L1. Moving it to `L1_MEMORY_CONFIG` eliminates 13.5 MB/token of DRAM round-trips on state reads+writes. Estimated: **~52 ms saved**.
- **Fusion 1+3** (SSM kernel): keeps `dA`, `dBx` in L1 CBs within the kernel — currently these 192 KB tensors bounce to DRAM between ops. Combined with state pinning: eliminates ~52 ms DRAM + ~47 ms dispatch.
- **Full trace capture**: eliminates ~900 PCIe dispatches per token entirely. Blocked by tt-metal#26649.

### Tensor sizes — what fits in L1 (per device, all 36 Mamba layers)
| Tensor | Size | Currently in |
|--------|------|-------------|
| SSM state `[H,D,N]` × 36 | 6.8 MB | **DRAM** ← fix this |
| SSM weights A, D, dt_bias × 36 | 6.8 MB | **DRAM** ← pin these |
| Conv cache `[C,K]` × 36 | 0.2 MB | **DRAM** ← pin these |
| All activations (dA, dBx, x, dt, B, C, gate, residual) × 36 | 21 MB | DRAM (needs trace to fix) |
| **Total pinnable without trace** | **13.8 MB** | 13% of chip L1 |
| Mamba `in_proj` weights | 9.4 MB/layer | DRAM — too large |
| MoE expert weights (top-6) | 14.2 MB | DRAM — routing unpredictable |
| LM head weights | 77 MB | DRAM — too large |

---

## Pending Improvements

### Priority 1 — Full Decode Trace (BLOCKED — attempt history below)

**Estimated gain**: eliminates ~900 dispatches/token → ~90 ms saved → **~25–35 tok/s**

**Why it's blocked**: `ttnn.all_gather` (used in MoE col-parallel reduce and Mamba TP gather) cannot be captured in a Metal trace. Metal trace records device op dispatches; `all_gather` internally allocates a fresh output tensor on every call, so its buffer address changes between compile run and replay → corrupted output or deadlock.

#### Current state of trace infrastructure

The **MLP-only trace** (`setup_mlp_trace` in [granite/decoder_layer.py](granite/decoder_layer.py#L282)) is already implemented and working for a partial gain. It traces `post_attn_norm → MoE + shared_mlp → residual add` per layer using synchronous `ttnn.all_gather` (not inside trace) — wait, actually `_mlp_forward` IS called inside `begin_trace_capture`. Whether this currently works is unclear — it was previously deadlocking (Attempt 2). Current `setup_mlp_trace` may have a guard: it skips if `not _use_col_parallel` (multi-row mesh). For tiny 1×4, `_use_col_parallel=True` so the trace attempt runs.

The trace call path in `model.py`:
- `_use_decode_trace = False` in `test_bench.py` — trace is **disabled by default**
- When enabled: `setup_decode_trace()` → `layer.setup_mlp_trace()` for each layer
- Decode: `_forward_decode_traced()` runs mixer untraced, then `execute_trace` for MLP

#### Attempt history

**Attempt 1 (2026-07-05) — 5.6 tok/s regression**
- `get_tt_ccl(device)` called inside `GraniteTTMoE.__init__` during weight loading
- `TT_CCL.__init__` → `ttnn.create_global_semaphore` → `mark_allocations_unsafe()` on all subsequent weight uploads → corrupted model state
- Fix applied: `self._tt_ccl = None` in `__init__`, set externally after load

**Attempt 2 (2026-07-06) — 20-hour deadlock**
- `setup_mlp_trace()` called `_mlp_forward()` inside `begin_trace_capture`
- `_mlp_forward` → `ttnn.all_gather` → allocates fresh output tensor during capture
- Metal warned: "Allocating device buffers is unsafe due to active trace"
- `execute_trace` then deadlocked — required kill -9
- Root cause: `ttnn.all_gather` allocates a new output tensor every call → not trace-safe

**Attempt 3 (2026-07-08) — 40-min deadlock, same root cause**
- Implemented `all_gather_async` with `persistent_output_tensor=` in both MoE and Mamba
- Deferred `get_tt_ccl` to after weight uploads (fixes `mark_allocations_unsafe`)
- Enabled `_use_decode_trace = True`
- Benchmark ran for 40+ minutes with zero output — hardware deadlocked during trace capture
- Key finding from gpt_oss comment: `all_gather_async` "writes to device, which is forbidden during trace capture"
- **Root cause confirmed**: BOTH `all_gather` AND `all_gather_async` cannot be inside `begin_trace_capture`/`end_trace_capture`
- `persistent_output_tensor` fixes the buffer-address-change problem for *replay*, but the capture itself still deadlocks
- All changes reverted. `_use_decode_trace = False` restored.

#### The correct path: restructure trace to exclude all_gather

Since no CCL collective (sync or async) can be captured in a trace, the MLP block cannot be traced as a unit on multi-device mesh. Options:

**Option A — Trace compute-only subgraphs around the gather (gpt_oss pattern)**:
- Capture trace 1: `post_attn_norm → matmul (gate+up)` (before gather)
- Execute `all_gather` synchronously outside trace (uses persistent_output_tensor for fixed address)
- Capture trace 2: `sum → matmul (down) → residual add` (after gather)
- Two trace IDs per layer, gather in between — complex but feasible
- Pre-allocated `_gather_in` and `_gather_out` tensors needed as trace boundaries

**Option B — Trace Mamba and Attention only, leave MoE completely untraced**:
- Mamba has no inter-device ops until the final `all_gather` at end of `forward_decode`
- Trace the entire Mamba body (`in_proj → conv → SSM → out_proj`) EXCLUDING the final `all_gather`
- Attention body similarly has no gather (KV cache read/write is local)
- Estimated gain: ~49 ms Mamba + ~2 ms Attention = ~51 ms → ~14 tok/s (vs 25–35 for full trace)
- Simpler: no MoE trace plumbing needed

**Option C — Wait for tt-metal#26649**:
- Upstream issue tracks `all_gather` inside trace support
- If merged, Attempt 2/3 approach works with no restructuring needed

**Current status**: reverted to baseline, `_use_decode_trace = False`, synchronous `all_gather` everywhere.

The deferred CCL init (`get_tt_ccl` called after weight uploads) is retained as a correctness fix — it prevents `mark_allocations_unsafe` from corrupting weights.

---

### ~~Priority — Pin `_ssm_state_tt` in L1~~ (BLOCKED — L1 CB clash)

**File**: [mamba/mamba_chunk_scan_parallel.py](mamba/mamba_chunk_scan_parallel.py)

**Attempted (2026-07-05) — failed**: `_ssm_state_tt` is 192 KB per layer. The `addcmul` kernel (program 70) statically allocates 595 KB of L1 circular buffers per core. Together they exceed the 1.5 MB per-core L1 budget.

Error: `Statically allocated circular buffers in program 70 clash with L1 buffers on core range. L1 buffer at 449408, static CB region ends at 595232.`

**What did work**: `_conv_cache_cols` (6 KB/layer, 216 KB total) — moved to `L1_MEMORY_CONFIG`. No L1 clash. Negligible DRAM saving (~0.2 MB/token) but correct.

**To make `_ssm_state_tt` L1-pinnable**: the `addcmul` kernel would need to use smaller CBs or be split across more cores to free address space. This requires Metal kernel modification — out of scope without deeper TTNN kernel access.

---

### ~~Priority — SSM Kernel MeshDevice fix~~ (DONE — 2026-07-07)

`kernel/ssm_update/op.py` updated to use `MeshProgramDescriptor` pattern:
- `ttnn.get_device_tensors(mesh_tensor)[dev_idx].buffer_address()` gives per-device DRAM offset
- One `ProgramDescriptor` per `MeshCoordinate`, same compile-time args, per-device runtime args
- Correctness: `python kernel/ssm_update/test_ssm_update.py --mesh` passes PCC=0.9999 on 1×4 mesh

**NOT wired into model** — without trace it is net slower:
- `generic_op` with `MeshProgramDescriptor`: ~1.27 ms/call (Python descriptor construction per decode step)
- Individual TTNN ops (5 dispatches): ~0.5 ms total
- For 20 Mamba layers: +15 ms net, raises mamba 49 ms → ~64 ms
- Benefit only materialises inside a Metal trace: Python cost paid once at capture, intermediate tensors (`dA`, `dBx`) stay in L1 CBs instead of bouncing to DRAM

---

### Priority 2 — Fusion 1+3: SSM Kernel (once trace works — ~+3–4 tok/s additional)

**Precondition**: full decode trace (Priority 1) must capture the Mamba body.

Replace the 5 SSM TTNN ops in [mamba/mamba_chunk_scan_parallel.py:721–740](mamba/mamba_chunk_scan_parallel.py#L721) with one `generic_op`:
```
h_out = dBx + dA * state      (addcmul) 
y     = sum(h_out * C, dim=N)  (bcast-mul + reduce)
```
**Requirements for trace-safe wiring**:
- Pre-allocate `_ssm_hout_tt` and `_ssm_y_tt` in `__init__` with fixed addresses
- Ping-pong: `_ssm_state_tt ↔ _ssm_hout_tt` swap each step (both pre-allocated → trace-safe)
- `_ssm_state_tt` must also be pre-allocated; prefill result is `ttnn.copy`-ed into it before first decode
- The `generic_op` call replaces lines 721–740 and is trace-captured as a single device kernel

**Net saving on top of trace**: ~468 fewer dispatches already eliminated by trace; kernel additionally removes ~192 KB/layer intermediate DRAM traffic → **~+3–4 tok/s** beyond trace alone.

---

### Priority 4 — On-Device MoE Routing for Prefill (~5–10% prefill)

**File**: [granite/moe_tt.py:163](granite/moe_tt.py#L163)

`compute_routing_cpu` downloads logits, runs topk/softmax on CPU, re-uploads every prefill layer (40 PCIe round-trips per forward pass). `compute_routing_device` already exists for S=1. Extend to S>1 to eliminate all routing PCIe traffic during prefill.

**Estimated gain**: 103.9 → ~110–114 tok/s on long_256. Risk: verify topk+softmax+embedding correctness for S>1.

---

### Priority 5 — MoE Prefill: On-Device All-Reduce (~2–3× prefill for small model only)

**File**: [granite/moe_tt.py](granite/moe_tt.py)

Only relevant for the small model (2×4 mesh). Tiny model is a single row — no cross-row reduce needed.

**Previous revert**: Two-axis all-gather hurt prefill by 53% — gathered `[devices, S, H]` then summed over `S` was slow.

**Fix**: `ttnn.reduce_scatter` on row axis → `ttnn.all_gather` on col axis:
```python
reduced  = ttnn.reduce_scatter(local_sum_tt, scatter_dim=2, math_op=ttnn.ReduceType.Sum, cluster_axis=0)
gathered = ttnn.all_gather(reduced, dim=1, cluster_axis=1)
result_tt = ttnn.sum(gathered, dim=1, keepdim=True)
```

---

### Not Recommended for Tiny Model

| Item | Reason |
|------|--------|
| TP Shared MLP | DRAM reads already 0.04 ms — `all_gather` dispatch cost erases savings. Negligible net gain. |
| TP Attention | Only 2.4 ms measured (2% of token time). DRAM reads 0.04 ms. Not worth complexity. |
| TP LM Head | LM head is 0.3 ms theoretical. Not the bottleneck. |
| More devices (32) | Dispatch overhead is fixed per op, not per device. No gain for dispatch-bound workload. |
| Batched conv1d (Fusion 2) | Tested — 5.4× slower under trace. Shift-register already optimal. |
| Gated SiLU fusion (Fusion 4) | Reverted — wrong semantics (`SiLU(y*g)` not `y*SiLU(g)`). |
| Mamba split-trace | 6.3 vs 8.0 tok/s — extra device copies cost more than trace saves. |

---

## Priority Order and Expected Impact

| Priority | Change | Decode gain | Prefill gain | Complexity | Status |
|:--------:|--------|:-----------:|:------------:|:----------:|:------:|
| 1 | **Full decode trace** (`all_gather_async` + `persistent_output_tensor`) | **~3–4× (~25–35 tok/s)** | — | High | Next to attempt |
| 2 | **SSM kernel fusion** (Fusion 1+3, inside trace) | **+~3–4 tok/s** additional | — | Medium | Waiting on P1 |
| 3 | On-device MoE routing (prefill) | — | **+5–10%** | Medium | Pending |
| 4 | MoE prefill all-reduce (small only) | — | **+2–3×** (small) | Medium | Pending |

### Trace implementation options (Priority 1)

**Current state**: all trace attempts have deadlocked (Attempts 1–3). Baseline is ~8 tok/s with `_use_decode_trace = False`.

**Next attempt — Option B (Mamba-only trace, ~14 tok/s estimate)**:
1. In [mamba_chunk_scan_parallel.py](mamba/mamba_chunk_scan_parallel.py), capture trace from `in_proj matmul` through `out_proj matmul`, stopping BEFORE the final `all_gather`
2. Do the `all_gather` outside the trace (synchronous, already allocates fixed address per session once buffer layout is stable)
3. Pre-allocate `_mamba_trace_in` and `_mamba_trace_out_pre_gather` as fixed-address trace boundaries
4. In [decoder_layer.py](granite/decoder_layer.py), update `forward_mlp_traced` to only trace the MLP when no CCL is involved — or gate the MoE trace per-layer

**Key invariant**: `get_tt_ccl` / `create_global_semaphore` must be called **after** all weight uploads — never during `__init__` while weights are being loaded (triggers `mark_allocations_unsafe`). Already fixed in `model.py` (deferred to after `del hf_model`).

## Targets

| Model | Current | After full trace | After +Fusion 1+3 |
|-------|--------:|:----------------:|:-----------------:|
| tiny decode | 8.0 tok/s | ~25–35 tok/s (TBD) | ~30–40 tok/s |
| tiny prefill long_256 | 105.1 tok/s | same | ~110–114 tok/s |

---

## Hardware Notes

- Fabric firmware (`FABRIC_1D`) cannot be re-initialized between Python processes in the same hardware session. Each subsequent `open_mesh_device` with fabric hangs with "Read unexpected run_mailbox value: 0x40". Performance tests must be the **first process** after a hardware reset.
- `tt-smi -r` or `tt-smi -glx_reset_auto` resets all devices. `tt-smi -r 0` only resets PCI device 0 — insufficient for Galaxy.
- Do not use `trace_region_size` in `open_mesh_device` — causes hangs. Matches `test_bench.py` behavior.
- Import `TTGraniteMoeHybridForCausalLM` **after** `open_mesh_device` to avoid early ttnn init conflicting with fabric.
- Multi-chunk attention (chunk_size < seq_len) uses a different SDPA path (`is_causal=False` + explicit mask) that hangs on WH. Keep chunk_size ≥ max prompt length.
