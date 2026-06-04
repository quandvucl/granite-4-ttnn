# Plan: Fully Utilizing Fabric on Granite MoE Hybrid

## Current Status

Baseline decode throughput (tiny, 4 devices, fabric enabled): **~7.92 tok/s**

| Component | Status | Parallelism |
|-----------|--------|-------------|
| Embedding (1536-dim lookup) | Replicated | 1x |
| Attention (20 layers, GQA 12H/4KV) | Replicated | 1x |
| Mamba (20 layers) | Column-parallel + all_gather | 4x |
| MoE experts (64 experts, top-6) | Expert-parallel + all_gather | 4x |
| Shared MLP (512 intermediate) | **Replicated** | 1x |
| LM head (1536 → vocab) | Replicated | 1x |
| Final norm | Replicated | 1x |

Mamba and MoE are already fabric-accelerated. The remaining gains come from parallelizing
the components still doing redundant full compute on all 4 chips.

---

## Step 1: Profile decode bottleneck (1 day)

Before implementing, measure where the 126 ms/token actually goes.

**How:** Add timing accumulation in `model.py` forward loop across all 40 layers and
print per-component breakdown after each inference.

```
Expected breakdown at decode (rough estimate):
  Attention total:    ~35 ms  (20 layers × ~1.75 ms each, replicated)
  Mamba total:        ~25 ms  (20 layers, already TP — ~1.25 ms each)
  MoE total:          ~30 ms  (20 layers, EP sharded — ~1.5 ms each)
  Shared MLP total:   ~20 ms  (20 layers × ~1.0 ms each, replicated)
  Norm/overhead:      ~16 ms
  Total:             ~126 ms → 7.92 tok/s
```

This confirms which step to tackle first and sizes the expected win.

---

## Step 2: Tensor-parallel Shared MLP (highest leverage)

**Files:** `granite/model.py` (`ReplicatedMLP` → `ColumnParallelMLP`)

**Model dimensions:**
- tiny:  hidden=1536, intermediate=512  → gate/up weights [1536, 512], down [512, 1536]
- small: hidden=4096, intermediate=768  → gate/up weights [4096, 768], down [768, 4096]

**Strategy:** Column-parallel gate+up, row-parallel down, one `all_gather` per layer.

```
gate_weight [H, F]  →  shard across cols → [H, F/4] per device
up_weight   [H, F]  →  shard across cols → [H, F/4] per device
down_weight [F, H]  →  shard across rows → [F/4, H] per device

forward(x):
  gate = x @ gate_weight   # [1,1,1,F/4] local
  up   = x @ up_weight     # [1,1,1,F/4] local
  mid  = silu(gate) * up   # [1,1,1,F/4] local
  out  = mid @ down_weight  # [1,1,1,H] partial sums
  out  = all_reduce(out)   # [1,1,1,H] full sum
```

**all_reduce via all_gather:** same pattern as Mamba TP — `ttnn.all_gather` on
the output dim, then sum the 4 shards on device. Requires fabric (already enabled).

**Expected gain:** 4x on MoE expert FFN linear ops and now also shared MLP linears.
Roughly: 20 ms → 5 ms for shared MLP = ~15 ms saved per token = **~10-11 tok/s**.

**Gating:** Only active when `use_all_gather=True` and `num_devices > 1` (same gate as Mamba TP).
Falls back to `ReplicatedMLP` path when running without fabric.

---

## Step 3: Tensor-parallel Attention (moderate gain)

**Files:** `granite/attention_nope.py`

**Model dimensions:**
- tiny:  12 heads, 4 KV heads, head_dim=128, hidden=1536
  - Q proj: [1536, 1536], K proj: [1536, 512], V proj: [1536, 512], O proj: [1536, 1536]
- small: 32 heads, 8 KV heads, head_dim=128, hidden=4096
  - Q proj: [4096, 4096], K proj: [4096, 1024], V proj: [4096, 1024], O proj: [4096, 4096]

**Strategy:** Shard Q/K/V projections across head dimension; each device handles `num_heads/4`
heads. O projection is row-parallel, followed by all_gather.

```
Q: [H, H]   → shard heads → each device: H/4 heads (tiny: 3 heads/device)
K: [H, KH]  → shard KV heads → 1 KV head/device (tiny: 4 KV heads / 4 devices)
V: [H, KH]  → same as K
O: [H, H]   → row-parallel input, all_gather output

Decode: each device computes attention for its head slice, writes to local KV cache shard.
```

**KV cache:** Each device only holds its own KV head slices — 4x smaller per device.
Cache reset in `decoder_layer.py:reset_cache()` must be updated to match shard sizes.

**Expected gain:** ~35 ms → ~10 ms for attention = ~25 ms saved per token = **~14-16 tok/s**.

**Complexity:** High. KV cache is pre-allocated per-device and sliced by head. Prefill
attention mask broadcasting must account for per-device head ranges. Most complex step.

---

## Step 4: Tensor-parallel LM Head

**Files:** `granite/model.py` (LMHead1D already exists but runs replicated)

**Model dimensions:**
- tiny:  [1536, 131072 vocab] — large matrix, bandwidth-bound
- small: [4096, 131072 vocab]

**Strategy:** Shard vocab dimension across 4 devices. Each device computes logits for
its vocab slice. Take max-logit argmax locally, then all_gather the 4 winning tokens +
scores and pick the global max. For greedy/top-1 this avoids all_gather of the full
vocab tensor.

```
lm_head_weight [H, V]  →  shard V → [H, V/4] per device
logits_shard = x @ lm_head_shard   # [1, V/4] local
# Option A (greedy): local argmax + all_gather(top-1 per shard) → pick global
# Option B (sampling): all_gather full vocab slices → concat → sample
```

**Expected gain:** LM head is ~5-10% of decode time but vocab is 131K — worth doing
after Steps 2+3 are landed.

---

## Step 5: 2×4 mesh for small model (expert-row parallelism)

**Files:** `granite/moe_tt.py`

Small model uses 8 devices in a 2×4 mesh. Currently MoE uses flat EP across 8 devices
with CPU reduce. This was reverted from a 2-axis reduce due to prefill regression (-53%).

**Revisit:** The 2-axis (col EP + row all_reduce) approach hurt prefill because the
row-axis all_gather was slow for long sequences. Consider:
- Keep col-axis EP (4 experts/row-group, 18 experts/device)
- Use row-axis all_reduce only for decode (seq_len=1), fall back to CPU reduce for prefill

This requires a `mode`-aware reduce path in `GraniteTTMoE.forward()`.

---

## Priority Order

| Step | Expected Decode Gain | Complexity | Notes |
|------|---------------------|------------|-------|
| 1. Profile | — | Low | Do first to validate estimates |
| 2. TP Shared MLP | +3-4 tok/s (~40%) | Medium | Best ROI, same pattern as Mamba |
| 3. TP Attention | +6-8 tok/s (~50%) | High | Large win but complex KV cache sharding |
| 4. TP LM Head | +0.5-1 tok/s | Low | Easy win, do after 2+3 |
| 5. 2×4 EP+TP for small | TBD | Medium | Only relevant for small model |

**Target after all steps:** tiny model ~18-22 tok/s (from current 7.92).

---

## Constraints

- Fabric must be active (`use_all_gather=True`) for Steps 2, 3, 4 to engage TP.
  All TP paths must fall back gracefully to replicated when fabric is unavailable.
- `reset_cache()` in `decoder_layer.py` must remain consistent with whatever KV
  cache layout Step 3 introduces.
- Prefill correctness must be verified after each step — TP bugs tend to show up
  as garbled output or wrong shapes at seq_len > 1.
- `bfloat8_b` MoE weights currently produce garbled text — investigate separately,
  not blocking for throughput work.
