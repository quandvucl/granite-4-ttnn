# Changelog

Changes made after the original REPORT.md benchmarks.

---

## Bug fix: small model garbage output on long prompts

**Problem:** Small model (`MeshShape(2,4)`) produced incoherent output for prompts ≥ 96 tokens (`long_128`, `long_256`). Short prompts were unaffected.

**Root cause:** `_load_weights` in `mamba/mamba_chunk_scan_parallel.py` used `ShardTensor2dMesh(..., mesh_shape=MeshShape(1, col_devices))` to distribute `in_proj` and `out_proj` weights. This only populated the 4 devices on row 0 of the 2×4 mesh; the 4 devices on row 1 received zero weights. After `all_gather(cluster_axis=1)` combined partial matmul results from both rows, the zeros from row 1 corrupted every output. Corrupted activations accumulated through the SSM recurrence and became visible as word-salad starting around token 32–64.

**Fix:** Pass the actual device mesh shape (`self.device.shape` = `MeshShape(2,4)`) to `ShardTensor2dMesh` so both rows receive valid column-shards. Per-device weight size is unchanged (`1/cols` of full weight), so no DRAM increase.

**Impact:** Output quality restored for all prompt lengths. Decode throughput unchanged.

---

## MoE on-device reduce

**Root cause (performance):** Each MoE layer gathered all expert outputs from all devices to CPU via PCIe, summed on CPU, then re-uploaded to all devices. This added 36 PCIe round-trips per forward pass (one per Mamba/MoE layer).

**Fix:** Each device now sums its local expert outputs first (`ttnn.sum(dim=1)`), then a single `ttnn.all_gather` + `ttnn.sum` reduces the partial sums entirely on-device, eliminating the PCIe round-trips.

For the small model's `MeshShape(2,4)` mesh, column-parallel expert sharding causes OOM (row-replicated weights double per-device DRAM for large models). Small model uses flat expert parallelism with a CPU reduce instead — a single small download per layer, acceptable cost.

### Updated benchmark results

#### Tiny model (4 devices, MeshShape 1×4)

**Model load:** 23.6s (REPORT baseline: 23.2s)

| Prompt | Tokens | Prefill tok/s | vs REPORT | Decode tok/s | vs REPORT |
|--------|-------:|--------------:|----------:|-------------:|----------:|
| short_8 | 5 | 2.49 | +2.4× | 6.20 | +2.0× |
| short_10 | 8 | 9.92 | +4.4× | 6.19 | +2.0× |
| medium_32 | 25 | 26.4 | +4.0× | 6.17 | +2.0× |
| long_128 | 96 | 90.1 | +4.3× | 6.19 | +2.0× |
| long_256 | 176 | 116.8 | +4.3× | 6.19 | +2.0× |

#### Small model (8 devices, MeshShape 2×4)

**Model load:** 116.2s (REPORT baseline: 117.0s)

| Prompt | Tokens | Prefill tok/s | vs REPORT | Decode tok/s | vs REPORT |
|--------|-------:|--------------:|----------:|-------------:|----------:|
| short_8 | 5 | 1.42 | +3.8× | 4.07 | +2.0× |
| short_10 | 8 | 5.61 | +7.5× | 4.06 | +2.1× |
| medium_32 | 25 | 15.4 | +6.9× | 4.03 | +2.1× |
| long_128 | 96 | 54.4 | +14.5× | 4.04 | +2.1× |
| long_256 | 176 | 68.8 | +14.6× | 4.05 | +2.1× |

### Summary

| Metric | Tiny (4-dev) | Small (8-dev) |
|--------|:------------:|:-------------:|
| Decode improvement vs REPORT | ~2.0× | ~2.1× |
| Prefill improvement vs REPORT (short) | 2.4–4.4× | 3.8–7.5× |
| Prefill improvement vs REPORT (long) | 4.3× | 14.5–14.6× |
| vs A100 decode (avg) | 0.68× | 0.74× |

#### TTNN vs A100 prefill by prompt

| Prompt | Tokens | Tiny vs A100 | Small vs A100 |
|--------|-------:|:------------:|:-------------:|
| short_8 | 5 | 0.44× | 1.49× |
| short_10 | 8 | 0.39× | 0.57× |
| medium_32 | 25 | 0.35× | 0.53× |
| long_128 | 96 | 0.32× | 0.57× |
| long_256 | 176 | 0.23× | 0.39× |