# Changelog

Changes made after the original REPORT.md benchmarks.

---

## 2026-05-29

**Improvement:**
- Prefill SSM state stays on device across chunks (eliminates inter-chunk PCIe)
- MoE expert weight precision (`moe_weight_dtype=bfloat8_b`) — kept for tiny, skipped for small (load time too high)

**Table 1 — vs previous version (before this session)**

*Tiny model (4 devices)*

| Prompt | Prefill before | Prefill after | Δ | Decode before | Decode after | Δ |
|--------|:--------------:|:-------------:|:---:|:-------------:|:------------:|:---:|
| short_8 | 2.5 | 2.5 | — | 6.20 | 6.47 | +4% |
| short_10 | 9.9 | 10.5 | +6% | 6.19 | 6.50 | +5% |
| medium_32 | 26.4 | 28.1 | +6% | 6.17 | 6.44 | +4% |
| long_128 | 90.1 | 94.7 | +5% | 6.19 | 6.48 | +5% |
| long_256 | 116.8 | 123.0 | +5% | 6.19 | 6.43 | +4% |

*Small model (8 devices)*

| Prompt | Prefill before | Prefill after | Δ | Decode before | Decode after | Δ |
|--------|:--------------:|:-------------:|:---:|:-------------:|:------------:|:---:|
| short_8 | 1.3 | 1.4 | +8% | 3.66 | 3.71 | +1% |
| short_10 | 4.8 | 5.5 | +15% | 3.63 | 3.71 | +2% |
| medium_32 | 13.1 | 14.5 | +11% | 3.61 | 3.71 | +3% |
| long_128 | 46.1 | 51.2 | +11% | 3.64 | 3.71 | +2% |
| long_256 | 60.9 | 64.0 | +5% | 3.64 | 3.74 | +3% |

---

**Table 2 — vs CPU and A100 (REPORT.md baseline)**

*Tiny model (4 devices) — Prefill (tok/s)*

| Prompt | TTNN | HF CPU | HF A100 | vs CPU | vs A100 |
|--------|:----:|:------:|:-------:|:------:|:-------:|
| short_8 | 2.5 | 0.44 | 5.6 | +5.7× | 0.45× |
| short_10 | 10.5 | 0.70 | 25.4 | +15× | 0.41× |
| medium_32 | 28.1 | 2.17 | 75.1 | +13× | 0.37× |
| long_128 | 94.7 | 8.22 | 279 | +12× | 0.34× |
| long_256 | 123.0 | 14.8 | 503 | +8.3× | 0.24× |

*Tiny model (4 devices) — Decode (tok/s)*

| Prompt | TTNN | HF CPU | HF A100 | vs CPU | vs A100 |
|--------|:----:|:------:|:-------:|:------:|:-------:|
| short_8 | 6.47 | 2.58 | 7.65 | +2.5× | 0.85× |
| short_10 | 6.50 | 2.60 | 9.40 | +2.5× | 0.69× |
| medium_32 | 6.44 | 2.61 | 9.32 | +2.5× | 0.69× |
| long_128 | 6.48 | 2.61 | 9.34 | +2.5× | 0.69× |
| long_256 | 6.43 | 2.62 | 9.19 | +2.5× | 0.70× |

*Small model (8 devices) — Prefill (tok/s)*

| Prompt | TTNN | HF CPU | HF A100 | vs CPU | vs A100 |
|--------|:----:|:------:|:-------:|:------:|:-------:|
| short_8 | 1.4 | 0.16 | 0.95 | +8.8× | 1.47× |
| short_10 | 5.5 | 0.26 | 9.84 | +21× | 0.56× |
| medium_32 | 14.5 | 0.81 | 28.8 | +18× | 0.50× |
| long_128 | 51.2 | 3.01 | 95.6 | +17× | 0.54× |
| long_256 | 64.0 | 5.39 | 178 | +12× | 0.36× |

*Small model (8 devices) — Decode (tok/s)*

| Prompt | TTNN | HF CPU | HF A100 | vs CPU | vs A100 |
|--------|:----:|:------:|:-------:|:------:|:-------:|
| short_8 | 3.71 | 0.66 | 5.39 | +5.6× | 0.69× |
| short_10 | 3.71 | 0.66 | 5.40 | +5.6× | 0.69× |
| medium_32 | 3.71 | 0.66 | 5.53 | +5.6× | 0.67× |
| long_128 | 3.71 | 0.66 | 5.55 | +5.6× | 0.67× |
| long_256 | 3.74 | 0.66 | 5.59 | +5.7× | 0.67× |

---

# Before

## Bug fix: small model garbage output on long prompts

**Problem:** Small model (`MeshShape(2,4)`) produced incoherent output for prompts ≥ 96 tokens (`long_128`, `long_256`). Short prompts were unaffected.

**Root cause:** `_load_weights` in `mamba/mamba_chunk_scan_parallel.py` used `ShardTensor2dMesh(..., mesh_shape=MeshShape(1, col_devices))` to distribute `in_proj` and `out_proj` weights. This only populated the 4 devices on row 0 of the 2×4 mesh; the 4 devices on row 1 received zero weights. After `all_gather(cluster_axis=1)` combined partial matmul results from both rows, the zeros from row 1 corrupted every output. Corrupted activations accumulated through the SSM recurrence and became visible as word-salad starting around token 32–64.

**Fix:** Pass the actual device mesh shape (`self.device.shape` = `MeshShape(2,4)`) to `ShardTensor2dMesh` so both rows receive valid column-shards. Per-device weight size is unchanged (`1/cols` of full weight), so no DRAM increase.

**Impact:** Output quality restored for all prompt lengths. Decode throughput unchanged.

---

## MoE on-device reduce

**Root cause (performance):** Each MoE layer gathered all expert outputs from all devices to CPU via PCIe, summed on CPU, then re-uploaded to all devices. This added 36 PCIe round-trips per forward pass (one per Mamba/MoE layer).

**Fix:** Each device now sums its local expert outputs first (`ttnn.sum(dim=1)`), then a single `ttnn.all_gather` + `ttnn.sum` reduces the partial sums entirely on-device, eliminating the PCIe round-trips.

For the small model's `MeshShape(2,4)` mesh, flat expert parallelism (9 experts/device across 8 devices) is used with a CPU reduce — a single small download per layer, acceptable cost. Column-parallel with row replication doubles per-device expert compute and hurts throughput.
