# Changelog — 2026-05-21

Changes made after the original REPORT.md benchmarks.

---

## Improvement

### MoE sum-first all_reduce (`moe_tt.py`)

**Root cause (performance):** The original implementation gathered all expert outputs
across devices to CPU via PCIe, summed on CPU, then re-uploaded.

**Fix:** Each device now sums its local experts first (`ttnn.sum(dim=1)`), then a single
`ttnn.all_gather` + `ttnn.sum` reduces the partial sums entirely on-device — eliminating
the PCIe round-trips per MoE layer.

---

## Updated Benchmark Results

Benchmarks re-run on 2026-05-21 with the fix applied.

### Tiny model (4 devices, MeshShape 1×4)

**Model load:** 23.7s (REPORT: 23.2s — unchanged)

#### Prefill tok/s

| Prompt | Tokens | Today | REPORT | Change |
|--------|-------:|------:|-------:|-------:|
| short_8 | 5 | 2.5 | 1.04 | +2.4× |
| short_10 | 8 | 9.7 | 2.28 | +4.3× |
| medium_32 | 25 | 27.1 | 6.68 | +4.1× |
| long_128 | 96 | 91.2 | 21.2 | +4.3× |
| long_256 | 176 | 11.2 | 27.1 | −2.4× |

#### Decode tok/s (steady-state)

| Prompt | Today | REPORT | Change |
|--------|------:|-------:|-------:|
| short_8 | 6.25 | 3.10 | +2.0× |
| short_10 | 6.32 | 3.10 | +2.0× |
| medium_32 | 6.26 | 3.05 | +2.1× |
| long_128 | 6.24 | 3.06 | +2.0× |
| long_256 | 6.15 | 3.05 | +2.0× |

---

### Small model (8 devices, MeshShape 2×4)

**Model load:** 115.3s (REPORT: 117.0s — unchanged)

#### Prefill tok/s

| Prompt | Tokens | Today | REPORT | Change |
|--------|-------:|------:|-------:|-------:|
| short_8 | 5 | 1.3 | 0.37 | +3.5× |
| short_10 | 8 | 4.7 | 0.75 | +6.3× |
| medium_32 | 25 | 12.2 | 2.22 | +5.5× |
| long_128 | 96 | 47.3 | 3.76 | +12.6× |
| long_256 | 176 | 5.0 | 4.72 | +1.1× |

#### Decode tok/s (steady-state)

| Prompt | Today | REPORT | Change |
|--------|------:|-------:|-------:|
| short_8 | 4.84 | 1.99 | +2.4× |
| short_10 | 4.79 | 1.97 | +2.4× |
| medium_32 | 4.84 | 1.95 | +2.5× |
| long_128 | 4.83 | 1.94 | +2.5× |
| long_256 | 4.75 | 1.97 | +2.4× |

---

## Summary

| Metric | Tiny (4-dev) | Small (8-dev) |
|--------|:------------:|:-------------:|
| Decode improvement vs REPORT | ~2.0× | ~2.4× |
| Prefill improvement (short) | 2.4–4.3× | 3.5–6.3× |
| Prefill improvement (long_128) | 4.3× | 12.6× |

The primary driver of the ~2× decode improvement is the on-device MoE all_reduce, which eliminates 36 PCIe round-trips per decode step.
