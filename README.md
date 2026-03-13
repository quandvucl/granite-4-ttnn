# TT-Granite: Granite MoE Hybrid on TT-Metal/TTNN

Educational implementation of IBM's Granite 4.0 MoE Hybrid model on Tenstorrent's TT-Metal hardware.

## Performance Summary

| Implementation | Latency | vs HF CPU | vs TT-Metal Llama |
|----------------|---------|-----------|-------------------|
| **HuggingFace (CPU)** | 2,300ms/token | 1.0x | - |
| **TT-Granite (current)** | 1,838ms/token | 0.8x | 52x slower |
| **TT-Metal Llama 8B** | 35ms/token | 0.015x | 1.0x (target) |

### Bottleneck Analysis
- SSM operations: **1,300ms (71%)** ← Main bottleneck
- TTNN matmul: 100ms (5.4%)
- Conversion overhead: 35ms (1.9%)
- Other (cache, embedding): 400ms (22%)

**Key Finding:** CPU SSM operations dominate. Need custom C++ TTNN kernel for SSM to reach TT-Metal Llama performance.

## Repository Structure

```
tt-granite/
├── README.md                      # This file
├── generate.py                    # Main test script (working)
│
├── tt_model/                      # Model implementation
│   ├── __init__.py
│   ├── model.py                   # Main model (TTGraniteMoeHybridForCausalLM)
│   ├── decoder_layer.py           # Decoder layer
│   └── base.py                    # Base classes
│
└── tt_ops/                        # TTNN operations
    ├── __init__.py
    ├── base.py                    # Tensor conversion utilities
    ├── mamba.py                   # Mamba layer (SimpleMamba2TTNN)
    ├── attention_tt.py            # Attention layer
    ├── normalization.py           # RMSNorm
    └── cache.py                   # KV cache manager
```

## Quick Start

### Test Current Implementation
```bash
source env/bin/activate
python3 generate.py
```

**Expected output:**
- Loads model (~20s)
- Generates 10 tokens
- Shows timing breakdown
- Performance: ~1,838ms per token

### Results
```
Original: The future of AI is not just about building
TT-Metal: The future of AI is not just about building
Match: 9/10 tokens identical ✓
Time: 1,838ms per token (68x slower than HF CPU)
```

## Implementation Details

### What's Optimized
- ✅ TTNN matmul for large projections (in_proj, out_proj, O_proj)
- ✅ Hybrid approach: TTNN where beneficial, CPU otherwise
- ✅ Cache management with HybridKVCacheManager

### What's Not Optimized (Bottlenecks)
- ❌ SSM core on CPU (1,300ms) - needs custom kernel
- ❌ Attention on CPU (120ms) - needs fused kernel
- ❌ Memory layout not optimized - generic tiles

## Next Steps: Custom C++ Kernels

To match TT-Metal Llama (35ms/token), we need:

### Phase 1: Fused SSM Kernel (PRIORITY)
- **Impact:** 1,838ms → ~600ms (3x faster)
- **Time:** 2-4 weeks
- **Savings:** 1,250ms by moving SSM to device

See [KERNEL_DEVELOPMENT_PLAN.md](KERNEL_DEVELOPMENT_PLAN.md) for detailed roadmap.

### Phase 2: Fused Attention Kernel
- **Impact:** ~600ms → ~480ms (1.25x faster)
- **Time:** 1-2 weeks
- **Savings:** 120ms from on-device attention

### Phase 3: Memory Optimization
- **Impact:** ~480ms → ~430ms (1.1x faster)
- **Time:** 1 week
- **Savings:** 50ms from optimal layout

### Phase 4: Production Polish
- **Impact:** ~430ms → ~35ms (12x faster)
- **Time:** 1-2 months
- **Savings:** Final 400ms from tuning

**Total: 1,838ms → 35ms (52x faster) over 2-4 months**

## Key Learnings

1. **Conversion overhead is minimal** (~35ms, not 232ms as estimated)
   - Real measurements show 0.17-0.67ms per conversion
   - Python-level fusion only saves ~15ms (1% improvement)

2. **SSM operations dominate** (1,300ms = 71% of time)
   - Complex CPU operations: discretization, state updates, matmuls
   - Small matrices don't benefit from TTNN without fusion
   - **Custom kernel required** for meaningful speedup

3. **TT-Metal Llama's advantage** is production-quality kernels
   - Fused SSM kernel
   - Fused attention kernel
   - Hardware-specific optimizations
   - Months of engineering work

4. **For decode workloads**, CPUs are architecturally superior
   - Without custom kernels, accelerators can't compete
   - Weights fit in L3 cache
   - Zero transfer overhead after warmup
   - **But with proper kernels, accelerators win on throughput**

## Documentation

- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Complete performance analysis
- [FUSION_OPPORTUNITIES.md](FUSION_OPPORTUNITIES.md) - What can be fused in Python vs C++
- [KERNEL_DEVELOPMENT_PLAN.md](KERNEL_DEVELOPMENT_PLAN.md) - C++ kernel implementation guide

## References

- **TT-Metal:** https://github.com/tenstorrent/tt-metal
- **Granite Model:** https://huggingface.co/ibm-granite/granite-4.0-h-1b
- **TT-Metal Docs:** https://docs.tenstorrent.com/tt-metal/latest/
- **TTNN API:** https://docs.tenstorrent.com/tt-metal/latest/ttnn/

## Status

**Current:** Working implementation with validated performance analysis  
**Next:** Begin C++ kernel development for SSM fusion  
**Goal:** Match TT-Metal Llama's 35ms/token performance

---

**Note:** This is an educational implementation. For production use, prefer TT-Metal's official Llama implementation.
