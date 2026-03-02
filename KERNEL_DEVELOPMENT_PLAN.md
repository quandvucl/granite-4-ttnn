# C++ Kernel Development Plan for TT-Granite

## Current State

**Performance:** 1,838ms per token (52x slower than TT-Metal Llama's 35ms)

**Bottleneck Analysis (validated with measurements):**
- SSM operations: **1,300ms (71%)** - BIGGEST BOTTLENECK
- TTNN matmul compute: ~100ms (5.4%)
- Conversion overhead: ~35ms (1.9%)
- Other (cache, embedding, etc): ~400ms (22%)

**Key Finding:** Conversions are NOT the bottleneck (only 35ms vs estimated 232ms). The SSM CPU operations dominate at 1,300ms.

---

## Target Architecture

TT-Metal Llama achieves 35ms/token through:
1. **Fused SSM kernel** - All SSM ops on device (saves 1,250ms!)
2. **Fused attention kernel** - Q/K/V/O + RoPE + cache on device (saves 120ms)
3. **Optimized memory layout** - Hardware-specific sharding (saves 50ms)
4. **Production polish** - Months of tuning (final 70ms)

---

## Kernel Implementation Roadmap

### Phase 1: Fused Mamba SSM Kernel (PRIORITY)
**Impact: 1,838ms → ~600ms (3x faster)**
**Time: 2-4 weeks**

#### What needs to be fused:
```cpp
// Current: 26 Mamba layers, each with SSM on CPU (50ms/layer)
// Goal: Single fused kernel per layer (2ms/layer)

FusedMambaKernel {
    // Input projection (already TTNN - keep)
    projected = X @ W_in;  // TTNN

    // FUSE THESE OPERATIONS ON DEVICE:
    gate, hidden, dt, B, C = split(projected);

    // Conv1d with state
    conv_state = roll(conv_state, -1);
    conv_state[-1] = hidden;
    hidden = conv1d(conv_state, weight);
    hidden = silu(hidden);

    // SSM discretization (CRITICAL - currently 50ms on CPU)
    dt = softplus(dt + dt_bias);
    dt = clamp(dt, min, max);
    dA = exp(dt * A);           // Element-wise
    dB = dt * B;                // Broadcasting
    dBx = dB * hidden;          // Element-wise

    // State update (CRITICAL - needs on-device state management)
    ssm_state = ssm_state * dA + dBx;  // In-place update

    // Output computation
    y = C @ ssm_state;          // Matmul
    y = y + hidden * D;         // Residual
    output = y * silu(gate);    // Gating

    // Output projection (already TTNN - keep)
    output = output @ W_out;    // TTNN
}
```

#### Key Challenges:
1. **On-device state management**
   - SSM state: `[batch, num_heads, head_dim, state_size]`
   - Conv state: `[batch, intermediate_size, kernel_size]`
   - Currently managed by HF cache_params on CPU
   - Need device-side state buffer + update logic

2. **Complex tensor shapes & broadcasting**
   - Irregular reshapes: `reshape(batch, n_groups, -1).expand(...)`
   - GQA expansion for grouped state
   - Needs careful layout planning

3. **Element-wise operations**
   - exp, softplus, clamp, mul, add
   - TTNN has these but need to fuse into single kernel

#### Implementation Steps:

**Week 1: Kernel skeleton**
```cpp
// File: tt_ops/kernels/fused_mamba_ssm.cpp
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>

// Kernel signature
ttnn::Tensor fused_mamba_ssm(
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& dt,
    const ttnn::Tensor& B,
    const ttnn::Tensor& C,
    ttnn::Tensor& ssm_state,        // In-place state update
    const ttnn::Tensor& A_log,
    const ttnn::Tensor& D,
    const ttnn::Tensor& dt_bias,
    const MambaConfig& config
);

// Start with simple ops on device
// - Implement discretization (dA, dB, dBx)
// - Test correctness vs CPU version
```

**Week 2: State management**
```cpp
// Implement on-device state buffer
struct MambaStateBuffer {
    ttnn::Tensor ssm_states[NUM_LAYERS];
    ttnn::Tensor conv_states[NUM_LAYERS];

    void update_ssm_state(int layer_idx, const ttnn::Tensor& dA, const ttnn::Tensor& dBx);
    void update_conv_state(int layer_idx, const ttnn::Tensor& hidden);
};

// Fuse state update into kernel
// Test with small model first (fewer layers)
```

**Week 3: Optimization**
```cpp
// Profile kernel
// Optimize memory access patterns
// Add compute config for N150/N300 hardware
// Batch operations where possible

// Memory layout optimization
MatmulProgramConfig get_mamba_program_config(
    const Shape& input_shape,
    const DeviceComputeCapabilities& caps
);
```

**Week 4: Integration & validation**
```python
# Python binding
def fused_mamba_forward(
    hidden_states: ttnn.Tensor,
    layer_idx: int,
    config: MambaConfig,
    state_buffer: MambaStateBuffer
) -> ttnn.Tensor:
    return ttnn.fused_mamba_ssm(
        hidden_states, ..., state_buffer.ssm_states[layer_idx], ...
    )

# Test accuracy: compare with HF implementation
# Test performance: should be ~2ms per layer vs 50ms
# Integrate into full model
```

---

### Phase 2: Fused Attention Kernel (MEDIUM PRIORITY)
**Impact: ~600ms → ~480ms (1.25x faster)**
**Time: 1-2 weeks**

#### What needs to be fused:
```cpp
FusedAttentionKernel {
    // Current: 6 attention layers, each ~20ms
    // Goal: Single kernel ~5ms per layer

    // QKV projection (already TTNN but separate)
    qkv = X @ W_qkv;              // Single matmul instead of 3
    Q, K, V = split(qkv);

    // RoPE on device (currently CPU)
    Q = apply_rope(Q, cos, sin);
    K = apply_rope(K, cos, sin);

    // Cache update on device (currently CPU)
    K, V = update_cache(K, V, kv_cache, position);

    // Attention (currently CPU)
    scores = Q @ K.T * scaling;
    scores = scores + mask;
    probs = softmax(scores);
    output = probs @ V;

    // O projection
    output = output @ W_o;
}
```

#### Benefits:
- Eliminate 6 layers × 3 conversions = 18 conversions (~5ms)
- Faster RoPE on device (~10ms)
- Optimized SDPA (~5ms)
- **Total: ~20ms savings per attention layer × 6 = 120ms**

---

### Phase 3: Memory Optimization (LOW PRIORITY)
**Impact: ~480ms → ~430ms (1.1x faster)**
**Time: 1 week**

#### Optimizations:
1. **Pre-sharded tensors**
   - Split weights across cores at load time
   - Avoid runtime sharding overhead

2. **Optimized tile sizes**
   - Use hardware-optimal tile sizes (32×32 for Wormhole)
   - Pad tensors to tile boundaries

3. **Memory placement**
   - Critical weights in L1 cache
   - State buffers in optimal DRAM banks
   - Double buffering for transfers

---

## Development Environment Setup

### Required Tools:
```bash
# TT-Metal SDK
git clone https://github.com/tenstorrent/tt-metal
cd tt-metal
# Follow installation: https://github.com/tenstorrent/tt-metal#installation

# Build tools
sudo apt install cmake ninja-build

# Set up environment
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

### Kernel Development Workflow:
```bash
# 1. Write kernel in C++
vim tt_ops/kernels/fused_mamba_ssm.cpp

# 2. Compile
cd tt_ops/kernels
cmake -B build -G Ninja
ninja -C build

# 3. Test
python test_fused_kernel.py

# 4. Profile
tt_metal_profiler test_fused_kernel.py

# 5. Iterate
```

---

## Testing Strategy

### Correctness Tests:
```python
# Compare fused kernel vs HF reference
def test_kernel_correctness():
    hf_output = hf_mamba(input, cache)
    tt_output = fused_mamba(input_tt, state_buffer)

    assert torch.allclose(tt_output, hf_output, atol=1e-2)
    print("✓ Kernel output matches HF")

# Test across different scenarios
test_cases = [
    ("prefill", seq_len=128),
    ("decode", seq_len=1),
    ("batch", batch_size=16),
]
```

### Performance Tests:
```python
def benchmark_kernel():
    times = []
    for _ in range(1000):
        start = time.time()
        output = fused_mamba(input_tt, state_buffer)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Kernel time: {avg_time*1000:.3f}ms")

    # Should be < 2ms (vs 50ms on CPU)
    assert avg_time < 0.002
```

---

## Expected Results by Phase

| Phase | Time | Per-token Latency | vs TT-Metal Llama |
|-------|------|-------------------|-------------------|
| **Current** | - | 1,838ms | 52x slower |
| **Phase 1: Fused SSM** | 2-4 weeks | ~600ms | 17x slower |
| **Phase 2: Fused Attention** | 1-2 weeks | ~480ms | 14x slower |
| **Phase 3: Memory Opt** | 1 week | ~430ms | 12x slower |
| **Phase 4: Production Polish** | 1-2 months | ~35ms | **1x (match!)** |

---

## Alternative: Use TT-Metal's Existing Kernels

Instead of writing kernels from scratch, we could:

1. **Study TT-Metal Llama implementation**
   - Path: `tt-metal/models/demos/t3000/llama2_70b/`
   - Extract their fused attention kernel
   - Adapt for Granite's architecture

2. **Use TT-Metal's building blocks**
   - `ttnn.experimental.operations.primary.transformers.rotary_embedding`
   - `ttnn.experimental.operations.primary.transformers.scaled_dot_product_attention`
   - Combine into custom Granite kernels

3. **Collaborate with TT-Metal team**
   - Submit PR to add Granite model to tt-metal/models
   - Get official optimization support
   - Benefit from their expertise

**Recommended:** Start with Phase 1 (Fused SSM) since it's the biggest win, then evaluate whether to continue custom development or collaborate with TT-Metal team.

---

## Resources

- **TT-Metal Documentation:** https://docs.tenstorrent.com/tt-metal/latest/
- **TTNN API Reference:** https://docs.tenstorrent.com/tt-metal/latest/ttnn/
- **Llama Reference Implementation:** `tt-metal/models/demos/t3000/llama2_70b/`
- **Kernel Examples:** `tt-metal/ttnn/cpp/ttnn/operations/`

---

## Next Steps

1. ✅ Clean up test files (done)
2. ⬜ Set up TT-Metal development environment
3. ⬜ Study TT-Metal Llama SSM/Mamba implementation (if exists)
4. ⬜ Write simple fused SSM kernel (C++)
5. ⬜ Test correctness vs CPU version
6. ⬜ Optimize & integrate into full model
7. ⬜ Measure speedup: target 1,838ms → 600ms

**Ready to start kernel development!**
