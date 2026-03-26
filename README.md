# TT-Granite: Granite MoE Hybrid on TT-Metal/TTNN

Educational implementation of IBM's Granite 4.0 MoE Hybrid model on Tenstorrent's TT-Metal hardware.

## Environment
```bash
docker run -d \
    --name tt-granite-dev \
    --privileged \
    --device /dev/tenstorrent \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v ~/projects/tt-granite:/work/tt-granite \
    --ipc=host \
    ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc \
    sleep infinity
docker exec -ti tt-granite-dev bash
python -m venv env
pip install -r requirements.txt
source env/bin/activate
```

in which `~/projects/tt-granite` is the path to the current repo

## How To Run Comparison

```bash
# Run on A100 GPU
python generate.py --model a100 --max-tokens 20

# Run on Tenstorrent
python generate.py --model tt --max-tokens 20

# Run on HuggingFace CPU
python generate.py --model hf --max-tokens 20
```

## Performance Summary

**Test Configuration:** Granite 4.0 Hybrid 1B model, batch size 1, generating 20 tokens

### Quick Comparison

| Hardware | Load Time | Generation Time | Throughput | Speedup vs CPU |
|----------|-----------|-----------------|------------|----------------|
| **NVIDIA A100 GPU** | 1.82s | 0.999s | **20.01 tok/s** | **12.8x faster** |
| **Tenstorrent (1 device)** | 4.61s | 8.360s | **2.39 tok/s** | **1.5x faster** |
| **HuggingFace CPU** | 0.60s | 12.848s | 1.56 tok/s | baseline |

**Key Findings:**
- All three implementations produce **identical outputs** (token-level exact match)
- A100 delivers the best throughput at 20 tok/s (12.8x faster than CPU)
- Tenstorrent shows 1.5x speedup over CPU with room for optimization
- CPU has fastest load time but slowest inference

---

### Detailed Results

#### NVIDIA A100 GPU
```
Model loaded in 1.82s
GPU Memory After Load: 2.72 GB

Batch size: 1
Prompts: 1
Prompt: "The future of AI is"
Generating 20 tokens per prompt...
Warming up GPU kernels... done

Output (first sample):
The future of AI is bright, but it's also full of challenges. As we continue to develop and integrate AI into our

Performance:
  Total time:  2.820s
    - Load:    1.821s
    - Generate:0.999s
  Total tokens: 20
  Throughput:  20.01 tokens/sec
  Per-sample:  20.01 tokens/sec
  Peak GPU Memory: 5.08 GB
```

#### HuggingFace CPU (Baseline)
```
Model loaded in 0.60s

Batch size: 1
Prompts: 1
Prompt: "The future of AI is"
Generating 20 tokens per prompt...

Output (first sample):
The future of AI is bright, but it's also full of challenges. As we continue to develop and integrate AI into our

Performance:
  Total time:  13.452s
    - Load:    0.604s
    - Generate:12.848s
  Total tokens: 20
  Throughput:  1.56 tokens/sec
  Per-sample:  1.56 tokens/sec
```

#### Tenstorrent (1 device)
```
Opened 1 device as MeshDevice(1x1)

Loading TT model...
Model loaded in 4.61s
Warming up (kernel compilation)... done

Batch size: 1
Prompts: 1
Prompt: "The future of AI is"
Generating 20 tokens per prompt...

Output (first sample):
The future of AI is bright, but it's also full of challenges. As we continue to develop and integrate AI into our

Performance:
  Total time:  12.972s
    - Load:    4.612s
    - Generate:8.360s
  Total tokens: 20
  Throughput:  2.39 tokens/sec
  Per-sample:  2.39 tokens/sec
```

## Implementation Considerations

This section documents optimization approaches that were explored but didn't yield the expected benefits for this specific model and hardware configuration. These insights are valuable for understanding the trade-offs when porting models to Tenstorrent hardware.

### What We Tried

#### 1. Full TTNN Tensor Conversion
**Approach:** Convert all operations to native TTNN tensors for maximum hardware acceleration.

**Challenge:** TTNN uses `bfloat16` by default, while PyTorch operations use `float32`.

**What breaks with full bfloat16:**
- **Mamba SSM state**: Accumulates errors over time, diverges after 10-15 tokens
- **MoE router**: Similar expert logits round to same value, picks wrong experts
- **Attention softmax**: Long sequences underflow, loses distant token information
- **Layer-by-layer accumulation**: Errors compound across 40 layers

**Decision:** Use TTNN bfloat16 only for heavy matmuls (QKV, MLP). Keep PyTorch FP32 for precision-critical operations (Mamba state, MoE routing, attention softmax, layer norms). 

#### 2. Tensor Parallelism for Multi-Device Scaling
**Approach:** Distribute model across multiple Tenstorrent devices using tensor parallelism to increase throughput.

**Challenge:** Communication overhead between devices dominated any compute savings. The Granite 1B Hybrid model is small enough to fit comfortably in a single device's memory (~1.2GB weights), so splitting it across devices only added:
- Cross-device tensor synchronization latency
- Bandwidth bottlenecks during weight transfers
- Additional complexity in managing distributed state

**Observation:** Mamba-1 and similar small models don't benefit from tensor parallelism. Unlike large transformers (>70B parameters), the communication cost exceeds compute savings. This aligns with Mamba's design philosophy—efficient SSMs that fit in single-device memory.

**Decision:** Use single-device deployment. Multi-device fabric is supported in the code but only beneficial for:
- Data parallelism (processing multiple prompts in parallel)
- Larger models (>10B parameters)

#### 3. L1 Memory Optimization
**Approach:** Move frequently accessed tensors (activations, attention scores) to L1 SRAM for faster access.

**Challenge:** L1 memory constraints. Tenstorrent devices have limited L1 SRAM per core:
- Granite Hybrid has 24 decoder layers with Mamba blocks
- Each Mamba block maintains SSM state (d_state=16, d_model=1536)
- Attention blocks require KV cache storage
- During prefill, intermediate activations for full sequence don't fit in L1