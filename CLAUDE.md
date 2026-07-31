# CLAUDE.md

## Python Environment

Always run Python commands inside the virtual environment:

```
source env/bin/activate && python ...
```

Never run `python` or `pip` directly without activating the virtualenv first.

## Hardware Resets

**Avoid any action that requires a hardware reset.** Resets take a long time and are disruptive.

## Do Not Modify

**`models/`** — vendored tt-metal model library (RMSNorm, Attention1D, LMHead1D, etc.). Never edit files here.

## Codebase Layout

```
granite/          # Top-level model: TTGraniteMoeHybridForCausalLM
  model.py          forward, chunked prefill, decode trace capture/replay
  decoder_layer.py  TTGraniteDecoderLayer (Mamba or Attention + MoE + SharedMLP)
  attention_nope.py AttentionNoPE — fused QKV, paged KV cache, NoPE
  moe_tt.py         GraniteTTMoE — expert-parallel (EP) batched matmul + all-gather
  config.py         TTGraniteConfig (architecture + TT settings)
  cache.py          MambaCacheManager — decode position counter; hybrid_cache attached externally

mamba/            # Mamba2 tensor-parallel implementation
  mamba_chunk_scan_parallel.py  TensorParallelMamba: prefill chunk-scan + decode recurrence
  config.py         Mamba2Config dataclass
  ssm_utils.py      extract_ssm_parameters (post-conv split + group expansion)
  utils.py          segment_sum_ttnn, make_segment_sum_masks

kernel/           # Custom Metal kernels
  ssm_update/
    op.py           ssm_update kernel launcher (fused h-update + y-reduce, plan-cached)
    compute/        Tensix compute kernel (Metal C++)
    dataflow/       Reader/writer kernels (Metal C++)

utils/            # Shared TTNN helpers
  base.py           to_tt_tensor, to_torch_tensor
  device.py         _is_mesh_device, _make_mesh_mapper, _to_tt, softplus_and_clamp_*

models/           # Vendored tt-metal library — DO NOT EDIT
```

## Key Execution Notes

- **Decode trace**: captured once after warmup via `capture_decode_trace()`, replayed for every subsequent decode step. Trace runs in bypass mode (ops recorded, not executed) so it must be captured *outside* the autoregressive loop with a dummy input.
- **Mesh shapes**: tiny uses `MeshShape(1,4)`, small uses `MeshShape(8,1)`. The 8×1 shape gives `is_true_2d_mesh()=False`, making `all_gather_async` trace-safe.
- **SSM state address stability**: `_ssm_state_tt` is a fixed-address DRAM tensor. Prefill updates it in-place via `ttnn.clone` + `ttnn.assign` to preserve the address baked into the trace.
- **MoE expert parallelism**: weights sharded across devices; routing computed on-device for decode to avoid PCIe round-trip.

## Benchmark Entry Points

```
python test_bench.py          # TT hardware benchmark (tiny + small)
python test_bench_hf.py       # HuggingFace CPU/CUDA baseline
```

Results saved to `report_results/`.
