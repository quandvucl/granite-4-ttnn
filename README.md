# TT-Granite: Granite MoE Hybrid on TT-Metal/TTNN

Implementation of IBM's Granite 4.0 MoE Hybrid model (tiny and small variants) on Tenstorrent's TT-Metal hardware.

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

## Running Benchmarks

Both scripts measure model load time, prefill (TTFT), and decode throughput across five prompt lengths (8–256 tokens). Results are saved to JSON files.

### `test_bench.py` — TT hardware

Runs the model on Tenstorrent hardware via TTNN.

```bash
# Single model
python test_bench.py --model tiny
python test_bench.py --model small

# More decode steps (default: 20)
python test_bench.py --decode-tokens 50

# Mamba prefill chunk size (default: 256)
python test_bench.py --chunk-size 128

# Disable fused Metal kernels (fall back to TTNN ops)
python test_bench.py --no-conv1d-kernel
python test_bench.py --no-ssm-kernel
python test_bench.py --no-conv1d-kernel --no-ssm-kernel
```

**Output files** are named by model, device count, chunk size, and kernel flags, e.g.:
- `bench_results_small_conv1d_ssm.json` — default (both kernels on)
- `bench_results_small_32_chunk128_conv1d_ssm.json` — 32 devices, chunk 128
- `bench_results_small_no_kernels.json` — both kernels disabled

### `test_bench_hf.py` — CPU / CUDA baseline

Runs the HuggingFace model on CPU or CUDA for reference numbers.

```bash
# Both models on CPU (default)
python test_bench_hf.py

# Single model
python test_bench_hf.py --model tiny
python test_bench_hf.py --model small

# CUDA
python test_bench_hf.py --device cuda

# Enable torch.compile (disabled by default)
python test_bench_hf.py --compile
python test_bench_hf.py --device cuda --compile

# More decode steps
python test_bench_hf.py --decode-tokens 50
```

**Output files**: `bench_results_hf_{model}_{device}[_compile].json`, e.g.:
- `bench_results_hf_small_cpu.json`
- `bench_results_hf_small_cuda_compile.json`

### `summarise_results.py` — aggregate results

Collects all `*.json` result files from `./report_results/` (recursively) and merges them into a single `./report_results/summary.json`.

```bash
# 1. Move or copy bench output files into report_results/
mkdir -p report_results
cp bench_results_*.json bench_results_hf_*.json report_results/

# 2. Run the aggregator
python summarise_results.py
```