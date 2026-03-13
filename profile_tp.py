#!/usr/bin/env python3
"""
Profile tensor parallelism to understand where time is spent.
"""
import time
import torch
import ttnn
from transformers import AutoTokenizer

def profile_single_layer():
    """Profile a single MLP forward pass with tensor parallelism."""
    print("="*70)
    print("Profiling Single MLP Layer with Tensor Parallelism (32 devices)")
    print("="*70)

    # Setup
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(4, 8)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    hidden_size = 1536
    intermediate_size = 4096

    # Create weights
    gate_weight = torch.randn(1, 1, hidden_size, intermediate_size, dtype=torch.bfloat16)
    up_weight = torch.randn(1, 1, hidden_size, intermediate_size, dtype=torch.bfloat16)
    down_weight = torch.randn(1, 1, intermediate_size, hidden_size, dtype=torch.bfloat16)

    # Shard weights
    print("\n1. Sharding weights...")
    t0 = time.time()
    col_mapper = ttnn.ShardTensor2dMesh(device, dims=(-1, None), mesh_shape=mesh_shape)
    row_mapper = ttnn.ShardTensor2dMesh(device, dims=(-2, None), mesh_shape=mesh_shape)

    gate_tt = ttnn.from_torch(gate_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, mesh_mapper=col_mapper)
    up_tt = ttnn.from_torch(up_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, mesh_mapper=col_mapper)
    down_tt = ttnn.from_torch(down_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, mesh_mapper=row_mapper)
    t1 = time.time()
    print(f"   Time: {(t1-t0)*1000:.1f}ms")

    # Create input (replicated)
    print("\n2. Creating replicated input...")
    t0 = time.time()
    input_torch = torch.randn(1, 1, 1, hidden_size, dtype=torch.bfloat16)
    rep_mapper = ttnn.ReplicateTensorToMesh(device)
    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=device, mesh_mapper=rep_mapper)
    t1 = time.time()
    print(f"   Time: {(t1-t0)*1000:.1f}ms")

    # Forward pass
    print("\n3. Column-parallel matmuls (gate + up)...")
    t0 = time.time()
    gate_out = input_tt @ gate_tt  # Each device gets 1/32 of output
    up_out = input_tt @ up_tt
    t1 = time.time()
    print(f"   Time: {(t1-t0)*1000:.1f}ms")

    print("\n4. SwiGLU activation...")
    t0 = time.time()
    activated = ttnn.mul(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    t1 = time.time()
    print(f"   Time: {(t1-t0)*1000:.1f}ms")

    print("\n5. Row-parallel matmul (down)...")
    t0 = time.time()
    output_partial = activated @ down_tt  # Each device has partial result
    t1 = time.time()
    print(f"   Time: {(t1-t0)*1000:.1f}ms")

    print("\n6. CPU reduction (gather-sum-broadcast)...")
    t0 = time.time()

    # Gather shards
    shards = ttnn.get_device_tensors(output_partial)
    t_gather = time.time()
    print(f"   Gather: {(t_gather-t0)*1000:.1f}ms")

    # Sum on CPU
    summed = shards[0].cpu().to_torch().clone()
    for i in range(1, len(shards)):
        summed.add_(shards[i].cpu().to_torch())
    t_sum = time.time()
    print(f"   Sum: {(t_sum-t_gather)*1000:.1f}ms")

    # Broadcast back
    output_tt = ttnn.from_torch(summed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=device, mesh_mapper=rep_mapper)
    t1 = time.time()
    print(f"   Broadcast: {(t1-t_sum)*1000:.1f}ms")
    print(f"   Total reduction: {(t1-t0)*1000:.1f}ms")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Matmuls (parallelized):  {((t1-t0) - (t1-t0))*1000:.1f}ms")  # Placeholder
    print(f"  CPU reduction:           {(t1-t0)*1000:.1f}ms")
    print(f"\nConclusion: If reduction > compute, TP has no benefit")

    ttnn.close_mesh_device(device)


def profile_full_model():
    """Profile full model forward pass."""
    print("\n" + "="*70)
    print("Profiling Full Model Forward Pass")
    print("="*70)

    from tt_model.model import TTGraniteMoeHybridForCausalLM

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device, all_devices = open_device(32)

    try:
        tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-h-1b')

        print("\nLoading model...")
        tt_model = TTGraniteMoeHybridForCausalLM.from_pretrained(
            'ibm-granite/granite-4.0-h-1b',
            device,
            verbose=True
        )

        # Warmup
        warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
        _ = tt_model.forward(warmup_ids)

        # Timed run
        input_ids = tokenizer("The future of AI is", return_tensors="pt")["input_ids"]

        print("\nTiming forward pass...")
        t0 = time.time()
        _ = tt_model.forward(input_ids)
        t1 = time.time()

        print(f"\nForward pass time: {(t1-t0)*1000:.1f}ms")

    finally:
        close_device(device, all_devices)


def open_device(num_devices):
    """Open device helper."""
    if num_devices == 32:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        full_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
        return full_mesh, full_mesh
    else:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        return device, None


def close_device(device, all_devices):
    """Close device helper."""
    if all_devices is not None:
        ttnn.close_mesh_device(all_devices)
    else:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    # Profile just the critical path
    profile_single_layer()

    # Uncomment to profile full model
    # profile_full_model()
