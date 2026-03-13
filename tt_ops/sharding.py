"""
Utilities for sharding tensors across mesh devices for tensor parallelism.
"""
import ttnn
import torch
from typing import Optional


def shard_linear_weight_column_wise(
    weight: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype=ttnn.bfloat16
) -> ttnn.Tensor:
    """
    Shard a linear layer weight [in_features, out_features] column-wise across mesh.

    Each device gets weight[:, start:end] where the column dimension is split.
    This enables parallel matrix multiplication followed by all-reduce.

    Args:
        weight: PyTorch weight tensor [in_features, out_features]
        mesh_device: TTNN mesh device
        dtype: Target dtype

    Returns:
        TTNN tensor sharded across mesh devices (column-wise)
    """
    num_devices = mesh_device.get_num_devices()
    out_features = weight.shape[1]

    if out_features % num_devices != 0:
        raise ValueError(f"out_features {out_features} must be divisible by num_devices {num_devices}")

    shard_size = out_features // num_devices

    # Create sharded tensor using ShardTensorToMesh
    # Shard along dimension 1 (columns)
    tt_weight = ttnn.Tensor(weight, dtype)
    tt_weight = ttnn.to_layout(tt_weight, ttnn.TILE_LAYOUT)

    # Use ShardTensorToMesh to distribute columns across devices
    tt_weight_sharded = ttnn.to_device(
        tt_weight,
        mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)
    )

    return tt_weight_sharded


def shard_linear_weight_row_wise(
    weight: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype=ttnn.bfloat16
) -> ttnn.Tensor:
    """
    Shard a linear layer weight [in_features, out_features] row-wise across mesh.

    Each device gets weight[start:end, :] where the row dimension is split.
    Requires input to be sharded the same way.

    Args:
        weight: PyTorch weight tensor [in_features, out_features]
        mesh_device: TTNN mesh device
        dtype: Target dtype

    Returns:
        TTNN tensor sharded across mesh devices (row-wise)
    """
    num_devices = mesh_device.get_num_devices()
    in_features = weight.shape[0]

    if in_features % num_devices != 0:
        raise ValueError(f"in_features {in_features} must be divisible by num_devices {num_devices}")

    tt_weight = ttnn.Tensor(weight, dtype)
    tt_weight = ttnn.to_layout(tt_weight, ttnn.TILE_LAYOUT)

    # Shard along dimension 0 (rows)
    tt_weight_sharded = ttnn.to_device(
        tt_weight,
        mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    return tt_weight_sharded


def replicate_to_mesh(
    tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT
) -> ttnn.Tensor:
    """
    Replicate a tensor to all devices in mesh (for small tensors like biases, norms).

    Args:
        tensor: PyTorch tensor
        mesh_device: TTNN mesh device
        dtype: Target dtype
        layout: Target layout

    Returns:
        TTNN tensor replicated across all mesh devices
    """
    tt_tensor = ttnn.Tensor(tensor, dtype)
    tt_tensor = tt_tensor.to(layout)

    # Use ReplicateTensorToMesh for broadcasting
    tt_tensor_replicated = ttnn.to_device(
        tt_tensor,
        mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )

    return tt_tensor_replicated


def all_reduce_across_mesh(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice
) -> ttnn.Tensor:
    """
    Perform all-reduce (sum) across all devices in mesh.

    Used after column-sharded matmul to combine partial results.

    Args:
        tensor: TTNN tensor (can be sharded or replicated)
        mesh_device: TTNN mesh device

    Returns:
        TTNN tensor with results summed across all devices
    """
    # Use ttnn.all_reduce for collective communication
    reduced = ttnn.all_reduce(
        tensor,
        mesh_device=mesh_device,
        op=ttnn.ReduceOp.SUM
    )

    return reduced


def column_parallel_linear(
    input_tt: ttnn.Tensor,
    weight_sharded: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    bias: Optional[ttnn.Tensor] = None
) -> ttnn.Tensor:
    """
    Perform column-parallel linear layer:
    1. Input (replicated) @ Weight (column-sharded)
    2. Each device computes part of output
    3. All-reduce to combine results

    Args:
        input_tt: Input tensor (replicated across devices) [batch, seq, in_features]
        weight_sharded: Weight tensor (column-sharded) [in_features, out_features/N]
        mesh_device: TTNN mesh device
        bias: Optional bias (replicated)

    Returns:
        Output tensor [batch, seq, out_features] (replicated after all-reduce)
    """
    # Each device: [batch, seq, in_features] @ [in_features, out_features/N]
    # Result: [batch, seq, out_features/N] on each device
    partial_output = input_tt @ weight_sharded

    # All-reduce: sum partial outputs from all devices
    # Result: [batch, seq, out_features] (replicated)
    output = all_reduce_across_mesh(partial_output, mesh_device)

    if bias is not None:
        output = output + bias

    return output


def row_parallel_linear(
    input_sharded: ttnn.Tensor,
    weight_sharded: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    bias: Optional[ttnn.Tensor] = None
) -> ttnn.Tensor:
    """
    Perform row-parallel linear layer:
    1. Input (sharded) @ Weight (row-sharded)
    2. Each device computes using its shard
    3. All-reduce to combine results

    Args:
        input_sharded: Input tensor (sharded along last dim) [batch, seq, in_features/N]
        weight_sharded: Weight tensor (row-sharded) [in_features/N, out_features]
        mesh_device: TTNN mesh device
        bias: Optional bias (replicated)

    Returns:
        Output tensor [batch, seq, out_features] (replicated after all-reduce)
    """
    # Each device: [batch, seq, in_features/N] @ [in_features/N, out_features]
    partial_output = input_sharded @ weight_sharded

    # All-reduce to sum partial results
    output = all_reduce_across_mesh(partial_output, mesh_device)

    if bias is not None:
        output = output + bias

    return output
