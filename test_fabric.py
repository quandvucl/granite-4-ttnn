#!/usr/bin/env python3
"""
Test fabric collective operations on 32-device mesh.
This will help us understand what works and what doesn't.
"""
import torch
import ttnn
import sys

def test_mesh_setup():
    """Test basic mesh device setup."""
    print("="*70)
    print("TEST 1: Mesh Device Setup")
    print("="*70)

    try:
        # Set fabric config
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

        # Open 32-device mesh (4x8)
        mesh_shape = ttnn.MeshShape(4, 8)
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        num_devices = device.get_num_devices()
        print(f"✓ Opened mesh device: {num_devices} devices")
        print(f"  Mesh shape: {device.shape}")
        print(f"  Compute grid: {device.compute_with_storage_grid_size()}")

        ttnn.close_mesh_device(device)
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_replicated_tensors():
    """Test creating replicated tensors."""
    print("\n" + "="*70)
    print("TEST 2: Replicated Tensors")
    print("="*70)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_shape = ttnn.MeshShape(4, 8)
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        # Create a simple tensor
        torch_tensor = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

        # Replicate across all devices
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper
        )

        print(f"✓ Created replicated tensor: {tt_tensor.shape}")

        # Get shards from all devices
        shards = ttnn.get_device_tensors(tt_tensor)
        print(f"✓ Got {len(shards)} shards from devices")

        # Verify all shards are identical
        shard0 = shards[0].cpu().to_torch()
        all_identical = all(
            torch.allclose(shard.cpu().to_torch(), shard0, rtol=1e-3, atol=1e-3)
            for shard in shards
        )

        if all_identical:
            print(f"✓ All shards are identical (replicated correctly)")
        else:
            print(f"✗ Shards differ (replication failed)")

        ttnn.close_mesh_device(device)
        return all_identical
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_reduce():
    """Test ttnn.all_reduce with different configurations."""
    print("\n" + "="*70)
    print("TEST 3: All-Reduce Operations")
    print("="*70)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_shape = ttnn.MeshShape(4, 8)
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        # Create tensor with different values on each device
        torch_tensor = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)

        # Replicate (all devices get same value initially)
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper
        )

        print(f"Created input tensor (all 1s): {tt_tensor.shape}")

        # Try different all_reduce configurations
        configs = [
            {"cluster_axis": 0, "num_links": 1, "topology": None},
            {"cluster_axis": 1, "num_links": 1, "topology": None},
            {"cluster_axis": None, "num_links": 1, "topology": ttnn.ccl.Topology.Linear},
            {"cluster_axis": None, "num_links": 1, "topology": ttnn.ccl.Topology.Ring},
        ]

        for i, config in enumerate(configs):
            try:
                print(f"\nTrying config {i+1}: {config}")
                result = ttnn.all_reduce(
                    tt_tensor,
                    **config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                print(f"  ✓ all_reduce succeeded!")

                # Check result
                result_torch = result.cpu().to_torch()
                print(f"  Result: min={result_torch.min():.1f}, max={result_torch.max():.1f}, mean={result_torch.mean():.1f}")

                result.deallocate(True)
                return True
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

        print(f"\n✗ All all_reduce configs failed")
        ttnn.close_mesh_device(device)
        return False

    except Exception as e:
        print(f"✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sharded_tensors():
    """Test creating sharded tensors."""
    print("\n" + "="*70)
    print("TEST 4: Sharded Tensors")
    print("="*70)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_shape = ttnn.MeshShape(4, 8)
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        # Create a larger tensor to shard
        torch_tensor = torch.randn(1, 1, 64, 1024, dtype=torch.bfloat16)

        # Try column-wise sharding (along width)
        print("Trying column-wise sharding (width dimension)...")
        mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(-1, None), mesh_shape=mesh_shape)
        tt_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper
        )

        print(f"✓ Created sharded tensor: {tt_tensor.shape}")

        # Get shards
        shards = ttnn.get_device_tensors(tt_tensor)
        print(f"✓ Got {len(shards)} shards")
        print(f"  Shard 0 shape: {shards[0].shape}")

        ttnn.close_mesh_device(device)
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("TTNN Fabric Diagnostics for 32-Device Mesh (4x8)")
    print("="*70 + "\n")

    results = {
        "mesh_setup": test_mesh_setup(),
        "replicated": test_replicated_tensors(),
        "all_reduce": test_all_reduce(),
        "sharded": test_sharded_tensors(),
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test:20s}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
