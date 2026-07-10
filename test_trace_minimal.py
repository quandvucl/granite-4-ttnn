"""Minimal trace test: does Metal trace work at all on this hardware?"""
import os, sys, torch, ttnn

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/work/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto",
)

def main():
    print("Opening galaxy 8x4, carving 1x4 submesh (mimics real model setup)...")
    GALAXY_SHAPE = ttnn.MeshShape(8, 4)
    full_mesh = ttnn.open_mesh_device(mesh_shape=GALAXY_SHAPE, trace_region_size=268435456)
    device = full_mesh.create_submeshes(ttnn.MeshShape(1, 4))[0]
    mapper = ttnn.ReplicateTensorToMesh(device)

    def make(shape, val=0.0):
        return ttnn.from_torch(
            torch.full(shape, val, dtype=torch.bfloat16),
            device=device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # Fixed-address inputs (like _decode_trace_in, _ssm_state_tt)
    a = make([1, 1, 32, 32], 1.0)
    b = make([1, 1, 32, 32], 2.0)

    print("Compile run...")
    c_compile = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c_compile.deallocate(True)
    ttnn.synchronize_device(device)
    print("Compile run done.")

    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    c_trace = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    print("Trace captured.")

    print("Executing trace (blocking=True)...")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    print("Trace executed!")

    result = ttnn.to_torch(c_trace, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0]
    print(f"Result[0,0,0] = {result[0,0,0].item()} (expected 3.0)")

    ttnn.release_trace(device, trace_id)
    a.deallocate(True); b.deallocate(True); c_trace.deallocate(True)
    ttnn.close_mesh_device(device)
    ttnn.close_mesh_device(full_mesh)
    print("PASS")

if __name__ == "__main__":
    main()
