"""
TTNN MoE for GraniteMoeHybrid — expert-parallel sharding across mesh columns.

1×4 (tiny): shard 9 experts/col, all_gather_async(axis=1, Linear)+sum.
2×4 (small): flat EP across 8 devices (9 experts/device), ShardTensorToMesh(dim=1).
  Full-mesh all-reduce: ttnn.all_gather(dim=1, no cluster_axis)+sum — trace-safe, no semaphores.
Decode router runs fully on-device (topk+softmax+embedding, no PCIe round-trip).
Prefill router runs on CPU (PCIe cost amortized over S tokens).
Decode trace: all_gather_async with persistent_output_buffer=None; output allocated in
  trace region so no host write during capture. Warmup step 1 compiles the program;
  trace capture (step 2) gets a cache hit.
"""

import torch
import ttnn
from utils.device import _is_mesh_device


class GraniteTTMoE:
    def __init__(self, hf_moe, device, weight_dtype=ttnn.bfloat8_b, act_dtype=ttnn.bfloat16,
                 use_all_gather=True, tt_ccl=None):
        self.hf_moe = hf_moe
        self.device = device
        self.dtype = weight_dtype
        self.act_dtype = act_dtype
        self.tt_ccl = tt_ccl

        self.is_mesh = _is_mesh_device(device)
        self.num_devices = device.get_num_devices() if self.is_mesh else 1

        self._topology = ttnn.Topology.Linear

        self.num_experts = hf_moe.input_linear.num_experts
        self.hidden_size = hf_moe.input_size
        self.intermediate_size = hf_moe.hidden_size
        self.top_k = hf_moe.router.top_k

        if self.is_mesh and self.num_devices > 1:
            mesh_shape = device.shape
            self._mesh_rows = mesh_shape[0]
            self._mesh_cols = mesh_shape[1]
        else:
            self._mesh_rows = 1
            self._mesh_cols = 1

        # EP sharding strategy:
        #
        # Single-row (tiny, 1×4): col-parallel EP.
        #   Shard experts across 4 cols (9/device); all_gather_async(axis=1)+sum.
        #
        # Multi-row (small, 2×4): flat EP across all 8 devices (9/device).
        #   ShardTensorToMesh(dim=1) maps experts 0..8→dev0, 9..17→dev1, ...
        #   Full-mesh all-reduce: ttnn.all_gather(dim=1, num_links=1) — no cluster_axis,
        #   trace-safe, no CCL semaphores. Slower than async but correct in trace.
        #   (Two-axis all_gather_async tried: cluster_axis=0 caused trace corruption.)
        self._use_col_parallel = (self._mesh_rows == 1) and self.num_devices > 1 and use_all_gather
        self._use_row_reduce = False  # disabled; using flat ttnn.all_gather for multi-row

        if self._use_col_parallel:
            effective_cols = self._mesh_cols
            while effective_cols > 1 and self.num_experts % effective_cols != 0:
                effective_cols -= 1
            self.effective_cols = effective_cols
            self.experts_per_device = self.num_experts // effective_cols
            self._ep_mapper = ttnn.ShardTensor2dMesh(
                self.device, dims=(None, 1),
                mesh_shape=ttnn.MeshShape(1, effective_cols),
            ) if effective_cols > 1 else (
                ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None
            )
        else:
            # Multi-row: flat EP across all devices.
            effective_devices = self.num_devices
            while effective_devices > 1 and self.num_experts % effective_devices != 0:
                effective_devices -= 1
            self.effective_cols = effective_devices
            self.experts_per_device = self.num_experts // effective_devices
            self._ep_mapper = ttnn.ShardTensorToMesh(
                self.device, dim=1,
            ) if effective_devices > 1 else (
                ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None
            )

        self._load_weights()
        self._load_router_weight()

    def _load_router_weight(self):
        replicate_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None
        router_w = self.hf_moe.router.layer.weight.to(torch.bfloat16).T.contiguous()
        self.router_weight_tt = ttnn.from_torch(
            router_w,
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper,
        )
        self._load_routing_matrices(replicate_mapper)

    def _load_routing_matrices(self, replicate_mapper):
        E = self.num_experts
        eye = torch.eye(E, dtype=torch.bfloat16)
        # Replicated [E, E] identity in ROW_MAJOR for embedding lookup (all devices need all rows)
        self._routing_eye_tt = ttnn.from_torch(
            eye, device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper,
        )
        # Column-sharded [E, E_local] identity for routing projection: [1,E] @ [E,E_local] = [1,E_local]
        # Using _ep_mapper ensures the col sharding matches the expert weight sharding.
        self._routing_proj_tt = ttnn.from_torch(
            eye, device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )

    def _load_weights(self):
        import sys
        W_in_4d  = self.hf_moe.input_linear.weight.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)
        W_out_4d = self.hf_moe.output_linear.weight.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)

        # Upload as bfloat16 (fast DMA), then cast on-device to target dtype.
        # Avoids ttnn's slow CPU-side tile-by-tile quantization for bfloat8_b
        # (~375s for 72-expert small model). On-device typecast takes <1s.
        print(f"  [moe_tt] uploading gate_up bf16 shape={list(W_in_4d.shape)} dtype={self.dtype}", flush=True)
        gate_up_bf16 = ttnn.from_torch(
            W_in_4d,
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        print(f"  [moe_tt] gate_up uploaded", flush=True)
        if self.dtype != ttnn.bfloat16:
            print(f"  [moe_tt] typecasting gate_up to {self.dtype}", flush=True)
            self.gate_up_proj = ttnn.typecast(gate_up_bf16, self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_up_bf16.deallocate(True)
            print(f"  [moe_tt] gate_up typecast done", flush=True)
        else:
            self.gate_up_proj = gate_up_bf16

        print(f"  [moe_tt] uploading down_proj bf16 shape={list(W_out_4d.shape)}", flush=True)
        down_bf16 = ttnn.from_torch(
            W_out_4d,
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        print(f"  [moe_tt] down_proj uploaded", flush=True)
        if self.dtype != ttnn.bfloat16:
            print(f"  [moe_tt] typecasting down_proj to {self.dtype}", flush=True)
            self.down_proj = ttnn.typecast(down_bf16, self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            down_bf16.deallocate(True)
            print(f"  [moe_tt] down_proj typecast done", flush=True)
        else:
            self.down_proj = down_bf16

    def compute_routing_cpu(self, hidden_states_tt):
        """
        Compute routing weights on CPU from device logits.
        Returns routing_tt [1, E_local, S, 1] on device.
        """
        E = self.num_experts
        S = hidden_states_tt.shape[2]
        logits_tt = ttnn.linear(
            hidden_states_tt, self.router_weight_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.is_mesh:
            logits_cpu = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))[0:1]
        else:
            logits_cpu = logits_tt.cpu().to_torch()
        logits_tt.deallocate(True)
        return self._routing_from_logits(logits_cpu.reshape(-1, E)[:S], S)

    def _routing_from_logits(self, logits_SE, S):
        """CPU: logits_SE [S, E] → routing_tt [1, E_local, S, 1] on device."""
        E = self.num_experts
        top_k_logits, top_k_indices = logits_SE.topk(self.top_k, dim=1)
        top_k_weights = torch.softmax(top_k_logits.float(), dim=1).to(torch.bfloat16)
        routing_SE = torch.zeros(S, E, dtype=torch.bfloat16)
        routing_SE.scatter_add_(1, top_k_indices, top_k_weights)
        routing_4d = routing_SE.T.unsqueeze(-1).unsqueeze(0)  # [1, E, S, 1]
        return ttnn.from_torch(
            routing_4d, device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )

    def compute_routing_device(self, hidden_states_tt):
        """
        Fully on-device routing for decode (S=1). Returns [1, E_local, 1, 1] per device.
        Uses topk + softmax + embedding-scatter to avoid PCIe round-trip.
        """
        E       = self.num_experts
        E_local = self.experts_per_device
        L1 = ttnn.L1_MEMORY_CONFIG

        logits_tt = ttnn.linear(
            hidden_states_tt, self.router_weight_tt,
            dtype=ttnn.bfloat16, memory_config=L1,
        )

        vals_tt, idxs_tt = ttnn.topk(logits_tt, k=self.top_k, dim=3)
        logits_tt.deallocate(True)

        weights_tt = ttnn.softmax(vals_tt, dim=3, memory_config=L1)
        vals_tt.deallocate(True)

        idxs_u32 = ttnn.typecast(idxs_tt, ttnn.uint32)
        idxs_tt.deallocate(True)

        idxs_2d = ttnn.reshape(idxs_u32, [1, self.top_k])
        idxs_u32.deallocate(True)

        onehot_tt = ttnn.embedding(
            idxs_2d, self._routing_eye_tt,
            layout=ttnn.TILE_LAYOUT, memory_config=L1,
        )
        idxs_2d.deallocate(True)

        weights_3d = ttnn.reshape(weights_tt, [1, self.top_k, 1])
        weights_tt.deallocate(True)

        scaled_tt = ttnn.mul(onehot_tt, weights_3d, memory_config=L1)
        onehot_tt.deallocate(True)
        weights_3d.deallocate(True)

        scaled_4d = ttnn.reshape(scaled_tt, [1, 1, self.top_k, E])
        scaled_tt.deallocate(True)
        routing_4d = ttnn.sum(scaled_4d, dim=2, keepdim=True, memory_config=L1)
        scaled_4d.deallocate(True)

        routing_local = ttnn.linear(
            routing_4d, self._routing_proj_tt,
            dtype=ttnn.bfloat16, memory_config=L1,
        )
        routing_4d.deallocate(True)

        routing_tt = ttnn.reshape(routing_local, [1, E_local, 1, 1])
        routing_local.deallocate(True)
        return routing_tt

    def _all_gather(self, x, dim, cluster_axis, memory_config):
        if self.tt_ccl is not None:
            return ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=dim,
                cluster_axis=cluster_axis,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                num_links=self.tt_ccl.get_num_links(cluster_axis),
                memory_config=memory_config,
                topology=self._topology,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return ttnn.all_gather(x, dim=dim, cluster_axis=cluster_axis, memory_config=memory_config)

    def forward(self, hidden_states_tt, routing_tt=None):
        """
        hidden_states_tt: TTNN [1, 1, S, H] replicated.
        routing_tt: optional precomputed [1, E_local, S, 1] — skips router if provided.
        Caller must NOT deallocate hidden_states_tt — also needed for shared MLP.
        Returns: TTNN [1, 1, S, H] replicated.
        """
        I       = self.intermediate_size
        E       = self.num_experts
        E_local = self.experts_per_device
        S       = hidden_states_tt.shape[2]
        MC      = ttnn.DRAM_MEMORY_CONFIG

        # ── 1. ROUTER ─────────────────────────────────────────────────────────
        if routing_tt is None:
            routing_tt = getattr(self, '_precomputed_routing', None)
        self._precomputed_routing = None  # consume it
        if routing_tt is None:
            if S == 1 and self._use_col_parallel:
                routing_tt = self.compute_routing_device(hidden_states_tt)
            else:
                routing_tt = self.compute_routing_cpu(hidden_states_tt)
        # per device: routing_tt [1, E_local, S, 1]

        # ── 2. GATE+UP ────────────────────────────────────────────────────────
        hidden_exp = ttnn.repeat(
            hidden_states_tt, ttnn.Shape([1, E_local, 1, 1]),
            memory_config=MC,
        )
        gate_up_out = ttnn.matmul(hidden_exp, self.gate_up_proj,
                                  dtype=self.act_dtype, memory_config=MC)
        hidden_exp.deallocate(True)

        # ── 3. ACTIVATION ────────────────────────────────────────────────────
        gate_tt = gate_up_out[:, :, :, :I]
        up_tt   = gate_up_out[:, :, :, I:]
        gate_up_out.deallocate(True)

        activated_tt = ttnn.mul(ttnn.silu(gate_tt, memory_config=MC), up_tt, memory_config=MC)
        gate_tt.deallocate(True)
        up_tt.deallocate(True)

        # ── 4. DOWN PROJECTION ───────────────────────────────────────────────
        expert_out_tt = ttnn.matmul(activated_tt, self.down_proj,
                                    dtype=self.act_dtype, memory_config=MC)
        activated_tt.deallocate(True)

        # ── 5. WEIGHT by routing ─────────────────────────────────────────────
        weighted_tt = ttnn.mul(expert_out_tt, routing_tt, memory_config=MC)
        expert_out_tt.deallocate(True)
        routing_tt.deallocate(True)

        # ── 6. Local sum then all_gather + sum across columns ─────────────────
        local_sum_tt = ttnn.sum(weighted_tt, dim=1, keepdim=True, memory_config=MC)
        weighted_tt.deallocate(True)

        if self._use_col_parallel and self.effective_cols > 1:
            # Single-row mesh (tiny): async col gather + sum.
            gathered_tt = self._all_gather(local_sum_tt, dim=1, cluster_axis=1, memory_config=MC)
            local_sum_tt.deallocate(True)
            result_tt = ttnn.sum(gathered_tt, dim=1, keepdim=True, memory_config=MC)
            gathered_tt.deallocate(True)
        elif not self._use_col_parallel and self.effective_cols > 1:
            # Multi-row mesh (small, 2×4): two-axis all_reduce, trace-safe.
            # axis=1 reduces across 4 cols, axis=0 reduces across 2 rows.
            after_cols = ttnn.all_reduce(local_sum_tt, cluster_axis=1, memory_config=MC)
            local_sum_tt.deallocate(True)
            result_tt = ttnn.all_reduce(after_cols, cluster_axis=0, memory_config=MC)
            after_cols.deallocate(True)
        else:
            result_tt = local_sum_tt

        return result_tt
