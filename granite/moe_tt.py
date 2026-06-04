"""
TTNN MoE for GraniteMoeHybrid — expert-parallel sharding across mesh columns.

Single-row mesh (tiny, 1×4): shard experts across cols; all_gather(axis=1)+sum on device.
Multi-row mesh (small, 2×4): flat EP across 8 devices; local sum then CPU reduce.
  Two-axis on-device reduce was tried but row-axis all_gather latency hurt prefill (-53%).
Decode router runs fully on-device (topk+softmax+embedding, no PCIe round-trip).
Prefill router runs on CPU (PCIe cost amortized over S tokens).
"""

import torch
import ttnn
from utils.device import _is_mesh_device


class GraniteTTMoE:
    def __init__(self, hf_moe, device, weight_dtype=ttnn.bfloat8_b, act_dtype=ttnn.bfloat16,
                 use_all_gather=True):
        self.hf_moe = hf_moe
        self.device = device
        self.dtype = weight_dtype
        self.act_dtype = act_dtype

        self.is_mesh = _is_mesh_device(device)
        self.num_devices = device.get_num_devices() if self.is_mesh else 1

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
        # Single-row (tiny, 1×4): shard across 4 cols; all_gather(axis=1) + sum on device.
        #
        # Multi-row (small, 2×4): flat EP across 8 devices (9 experts/device).
        #   CPU reduce: download partial sums, sum on CPU, re-upload.
        #   Two-axis on-device reduce tested but reverted: row-axis all_gather latency
        #   dominates for long sequences, hurting prefill (-53%) despite decode gains (+14%).
        self._use_col_parallel = (self._mesh_rows == 1) and self.num_devices > 1 and use_all_gather
        # Multi-row decode: all_gather across rows (cluster_axis=0) for S=1 only.
        # Prefill still uses CPU reduce (row all_gather latency dominates long sequences).
        self._use_row_reduce = (self._mesh_rows > 1) and use_all_gather

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
            # Decode: on-device row all_gather + col all_gather (if use_row_reduce).
            # Prefill: CPU reduce (PCIe cheaper than row all_gather for long S).
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
        W_in_4d  = self.hf_moe.input_linear.weight.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)
        W_out_4d = self.hf_moe.output_linear.weight.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)

        self.gate_up_proj = ttnn.from_torch(
            W_in_4d,
            device=self.device, dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        self.down_proj = ttnn.from_torch(
            W_out_4d,
            device=self.device, dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )

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
                # Decode with fabric: fully on-device routing avoids PCIe round-trip.
                routing_tt = self.compute_routing_device(hidden_states_tt)
            else:
                # Prefill or no-fabric: CPU routing (PCIe cost amortized, or fabric unavailable).
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
            gathered_tt = ttnn.all_gather(
                local_sum_tt, dim=1, cluster_axis=1, memory_config=MC,
            )
            local_sum_tt.deallocate(True)
            result_tt = ttnn.sum(gathered_tt, dim=1, keepdim=True, memory_config=MC)
            gathered_tt.deallocate(True)
        elif not self._use_col_parallel and self.effective_cols > 1:
            if self._use_row_reduce and S == 1:
                row_gathered = ttnn.all_gather(
                    local_sum_tt, dim=1, cluster_axis=0, memory_config=MC,
                )
                local_sum_tt.deallocate(True)
                col_sum_tt = ttnn.sum(row_gathered, dim=1, keepdim=True, memory_config=MC)
                row_gathered.deallocate(True)
                col_gathered = ttnn.all_gather(
                    col_sum_tt, dim=1, cluster_axis=1, memory_config=MC,
                )
                col_sum_tt.deallocate(True)
                result_tt = ttnn.sum(col_gathered, dim=1, keepdim=True, memory_config=MC)
                col_gathered.deallocate(True)
            else:
                # Multi-row mesh prefill or no-fabric: CPU reduce.
                all_sums = ttnn.to_torch(
                    local_sum_tt,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                )
                local_sum_tt.deallocate(True)
                global_sum = all_sums.sum(dim=0, keepdim=True)
                result_tt = ttnn.from_torch(
                    global_sum,
                    device=self.device,
                    dtype=self.act_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None,
                )
        else:
            result_tt = local_sum_tt

        return result_tt
