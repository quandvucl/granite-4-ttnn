"""
TTNN MoE for GraniteMoeHybrid — expert-parallel sharding across mesh columns.

Each mesh column owns num_experts/col_devices experts.
Rows replicate the same experts (no cross-row communication needed).
Router runs on CPU (one linear layer, negligible cost).
EP all_reduce: local expert sum → all_gather(cluster_axis=1) → sum on device.
No PCIe round-trips in the expert-compute hot path.
"""

import torch
import ttnn
from utils import to_torch_tensor
from utils.device import _is_mesh_device


class GraniteTTMoE:
    def __init__(self, hf_moe, device, weight_dtype=ttnn.bfloat8_b, act_dtype=ttnn.bfloat16):
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

        # EP sharding strategy depends on mesh shape:
        #
        # Single-row mesh (e.g. tiny, MeshShape(1,4)):
        #   Shard across all 4 devices; use all_gather(cluster_axis=1) for on-device reduce.
        #   No row replication overhead.
        #
        # Multi-row mesh (e.g. small, MeshShape(2,4)):
        #   Replicated row sharding doubles per-device weight DRAM → OOM for large models.
        #   Fall back to flat num_devices-way EP + CPU reduce (original approach).
        #   The CPU reduce is a single small download per layer (already optimised with
        #   local sum first), so the cost is acceptable.
        if self.is_mesh and self.num_devices > 1:
            mesh_shape = device.shape
            self._mesh_rows = mesh_shape[0]
            self._mesh_cols = mesh_shape[1]
        else:
            self._mesh_rows = 1
            self._mesh_cols = 1

        self._use_col_parallel = (self._mesh_rows == 1) and self.num_devices > 1

        if self._use_col_parallel:
            # Single-row: shard across all columns, on-device all_gather reduce.
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
            # Multi-row: flat EP across all devices, CPU reduce.
            effective_devices = self.num_devices
            while effective_devices > 1 and self.num_experts % effective_devices != 0:
                effective_devices -= 1
            self.effective_cols = effective_devices  # reuse field name for experts_per_device calc
            self.experts_per_device = self.num_experts // effective_devices
            self._ep_mapper = ttnn.ShardTensorToMesh(
                self.device, dim=1
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

    def forward(self, hidden_states_tt):
        """
        hidden_states_tt: TTNN [1, 1, S, H] replicated.
        Caller must NOT deallocate — also needed for shared MLP.
        Returns: TTNN [1, 1, S, H] replicated.
        """
        I       = self.intermediate_size
        E       = self.num_experts
        E_local = self.experts_per_device
        S       = hidden_states_tt.shape[2]

        # ── 1. ROUTER (CPU) ───────────────────────────────────────────────────
        logits_tt = ttnn.linear(
            hidden_states_tt, self.router_weight_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits_cpu = to_torch_tensor(logits_tt, target_shape=None)
        logits_tt.deallocate(True)
        logits_SE = logits_cpu.reshape(-1, E)[:S]                          # [S, E]

        top_k_logits, top_k_indices = logits_SE.topk(self.top_k, dim=1)   # [S, top_k]
        top_k_weights = torch.softmax(top_k_logits.float(), dim=1).to(torch.bfloat16)

        routing_SE = torch.zeros(S, E, dtype=torch.bfloat16)
        routing_SE.scatter_add_(1, top_k_indices, top_k_weights)
        routing_4d = routing_SE.T.unsqueeze(-1).unsqueeze(0)              # [1, E, S, 1]

        routing_tt = ttnn.from_torch(
            routing_4d,
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        # per device: routing_tt [1, E_local, S, 1]

        # ── 2. GATE+UP ────────────────────────────────────────────────────────
        hidden_exp = ttnn.repeat(
            hidden_states_tt, ttnn.Shape([1, E_local, 1, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up_out = ttnn.matmul(hidden_exp, self.gate_up_proj,
                                  dtype=self.act_dtype,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_exp.deallocate(True)
        # [1, E_local, S, 2I]

        # ── 3. ACTIVATION ────────────────────────────────────────────────────
        gate_tt = gate_up_out[:, :, :, :I]
        up_tt   = gate_up_out[:, :, :, I:]
        gate_up_out.deallocate(True)

        activated_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_tt.deallocate(True)
        up_tt.deallocate(True)

        # ── 4. DOWN PROJECTION ───────────────────────────────────────────────
        expert_out_tt = ttnn.matmul(activated_tt, self.down_proj,
                                    dtype=self.act_dtype,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated_tt.deallocate(True)

        # ── 5. WEIGHT by routing ─────────────────────────────────────────────
        weighted_tt = ttnn.mul(expert_out_tt, routing_tt,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_out_tt.deallocate(True)
        routing_tt.deallocate(True)

        # ── 6. Local sum on device, then all_gather + sum across columns ──────
        # Sum [1, E_local, S, H] → [1, 1, S, H] per device.
        local_sum_tt = ttnn.sum(
            weighted_tt, dim=1, keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, S, H]
        weighted_tt.deallocate(True)

        if self._use_col_parallel and self.effective_cols > 1:
            # Single-row mesh: all_gather on fabric across columns, sum on device.
            gathered_tt = ttnn.all_gather(
                local_sum_tt, dim=1, cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            local_sum_tt.deallocate(True)
            result_tt = ttnn.sum(
                gathered_tt, dim=1, keepdim=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gathered_tt.deallocate(True)
        elif not self._use_col_parallel and self.effective_cols > 1:
            # Multi-row mesh: CPU reduce (avoids OOM from row-replicated weights).
            all_sums = ttnn.to_torch(
                local_sum_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
            )  # [N, 1, S, H]
            local_sum_tt.deallocate(True)
            global_sum = all_sums.sum(dim=0, keepdim=True)  # [1, 1, S, H]
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
