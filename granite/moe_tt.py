"""
TTNN MoE for GraniteMoeHybrid — expert-parallel sharding across 8 devices.

Each device owns 72/8 = 9 experts.  Expert weights are preloaded to DRAM at init.
Router runs on CPU (one linear layer, negligible cost).
Expert compute: dense batched matmul — no Python loop.
EP all_reduce at the end.

HF forward to match:
    expert(x) = output_linear(silu(gate_half) * up_half)   # no biases
    y = sum_k( softmax(topk(router(x)))[k] * expert_k(x) )
"""

import torch
import ttnn
from utils import to_torch_tensor
from utils.device import _is_mesh_device


class GraniteTTMoE:
    """
    TTNN Mixture-of-Experts for Granite.

    Single device: not supported — 27 GB of expert weights don't fit.
                   Caller should fall back to hf_layer.block_sparse_moe().
    8 devices:     9 experts/device, ~3.4 GB/device across 40 layers.
    """

    def __init__(self, hf_moe, device, weight_dtype=ttnn.bfloat8_b, act_dtype=ttnn.bfloat16):
        self.hf_moe = hf_moe
        self.device = device
        self.dtype = weight_dtype       # expert weight storage dtype
        self.act_dtype = act_dtype      # activation compute dtype

        self.is_mesh = _is_mesh_device(device)
        self.num_devices = device.get_num_devices() if self.is_mesh else 1

        self.num_experts = hf_moe.input_linear.num_experts
        self.hidden_size = hf_moe.input_size
        self.intermediate_size = hf_moe.hidden_size
        self.top_k = hf_moe.router.top_k

        # If num_experts is not divisible by num_devices, fall back to the largest
        # divisor to avoid wasted padding complexity. Each device still gets a whole
        # number of experts; remaining experts are handled by the first devices.
        # Simple case: find largest N' <= num_devices that divides num_experts.
        effective_devices = self.num_devices
        while effective_devices > 1 and self.num_experts % effective_devices != 0:
            effective_devices -= 1
        self.effective_devices = effective_devices
        self.experts_per_device = self.num_experts // effective_devices

        # ShardTensorToMesh(dim=1): shard expert dim across effective_devices.
        # On single device or when effective_devices==1, no shard mapper needed.
        self._ep_mapper = ttnn.ShardTensorToMesh(
            self.device, dim=1
        ) if self.effective_devices > 1 else None

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
        Args:
            hidden_states_tt: TTNN tensor [1, 1, S, H], replicated on all devices.
                              Caller must NOT deallocate this tensor — it is used
                              but not consumed here (also needed for shared MLP).
        Returns:
            moe_out_tt: TTNN tensor [1, 1, S, H] on device (replicated after all_reduce).
        """
        I  = self.intermediate_size
        E  = self.num_experts
        E_local = self.experts_per_device
        S  = hidden_states_tt.shape[2]   # real seq length (dim 2 of [1,1,S,H])

        # ------------------------------------------------------------------
        # 1. ROUTER: linear on device, topk+softmax on CPU (P5)
        # ------------------------------------------------------------------
        # Router linear: [1, 1, S, H] @ [H, E] → [1, 1, S, E]
        logits_tt = ttnn.linear(
            hidden_states_tt, self.router_weight_tt,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Download [1, 1, S, E] logits — small (S*E*2 bytes) — for topk
        logits_cpu = to_torch_tensor(logits_tt, target_shape=None)
        logits_tt.deallocate(True)
        logits_SE = logits_cpu.reshape(-1, E)[:S]                          # [S, E]

        top_k_logits, top_k_indices = logits_SE.topk(self.top_k, dim=1)   # [S, top_k]
        top_k_weights = torch.softmax(top_k_logits.float(), dim=1).to(torch.bfloat16)

        # Dense routing matrix [S, E] via scatter, then [1, E, S, 1] for device
        routing_SE = torch.zeros(S, E, dtype=torch.bfloat16)
        routing_SE.scatter_add_(1, top_k_indices, top_k_weights)
        routing_4d = routing_SE.T.unsqueeze(-1).unsqueeze(0)              # [1, E, S, 1]

        # Upload routing (tiny: [1, E, S, 1] = E × S × 2 bytes, EP-sharded)
        routing_tt = ttnn.from_torch(
            routing_4d,
            device=self.device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        # Each device now has routing_tt: [1, E_local, S, 1]

        # ------------------------------------------------------------------
        # 2. EXPAND hidden across expert dim: [1, 1, S, H] → [1, E_local, S, H]
        # ------------------------------------------------------------------
        # ttnn.repeat replicates the tensor E_local times along dim 1
        hidden_exp = ttnn.repeat(hidden_states_tt, ttnn.Shape([1, E_local, 1, 1]))
        # [1, E_local, S, H] on each device

        # ------------------------------------------------------------------
        # 3. GATE+UP PROJECTION: [1, E_local, S, H] @ [1, E_local, H, I*2]
        #                      → [1, E_local, S, I*2]
        # ------------------------------------------------------------------
        gate_up_out = ttnn.matmul(hidden_exp, self.gate_up_proj,
                                  dtype=self.act_dtype,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_exp.deallocate(True)

        # ------------------------------------------------------------------
        # 4. ACTIVATION: silu(gate_half) * up_half
        # ------------------------------------------------------------------
        gate_tt = gate_up_out[:, :, :, :I]
        up_tt   = gate_up_out[:, :, :, I:]
        gate_up_out.deallocate(True)

        activated_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_tt.deallocate(True)
        up_tt.deallocate(True)

        # ------------------------------------------------------------------
        # 5. DOWN PROJECTION: [1, E_local, S, I] @ [1, E_local, I, H]
        #                   → [1, E_local, S, H]
        # ------------------------------------------------------------------
        expert_out_tt = ttnn.matmul(activated_tt, self.down_proj,
                                    dtype=self.act_dtype,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated_tt.deallocate(True)

        # ------------------------------------------------------------------
        # 6. WEIGHT by routing: [1, E_local, S, H] * [1, E_local, S, 1]
        #                     → [1, E_local, S, H]
        # ------------------------------------------------------------------
        weighted_tt = ttnn.mul(expert_out_tt, routing_tt,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_out_tt.deallocate(True)
        routing_tt.deallocate(True)

        # ------------------------------------------------------------------
        # 7 + 8. SUM over experts + EP reduce.
        # Sum [1,E_local,S,H] → [1,1,S,H] on device first — E_local× less data
        # to download (9× for small, 16× for tiny) vs gathering before summing.
        # ------------------------------------------------------------------
        local_sum_tt = ttnn.sum(
            weighted_tt, dim=1, keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, S, H] per device
        weighted_tt.deallocate(True)

        if self.effective_devices > 1:
            # Download [N, 1, S, H] — E_local× smaller than [N, E_local, S, H]
            all_sums = ttnn.to_torch(
                local_sum_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
            )  # [N, 1, S, H]
            local_sum_tt.deallocate(True)
            global_sum = all_sums.sum(dim=0, keepdim=True)  # [1, 1, S, H]
        else:
            global_sum = ttnn.to_torch(local_sum_tt)  # [1, 1, S, H]
            local_sum_tt.deallocate(True)

        result_tt = ttnn.from_torch(
            global_sum,
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None,
        )
        return result_tt
