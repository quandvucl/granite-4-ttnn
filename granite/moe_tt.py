"""
TTNN MoE for GraniteMoeHybrid — expert-parallel sharding across mesh devices.

1×4 tiny (4 devices): ShardTensor2dMesh(dims=(None,1)), 9 experts/device.
  Reduce: all_gather_async(cluster_axis=1, Linear) + sum. Trace-safe.
8×1 small (8 devices): ShardTensorToMesh(dim=1), 9 experts/device.
  Reduce: all_gather_async(cluster_axis=0, Linear) + sum. Trace-safe.
  MeshShape(8,1): mesh_shape[1]=1 → is_true_2d_mesh()=false → composite_all_gather bypassed.

Decode router (S=1): on-device topk+softmax+embedding, no PCIe. Both tiny and small.

  Tiny path: embed into full [E,E] eye (replicated) → [top_k, E] one-hots → sum →
    [1,1,1,E] → linear with ShardTensor2dMesh(dims=(None,1)) proj [E,E_local].
    ShardTensor2dMesh gives TTNN col-parallel context → correct independent output.

  Small path: embed directly into sharded [E, E_local] eye per device.
    ShardTensorToMesh(dim=1) on eye[E,E] → device d holds eye[:, d*E_local:(d+1)*E_local].
    Embedding is a row-lookup (not a matmul) — no collective op triggered.
    Each device independently extracts its own E_local routing weights.

  linear(replicated, flat-sharded) is avoided for small: on a 2D mesh TTNN applies
  TP all-reduce semantics to matmul, collapsing all-devices to device-0 result.

Prefill router (S>1): CPU (PCIe cost amortized over S tokens).
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

        # EP sharding:
        #   Tiny (1×4): ShardTensor2dMesh(dims=(None,1)), 9 experts/device, all_gather(axis=1).
        #   Small (8×1): ShardTensorToMesh(dim=1), 9 experts/device, all_gather(axis=0).
        #   Both trace-safe: single-axis mesh → is_true_2d_mesh()=false → no composite path.
        self._use_col_parallel = (self._mesh_rows == 1) and self.num_devices > 1 and use_all_gather
        self._is_multirow = (self._mesh_rows > 1) and self.num_devices > 1 and use_all_gather

        if self._use_col_parallel:
            effective_cols = self._mesh_cols
            while effective_cols > 1 and self.num_experts % effective_cols != 0:
                effective_cols -= 1
            self.effective_cols = effective_cols
            self.effective_devs = effective_cols
            self.experts_per_device = self.num_experts // effective_cols
            self._ep_mapper = ttnn.ShardTensor2dMesh(
                self.device, dims=(None, 1),
                mesh_shape=ttnn.MeshShape(1, effective_cols),
            ) if effective_cols > 1 else (
                ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None
            )
        elif self._is_multirow:
            effective_devices = self.num_devices
            while effective_devices > 1 and self.num_experts % effective_devices != 0:
                effective_devices -= 1
            self.effective_cols = effective_devices
            self.effective_devs = effective_devices
            self.experts_per_device = self.num_experts // effective_devices
            self._ep_mapper = ttnn.ShardTensorToMesh(
                self.device, dim=1,
            ) if effective_devices > 1 else (
                ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None
            )
        else:
            self.effective_cols = 1
            self.effective_devs = 1
            self.experts_per_device = self.num_experts
            self._ep_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

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
        E       = self.num_experts
        E_local = self.experts_per_device
        eye     = torch.eye(E, dtype=torch.bfloat16)

        if self._is_multirow:
            # Small (2×4 flat EP): embedding table is sharded per device.
            # ShardTensorToMesh(dim=1) on eye[E,E] → device d gets eye[:, d*E_local:(d+1)*E_local]
            # shape [E, E_local]. Embedding is a row-lookup with no collective — each device
            # independently returns its local routing slice. No linear projection needed.
            self._routing_eye_tt = ttnn.from_torch(
                eye, device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._ep_mapper,
            )
            self._routing_proj_tt = None  # not used for multirow
        else:
            # Tiny (1×4 col-parallel) and single-device: replicated full [E,E] eye.
            self._routing_eye_tt = ttnn.from_torch(
                eye, device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )
            # [E, E_local] projection for col-parallel: ShardTensor2dMesh gives TTNN
            # the col-parallel context so linear(replicated, col-sharded) is correct.
            self._routing_proj_tt = ttnn.from_torch(
                eye, device=self.device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._ep_mapper,
            )

    def _load_weights(self):
        W_in_4d  = self.hf_moe.input_linear.weight.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)
        W_out_4d = self.hf_moe.output_linear.weight.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)

        # Peak DRAM budget: upload as bf16 then typecast only when per-device bf16 fits (~<100 MB).
        # For large models (small, 9 experts/device), bf16 intermediate exceeds the largest free
        # contiguous DRAM block → OOM. Upload directly as target dtype instead (half the peak).
        _per_dev_bytes_bf16 = W_in_4d.numel() * 2 // max(self.effective_devs, 1)
        _use_direct = self.dtype != ttnn.bfloat16 and _per_dev_bytes_bf16 > 100 * 1024 * 1024

        print(f"  [moe_tt] uploading gate_up shape={list(W_in_4d.shape)} dtype={self.dtype} direct={_use_direct}", flush=True)
        self.gate_up_proj = ttnn.from_torch(
            W_in_4d,
            device=self.device,
            dtype=self.dtype if _use_direct else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        if not _use_direct and self.dtype != ttnn.bfloat16:
            tmp = self.gate_up_proj
            self.gate_up_proj = ttnn.typecast(tmp, self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tmp.deallocate(True)

        print(f"  [moe_tt] uploading down_proj shape={list(W_out_4d.shape)}", flush=True)
        self.down_proj = ttnn.from_torch(
            W_out_4d,
            device=self.device,
            dtype=self.dtype if _use_direct else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._ep_mapper,
        )
        if not _use_direct and self.dtype != ttnn.bfloat16:
            tmp = self.down_proj
            self.down_proj = ttnn.typecast(tmp, self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tmp.deallocate(True)
        print(f"  [moe_tt] weights loaded", flush=True)

    def compute_routing_cpu(self, hidden_states_tt):
        """Compute routing weights on CPU. Returns routing_tt [1, E_local, S, 1] on device."""
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
        Uses topk + softmax + embedding to avoid PCIe round-trip.

        Tiny (col-parallel): embedding table is replicated [E,E]; embed → [top_k,E] one-hots
          → sum → [1,1,1,E] → linear with ShardTensor2dMesh proj → [1,1,1,E_local] per device.

        Small (multirow, flat EP): embedding table is sharded [E,E_local] per device.
          Embedding row-lookup returns [top_k, E_local] directly for each device — indices are
          replicated (from replicated logits+topk), table differs per device, no collective.
          Sum top_k rows → [1,1,1,E_local] per device. No linear projection.
        """
        E_local = self.experts_per_device
        L1      = ttnn.L1_MEMORY_CONFIG

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

        # embed: [1, top_k] indices → [1, top_k, E_local] (multirow) or [1, top_k, E] (col-parallel)
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

        if self._is_multirow:
            # scaled_tt: [1, top_k, E_local] — sum over top_k dimension directly
            routing_local = ttnn.sum(scaled_tt, dim=1, keepdim=False, memory_config=L1)
            scaled_tt.deallocate(True)
            routing_tt = ttnn.reshape(routing_local, [1, E_local, 1, 1])
            routing_local.deallocate(True)
        else:
            # col-parallel: scaled_tt [1, top_k, E] → sum → [1,1,1,E] → linear proj → [1,1,1,E_local]
            E = self.num_experts
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

    def _all_gather_col(self, x, dim, memory_config):
        """All-gather along column axis (cluster_axis=1). Trace-safe on single-row mesh."""
        if self.tt_ccl is not None:
            return ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=dim,
                cluster_axis=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
                num_links=self.tt_ccl.get_num_links(1),
                memory_config=memory_config,
                topology=self._topology,
                chunks_per_sync=1,
                num_workers_per_link=1,
                num_buffers_per_channel=2,
            )
        return ttnn.all_gather(x, dim=dim, cluster_axis=1, memory_config=memory_config)

    def _all_gather_row(self, x, dim, memory_config):
        """All-gather along row axis (cluster_axis=0). Trace-safe on single-row mesh."""
        if self.tt_ccl is not None:
            return ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=dim,
                cluster_axis=0,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(0),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(0),
                num_links=self.tt_ccl.get_num_links(0),
                memory_config=memory_config,
                topology=self._topology,
                chunks_per_sync=1,
                num_workers_per_link=1,
                num_buffers_per_channel=2,
            )
        return ttnn.all_gather(x, dim=dim, cluster_axis=0, memory_config=memory_config)

    def forward(self, hidden_states_tt, routing_tt=None):
        """
        hidden_states_tt: TTNN [1, 1, S, H] replicated.
        routing_tt: optional precomputed [1, E_local, S, 1] — skips router if provided.
        Caller must NOT deallocate hidden_states_tt — also needed for shared MLP.
        Returns: TTNN [1, 1, S, H] replicated.
        """
        I       = self.intermediate_size
        E_local = self.experts_per_device
        S       = hidden_states_tt.shape[2]
        MC      = ttnn.DRAM_MEMORY_CONFIG

        # ── 1. ROUTER ─────────────────────────────────────────────────────────
        if routing_tt is None:
            routing_tt = getattr(self, '_precomputed_routing', None)
        self._precomputed_routing = None
        if routing_tt is None:
            if S == 1 and (self._use_col_parallel or self._is_multirow):
                routing_tt = self.compute_routing_device(hidden_states_tt)
            else:
                routing_tt = self.compute_routing_cpu(hidden_states_tt)

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

        # ── 6. Local sum then reduce ──────────────────────────────────────────
        local_sum_tt = ttnn.sum(weighted_tt, dim=1, keepdim=True, memory_config=MC)
        weighted_tt.deallocate(True)

        if self._use_col_parallel and self.effective_cols > 1:
            # Single-row (tiny): async col gather + sum.
            gathered_tt = self._all_gather_col(local_sum_tt, dim=1, memory_config=MC)
            local_sum_tt.deallocate(True)
            result_tt = ttnn.sum(gathered_tt, dim=1, keepdim=True, memory_config=MC)
            gathered_tt.deallocate(True)
        elif self._is_multirow and self.effective_devs > 1:
            if self._mesh_cols == 1:
                # 8×1 mesh: gather along rows, trace-safe (mesh_shape[1]=1 → not is_true_2d_mesh).
                gathered_tt = self._all_gather_row(local_sum_tt, dim=1, memory_config=MC)
                local_sum_tt.deallocate(True)
                result_tt = ttnn.sum(gathered_tt, dim=1, keepdim=True, memory_config=MC)
                gathered_tt.deallocate(True)
            else:
                # 2×4 mesh: all_reduce, non-trace (FABRIC_2D + is_true_2d_mesh).
                after_cols = ttnn.all_reduce(local_sum_tt, cluster_axis=1, memory_config=MC)
                local_sum_tt.deallocate(True)
                result_tt = ttnn.all_reduce(after_cols, cluster_axis=0, memory_config=MC)
                after_cols.deallocate(True)
        else:
            result_tt = local_sum_tt

        return result_tt
