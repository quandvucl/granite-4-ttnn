"""TT-accelerated Granite hybrid model for causal language modeling."""

import sys
from pathlib import Path
from typing import Optional

import torch
import ttnn
from transformers import AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent))
from granite.config import TTGraniteConfig
from granite.decoder_layer import TTGraniteDecoderLayer
from utils import to_torch_tensor
from granite import MambaCacheManager

# Use production-ready components from models/common/
from models.common.modules.lm_head.lm_head_1d import LMHead1D
from models.common.modules.embedding.embedding_1d import Embedding1D
from models.common.rmsnorm import RMSNorm
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.tt_ccl import get_tt_ccl
from models.tt_transformers.tt.common import Mode


class ReplicatedMLP:
    """Shared MLP with weights replicated across all devices — no all_reduce."""

    def __init__(self, w1, w2, w3, dtype):
        self.w1 = w1  # gate
        self.w3 = w3  # up
        self.w2 = w2  # down
        self.dtype = dtype

    def forward(self, x: "ttnn.Tensor", mode=None) -> "ttnn.Tensor":
        MC = ttnn.L1_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w1, dtype=self.dtype, memory_config=MC)
        up   = ttnn.linear(x, self.w3, dtype=self.dtype, memory_config=MC)
        gate = ttnn.silu(gate, memory_config=MC)
        mid  = ttnn.mul(gate, up, memory_config=MC)
        gate.deallocate(True)
        up.deallocate(True)
        out  = ttnn.linear(mid, self.w2, dtype=self.dtype, memory_config=MC)
        mid.deallocate(True)
        return out


class ColumnParallelMLP:
    """Column-parallel gate+up (shard F across mesh cols), row-parallel down.

    Each device holds gate/up: [H, F/num_cols] and down: [F/num_cols, H].
    After the down projection each device has a partial sum [1,1,S,H].
    all_gather(col-axis) + sum completes the all-reduce.
    Rows receive replicated shards, so no row all_gather is needed.
    """

    def __init__(self, w1, w2, w3, num_cols, dtype, tt_ccl=None, cluster_axis=1):
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2
        self.num_cols = num_cols
        self.dtype = dtype
        self.tt_ccl = tt_ccl
        self._cluster_axis = cluster_axis
        self._topology = ttnn.Topology.Linear

    def forward(self, x, mode=None):
        MC = ttnn.L1_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w1, dtype=self.dtype, memory_config=MC)
        up   = ttnn.linear(x, self.w3, dtype=self.dtype, memory_config=MC)
        gate = ttnn.silu(gate, memory_config=MC)
        mid  = ttnn.mul(gate, up, memory_config=MC)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(mid, self.w2, dtype=self.dtype, memory_config=MC)
        mid.deallocate(True)
        if self.num_cols > 1:
            ax = self._cluster_axis
            if self.tt_ccl is not None:
                gathered = ttnn.experimental.all_gather_async(
                    out,
                    persistent_output_buffer=None,
                    dim=1, cluster_axis=ax,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(ax),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(ax),
                    num_links=self.tt_ccl.get_num_links(ax),
                    memory_config=MC,
                    topology=self._topology,
                    chunks_per_sync=1,
                    num_workers_per_link=1,
                    num_buffers_per_channel=2,
                )
            else:
                gathered = ttnn.all_gather(out, dim=1, cluster_axis=ax, memory_config=MC)
            out.deallocate(True)
            out = ttnn.sum(gathered, dim=1, keepdim=True, memory_config=MC)
            gathered.deallocate(True)
        return out


class TTGraniteMoeHybridForCausalLM:
    """
    Tenstorrent-accelerated Granite MoE Hybrid model for causal language modeling.

    Implements hybrid execution:
    - Attention layers (5, 15, 25, 35): Run on TT hardware
    - Mamba layers (all others): Run on CPU
    - Shared MLP: Run on TT hardware
    - Embeddings: Run on TT hardware
    - LM head: Run on TT hardware
    """

    def __init__(
        self,
        device,
        hf_model,
        config: TTGraniteConfig,
        verbose: bool = True,
        use_tt_attention: bool = False,
        use_tt_mamba: bool = False,
        use_tt_moe: bool = True,
        mamba_chunk_size: int = None,
        moe_weight_dtype=None,
        mamba_weight_dtype=None,
        moe_use_all_gather=True,
    ):
        self.device = device
        self.mamba_chunk_size = mamba_chunk_size
        self.moe_use_all_gather = moe_use_all_gather
        self.hf_model = hf_model
        self.hf_config = hf_model.config
        self.config = config
        self.verbose = verbose
        self.use_tt_attention = use_tt_attention
        self.use_tt_mamba = use_tt_mamba
        self.use_tt_moe = use_tt_moe
        self.moe_weight_dtype = moe_weight_dtype if moe_weight_dtype is not None else ttnn.bfloat8_b
        self.mamba_weight_dtype = mamba_weight_dtype  # None means use activation dtype (bfloat16)

        # Extract HF components
        self.embed_tokens = hf_model.model.embed_tokens

        if verbose:
            print("  Initializing LM head...")

        lm_head_weight = hf_model.lm_head.weight.T.contiguous()  # [hidden, vocab]
        vocab_size = lm_head_weight.shape[1]
        self.logits_scaling = config.logits_scaling

        padded_vocab_size = ((vocab_size + 31) // 32) * 32
        if vocab_size < padded_vocab_size:
            lm_head_weight = torch.cat([
                lm_head_weight,
                torch.zeros(lm_head_weight.shape[0], padded_vocab_size - vocab_size,
                            dtype=lm_head_weight.dtype)
            ], dim=-1)
        lm_head_lazy = LazyWeight(
            source=lm_head_weight, device=device, dtype=config.get_ttnn_dtype(),
        )
        self.lm_head = LMHead1D([lm_head_lazy])

        if verbose:
            print("\n=== Initializing TTGraniteMoeHybridForCausalLM ===")

        is_mesh = hasattr(device, "get_num_devices")
        self.is_mesh = is_mesh

        if verbose:
            print("  Initializing embedding (Embedding1D)...")

        embedding_lazy = LazyWeight(
            source=hf_model.model.embed_tokens.weight,
            device=device,
            dtype=config.get_ttnn_dtype(),
        )
        self.embedding = Embedding1D(embedding_lazy, embed_scale=config.embedding_multiplier)
        self._use_tt_embeddings = is_mesh

        if is_mesh:
            if verbose:
                print(f"  Mesh device detected ({device.get_num_devices()} devices) - tensor parallelism enabled")
            self.tt_ccl = get_tt_ccl(device) if device.get_num_devices() > 1 else None
            _ms = device.shape
            self._is_multirow_mesh = _ms[0] > 1 and _ms[1] > 1
        else:
            self.tt_ccl = None
            self._is_multirow_mesh = False

        # Initialize production-ready RMSNorm
        if verbose:
            print("  Initializing final norm (RMSNorm)...")

        # Create a minimal state dict for RMSNorm
        state_dict = {"norm.weight": hf_model.model.norm.weight}

        self.norm = RMSNorm(
            device=device,
            dim=config.hidden_size,
            state_dict=state_dict,
            weight_key="norm",
            weight_dtype=config.get_ttnn_dtype(),
            eps=config.rms_norm_eps,
            fp32_dest_acc_en=True,
        )

        # Initialize decoder layers
        self.layers = []
        for layer_idx, hf_layer in enumerate(hf_model.model.layers):
            is_attention = layer_idx in config.attention_layer_indices

            # Initialize production-ready MLP1D from models/common/
            hf_mlp = hf_layer.shared_mlp

            # Extract MLP weights
            input_linear_weight_hf = hf_mlp.input_linear.weight  # [intermediate*2, hidden]
            output_linear_weight_hf = hf_mlp.output_linear.weight  # [hidden, intermediate]

            # Transpose and split: [intermediate*2, hidden] -> [hidden, intermediate*2]
            input_linear_t = input_linear_weight_hf.T.contiguous()
            actual_intermediate_size = input_linear_t.shape[1] // 2

            # Split into gate and up weights
            gate_weight = input_linear_t[:, :actual_intermediate_size].contiguous()
            up_weight = input_linear_t[:, actual_intermediate_size:].contiguous()
            down_weight = output_linear_weight_hf.T.contiguous()

            # Enable column-parallel MLP for meshes with >= 4 devices.
            # Small on 4 dev (H=4096, F=1536): 40×37.7 MB replicated = 1.5 GB — needs TP.
            # Tiny (H=1536, F=768, 4 devices): TP also works (F/4=384, divisible).
            _num_dev = device.get_num_devices() if is_mesh else 1
            use_tensor_parallel = is_mesh and _num_dev > 4
            dtype = config.get_ttnn_dtype()

            if use_tensor_parallel:
                mesh_shape = device.shape
                num_cols = mesh_shape[1]
                num_rows = mesh_shape[0]
                F = gate_weight.shape[1]
                # Only shard shared MLP across cols (N×M mesh, M>1).
                # 8×1: num_cols=1 → replicated (no TP for shared MLP on row-only mesh).
                # Replicated is faster: matmul at F=192/device underutilizes Tensix cores,
                # and 40 all_gather calls/step exceed savings vs replicated at F=1536.
                if num_cols == 1:
                    use_tensor_parallel = False
                else:
                    num_tp, cluster_axis = num_cols, 1
                    fwd_mapper = ttnn.ShardTensor2dMesh(device, dims=(None, 1), mesh_shape=mesh_shape)
                    rev_mapper = ttnn.ShardTensor2dMesh(device, dims=(None, 0), mesh_shape=mesh_shape)
                if use_tensor_parallel and F % num_tp == 0:
                    def _upload_fwd(t):
                        return ttnn.from_torch(
                            t, device=device, dtype=dtype,
                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=fwd_mapper,
                        )

                    def _upload_rev(t):
                        return ttnn.from_torch(
                            t, device=device, dtype=dtype,
                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=rev_mapper,
                        )

                    layer_mlp = ColumnParallelMLP(
                        w1=_upload_fwd(gate_weight),
                        w2=_upload_rev(down_weight),
                        w3=_upload_fwd(up_weight),
                        num_cols=num_tp,
                        dtype=dtype,
                        tt_ccl=self.tt_ccl,
                        cluster_axis=cluster_axis,
                    )
                else:
                    use_tensor_parallel = False  # fallthrough to replicated

            if not use_tensor_parallel:
                def _upload_replicated(t):
                    return ttnn.from_torch(
                        t, device=device, dtype=dtype,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
                    )

                layer_mlp = ReplicatedMLP(
                    w1=_upload_replicated(gate_weight),
                    w2=_upload_replicated(down_weight),
                    w3=_upload_replicated(up_weight),
                    dtype=dtype,
                )

            layer = TTGraniteDecoderLayer(
                device=device,
                layer_idx=layer_idx,
                config=config,
                hf_layer=hf_layer,
                shared_mlp=layer_mlp,
                is_attention_layer=is_attention,
                hf_config=hf_model.config,
                dtype=config.get_ttnn_dtype(),
                tensor_parallel=use_tensor_parallel,
                tt_ccl=self.tt_ccl,
                use_tt_attention=self.use_tt_attention,
                use_tt_mamba=self.use_tt_mamba,
                use_tt_moe=self.use_tt_moe,
                mamba_chunk_size=self.mamba_chunk_size,
                moe_weight_dtype=self.moe_weight_dtype,
                mamba_weight_dtype=self.mamba_weight_dtype,
                moe_use_all_gather=self.moe_use_all_gather,
            )

            self.layers.append(layer)

            layer_type = "Attention" if is_attention else "Mamba"
            print(f"  layer {layer_idx:2d} ({layer_type}) done", flush=True)

        # Initialize Mamba cache manager (Attention1D manages its own KV cache)
        self.cache_manager = MambaCacheManager(
            num_layers=config.num_hidden_layers,
            batch_size=config.batch_size,
            attention_layer_indices=config.attention_layer_indices,
        )

        # Decode trace state (captured explicitly via capture_decode_trace() after warmup)
        self._decode_trace_id = None
        self._decode_trace_input = None   # persistent DRAM: updated with embeddings before replay
        self._decode_trace_output = None  # tensor in trace region: logits output
        self._trace_zeros = None          # persistent DRAM zeros: addend for copy-via-add inside trace
        self._in_trace = False            # True only during begin/end_trace_capture and execute_trace
        if self.tt_ccl is not None: self.tt_ccl._in_trace = False



        # Set sub-device stall group at init so all ops use the trace-compatible dispatch path.
        _trace_supported_init = (self.tt_ccl is not None and self.is_mesh)
        if _trace_supported_init:
            self.device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

        # Release HF model CPU tensors — all weights are now on TT devices.
        # Keep only hf_config (already stored separately) for cache init.
        del self.hf_model
        self.hf_model = None

        if verbose:
            print(f"\n✓ Model initialized with {len(self.layers)} layers")
            print(f"  - {len(config.attention_layer_indices)} attention layers")
            print(f"  - {config.num_hidden_layers - len(config.attention_layer_indices)} mamba layers")
            if is_mesh and device.get_num_devices() > 1:
                print(f"  - Mesh device: {device.get_num_devices()} devices")
                if use_tensor_parallel:
                    print(f"  - Tensor parallelism: ENABLED")
                else:
                    print(f"  - Tensor parallelism: DISABLED (replicated weights)")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        _state_only: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.  hidden_states stays on TTNN for all 40 layers.

        Boundary conversions:
          - CPU → TTNN: once after embedding
          - TTNN → CPU: once before final norm+lm_head (or never if lm_head is on TTNN)

        For prefill with seq_len > mamba_chunk_size, the sequence is split into
        segments of mamba_chunk_size tokens processed sequentially through all layers.
        This reduces peak DRAM allocation, improving L1 reuse on long sequences.
        """
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, f"Only batch_size=1 supported, got {batch_size}"

        # Chunked prefill: split long sequences into smaller segments.
        chunk_sz = self.mamba_chunk_size
        if chunk_sz is not None and seq_len > chunk_sz and seq_len > 1:
            return self._forward_chunked(input_ids, position_ids, attention_mask, use_cache, chunk_sz)

        if position_ids is None:
            position_ids = torch.arange(
                self.cache_manager.get_position(),
                self.cache_manager.get_position() + seq_len,
                dtype=torch.long,
            ).unsqueeze(0)

        # ── Embed tokens (CPU) ───────────────────────────────────────
        hidden_states_cpu = self.embed_tokens(input_ids).float().to(torch.bfloat16)
        if self.config.embedding_multiplier != 1.0:
            hidden_states_cpu = hidden_states_cpu * self.config.embedding_multiplier

        # ── CPU → TTNN (once) ────────────────────────────────────────
        # Shape [1, seq, H] → [1, 1, seq, H]
        hs4d = hidden_states_cpu.reshape(1, 1, seq_len, self.config.hidden_size)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

        # ── Shared position metadata (CPU, cheap) ───────────────────
        position_embeddings = None  # Granite 4H uses NoPE — no RoPE needed

        cache_position = torch.arange(
            position_ids[0, 0].item(),
            position_ids[0, 0].item() + seq_len,
        )
        has_previous_state = position_ids[0, 0].item() > 0

        if not hasattr(self.cache_manager, "hybrid_cache"):
            from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
                HybridMambaAttentionDynamicCache,
            )
            self.cache_manager.hybrid_cache = HybridMambaAttentionDynamicCache(
                config=self.hf_config,
                batch_size=1,
                dtype=hidden_states_cpu.dtype,
                device=hidden_states_cpu.device,
            )
        self.cache_manager.hybrid_cache.has_previous_state = has_previous_state

        mamba_mask = attention_mask
        if cache_position[0].item() > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None

        is_decode = (seq_len == 1)
        start_pos = position_ids[0, 0].item()

        # ── Update attention position buffers outside any trace ───────
        if is_decode:
            for layer in self.layers:
                if layer.is_attention_layer and layer.simple_attention is not None:
                    layer.simple_attention.update_decode_pos(start_pos)

        # ── Decode trace replay (fast path) ─────────────────────────
        # Trace requires mesh_shape[1]==1 or mesh_shape[0]==1 (not both >1).
        # is_true_2d_mesh() = mesh[0]>1 && mesh[1]>1 → composite_all_gather (dynamic alloc).
        # Tiny 1×4: mesh[0]=1 → safe. Small 8×1: mesh[1]=1 → safe. 2×4 would not be safe.
        _trace_supported = (self.tt_ccl is not None and self.moe_use_all_gather
                            and not self._is_multirow_mesh)
        if is_decode and not _state_only and _trace_supported and self._decode_trace_id is not None:
            new_input = ttnn.from_torch(
                hs4d, device=self.device, dtype=self.config.get_ttnn_dtype(),
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            ttnn.assign(new_input, self._decode_trace_input)
            new_input.deallocate(True)
            self._in_trace = True
            if self.tt_ccl is not None:
                self.tt_ccl._in_trace = True
                # Restore index-0 so replay uses the same handles baked in during capture.
                if hasattr(self.tt_ccl, 'reset_semaphore_indices'):
                    self.tt_ccl.reset_semaphore_indices()
                else:
                    self.tt_ccl.barrier_semaphore_idx = [0, 0, 0]
                    self.tt_ccl.ag_semaphores_idx = [0, 0, 0]
                    self.tt_ccl.rs_semaphores_idx = [0, 0, 0]
            ttnn.execute_trace(self.device, self._decode_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            self._in_trace = False
            if self.tt_ccl is not None: self.tt_ccl._in_trace = False
            trace_logits = self._extract_logits(self._decode_trace_output)
            if use_cache:
                self.cache_manager.increment_position(seq_len)
            return trace_logits

        # ── 40-layer loop — hidden_tt stays on device ────────────────
        hidden_tt = ttnn.from_torch(
            hs4d, device=self.device, dtype=self.config.get_ttnn_dtype(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        for layer_idx, layer in enumerate(self.layers):
            layer_mask = mamba_mask if layer_idx not in self.config.attention_layer_indices else attention_mask
            hidden_tt = layer.forward(
                hidden_tt,
                self.cache_manager,
                position_ids,
                layer_mask,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                has_previous_state=has_previous_state,
            )


        if use_cache:
            self.cache_manager.increment_position(seq_len)

        # ── State-only path: skip norm + LM head (used by non-final prefill chunks) ──
        if _state_only:
            hidden_tt.deallocate(True)
            return None

        # ── Final norm + LM head ─────────────────────────────────────
        mode = Mode.DECODE if seq_len == 1 else Mode.PREFILL
        hidden_tt = self.norm.forward(hidden_tt, mode=mode)
        logits_tt = self.lm_head.forward(hidden_tt)
        hidden_tt.deallocate(True)

        if self.logits_scaling != 1.0:
            logits_tt = ttnn.multiply(logits_tt, 1.0 / self.logits_scaling)

        # ── TTNN → CPU (once) ────────────────────────────────────────
        return self._extract_logits(logits_tt)

    def capture_decode_trace(self):
        """Capture the decode trace using a dummy token at the current position.

        Must be called after prefill + at least one non-trace decode step (to warm up
        all lazy weight uploads and compile kernels). The trace captures the 40-layer
        decode forward; subsequent forward() calls with seq_len==1 will replay it.

        Trace capture runs in bypass mode (ops recorded but not dispatched to device),
        so this does NOT advance KV cache or SSM state. The current position is NOT
        incremented. Call this once after warmup; forward() will auto-replay.
        """
        _trace_supported = (self.tt_ccl is not None and self.moe_use_all_gather
                            and not self._is_multirow_mesh)
        if not _trace_supported:
            print(f"  [trace] SKIPPED: tt_ccl={self.tt_ccl is not None} "
                  f"moe_all_gather={self.moe_use_all_gather} multirow={self._is_multirow_mesh}", flush=True)
            return
        if self._decode_trace_id is not None:
            print(f"  [trace] already captured, skipping", flush=True)
            return

        start_pos = self.cache_manager.get_position()
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

        # Use current position for attention position tensors during capture
        for layer in self.layers:
            if layer.is_attention_layer and layer.simple_attention is not None:
                layer.simple_attention.update_decode_pos(start_pos)

        # Flush pending writes before capture
        ttnn.synchronize_device(self.device)

        # Dummy embedding (zeros — capture only records ops, doesn't affect device state)
        dummy_hs = torch.zeros(1, 1, 1, self.config.hidden_size, dtype=torch.bfloat16)

        self._decode_trace_input = ttnn.from_torch(
            dummy_hs, device=self.device, dtype=self.config.get_ttnn_dtype(),
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._trace_zeros = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.config.hidden_size, dtype=torch.bfloat16),
            device=self.device, dtype=self.config.get_ttnn_dtype(),
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        self._in_trace = True
        if self.tt_ccl is not None:
            self.tt_ccl._in_trace = True
            # Reset semaphore cycle counters so the trace always bakes in index-0 handles.
            if hasattr(self.tt_ccl, 'reset_semaphore_indices'):
                self.tt_ccl.reset_semaphore_indices()
            else:
                self.tt_ccl.barrier_semaphore_idx = [0, 0, 0]
                self.tt_ccl.ag_semaphores_idx = [0, 0, 0]
                self.tt_ccl.rs_semaphores_idx = [0, 0, 0]
        _trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        hidden_tt = ttnn.add(self._decode_trace_input, self._trace_zeros,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        position_ids = torch.tensor([[start_pos]], dtype=torch.long)
        cache_position = torch.arange(start_pos, start_pos + 1)
        has_previous_state = start_pos > 0

        if not hasattr(self.cache_manager, "hybrid_cache"):
            from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
                HybridMambaAttentionDynamicCache,
            )
            self.cache_manager.hybrid_cache = HybridMambaAttentionDynamicCache(
                config=self.hf_config, batch_size=1,
                dtype=torch.bfloat16, device=torch.device("cpu"),
            )
        self.cache_manager.hybrid_cache.has_previous_state = has_previous_state

        for layer_idx, layer in enumerate(self.layers):
            hidden_tt = layer.forward(
                hidden_tt, self.cache_manager, position_ids,
                attention_mask=None,
                position_embeddings=None,
                cache_position=cache_position,
                has_previous_state=has_previous_state,
            )

        mode = Mode.DECODE
        hidden_tt = self.norm.forward(hidden_tt, mode=mode)
        logits_tt = self.lm_head.forward(hidden_tt)
        hidden_tt.deallocate(True)

        if self.logits_scaling != 1.0:
            logits_tt = ttnn.multiply(logits_tt, 1.0 / self.logits_scaling)

        self._decode_trace_output = logits_tt
        ttnn.end_trace_capture(self.device, _trace_id, cq_id=0)
        self._decode_trace_id = _trace_id
        self._in_trace = False
        if self.tt_ccl is not None:
            self.tt_ccl._in_trace = False
            # Reset indices so replays always use the same index-0 semaphore handles
            # that were baked in during capture.
            if hasattr(self.tt_ccl, 'reset_semaphore_indices'):
                self.tt_ccl.reset_semaphore_indices()
            else:
                self.tt_ccl.barrier_semaphore_idx = [0, 0, 0]
                self.tt_ccl.ag_semaphores_idx = [0, 0, 0]
                self.tt_ccl.rs_semaphores_idx = [0, 0, 0]
        print(f"  [trace] capture done, trace_id={_trace_id}", flush=True)

    def _extract_logits(self, logits_tt):
        """Convert TTNN logits tensor to CPU torch tensor."""
        # Logits: [1, 1, S, vocab] → [1, S, vocab]
        if self.is_mesh:
            # LMHead1D shards vocab across devices on dim=-1.
            # Gather along dim=3 (vocab) to reconstruct full logits on device 0.
            logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=3))[0:1]
        else:
            logits = logits_tt.cpu().to_torch()
        if logits_tt is not self._decode_trace_output:
            logits_tt.deallocate(True)
        # Collapse any extra leading dims to [batch, seq, vocab]
        while logits.dim() > 3:
            logits = logits[0]

        # Trim padding from vocab dim if lm_head was padded
        return logits

    def _forward_chunked(self, input_ids, position_ids, attention_mask, use_cache, chunk_sz):
        """
        Prefill long sequences in chunks of chunk_sz tokens.

        Processes each chunk through all 40 layers sequentially.  Mamba SSM state
        and attention KV cache accumulate naturally across chunks.  Only the final
        chunk's last-token logits are returned (sufficient for greedy/sampling decode).
        """
        batch_size, seq_len = input_ids.shape
        start_pos = self.cache_manager.get_position()

        # Build chunks; last chunk may be shorter.
        chunks = []
        for offset in range(0, seq_len, chunk_sz):
            end = min(offset + chunk_sz, seq_len)
            chunk_ids = input_ids[:, offset:end]
            chunk_pos = torch.arange(start_pos + offset, start_pos + end, dtype=torch.long).unsqueeze(0)
            chunks.append((chunk_ids, chunk_pos, offset))

        n = len(chunks)
        last_logits = None
        for i, (chunk_ids, chunk_pos, _) in enumerate(chunks):
            is_last = (i == n - 1)
            # Non-final chunks: skip norm + LM head (their hidden states are discarded).
            # Final chunk: run full forward to get the logits we actually need.
            last_logits = self.forward(
                chunk_ids,
                position_ids=chunk_pos,
                attention_mask=None,
                use_cache=use_cache,
                _state_only=not is_last,
            )

        return last_logits

    def reset_cache(self):
        """Reset KV and Mamba caches for a new sequence. Does NOT release the decode trace."""
        self.cache_manager.reset()

        if hasattr(self.cache_manager, "hybrid_cache"):
            delattr(self.cache_manager, "hybrid_cache")

        for layer in self.layers:
            if hasattr(layer, 'reset_cache'):
                layer.reset_cache()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device,
        dtype: str = "bfloat16",
        torch_dtype=torch.bfloat16,
        trust_remote_code: bool = True,
        verbose: bool = True,
        use_tt_attention: bool = False,
        use_tt_mamba: bool = False,
        use_tt_moe: bool = True,
        mamba_chunk_size: int = None,
        max_cache_length: int = None,
        moe_weight_dtype=None,
        mamba_weight_dtype=None,
        moe_use_all_gather=True,
    ):
        if verbose:
            print(f"\n=== Loading {model_name} ===")

        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        hf_model.eval()  # Set to evaluation mode

        if verbose:
            total_params = sum(p.numel() for p in hf_model.parameters())
            print(f"Loaded HF model: {total_params / 1e6:.1f}M parameters")

        # Create TT config from HF config
        config_kwargs = {"dtype": dtype}
        if max_cache_length is not None:
            config_kwargs["max_cache_length"] = max_cache_length
        tt_config = TTGraniteConfig.from_hf_config(hf_model.config, **config_kwargs)

        if verbose:
            tt_config.print_summary()

        tt_model = cls(
            device,
            hf_model,
            tt_config,
            verbose=verbose,
            use_tt_attention=use_tt_attention,
            use_tt_mamba=use_tt_mamba,
            use_tt_moe=use_tt_moe,
            mamba_chunk_size=mamba_chunk_size,
            moe_weight_dtype=moe_weight_dtype,
            mamba_weight_dtype=mamba_weight_dtype,
            moe_use_all_gather=moe_use_all_gather,
        )

        return tt_model
