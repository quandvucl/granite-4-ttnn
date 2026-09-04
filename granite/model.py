"""TT-accelerated Granite hybrid model for causal language modeling."""

import sys
from pathlib import Path
from typing import Optional

import torch
import ttnn
from transformers import AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent))
from granite import MambaCacheManager
from granite.config import TTGraniteConfig
from granite.decoder_layer import TTGraniteDecoderLayer
from models.common.modules.embedding.embedding_1d import Embedding1D
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode


class ReplicatedMLP:
    """SwiGLU MLP with full weights replicated on every device.

    Each device runs the full matmul independently - no communication needed.
    Used on the tiny 4-device (1x4) mesh where 40 all-gathers per forward pass
    would cost more than the memory saved by sharding.
    """

    def __init__(self, w1, w2, w3, dtype):
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2
        self.dtype = dtype

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # SwiGLU: out = (silu(x @ w_gate) * (x @ w_up)) @ w_down
        # gate = silu(x @ w1), up = x @ w3, mid = gate * up
        MC = ttnn.L1_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w1, dtype=self.dtype, memory_config=MC)
        up = ttnn.linear(x, self.w3, dtype=self.dtype, memory_config=MC)
        gate = ttnn.silu(gate, memory_config=MC)
        mid = ttnn.mul(gate, up, memory_config=MC)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(mid, self.w2, dtype=self.dtype, memory_config=MC)
        mid.deallocate(True)
        return out


class ColumnParallelMLP:
    """SwiGLU MLP where w_gate and w_up are column-sharded across mesh columns.

    Each of the num_cols devices computes a partial w_down output (its own shard of
    the intermediate dimension); a ring all-gather + sum across those num_cols devices
    then reconstructs the full hidden_size output on every device. Used on >4-device meshes where weight memory savings
    outweigh the all-gather cost (vs ReplicatedMLP which skips communication entirely).
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

    def forward(self, x):
        MC = ttnn.L1_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w1, dtype=self.dtype, memory_config=MC)
        up = ttnn.linear(x, self.w3, dtype=self.dtype, memory_config=MC)
        gate = ttnn.silu(gate, memory_config=MC)
        mid = ttnn.mul(gate, up, memory_config=MC)
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
                    dim=1,
                    cluster_axis=ax,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(
                        ax
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(
                        ax
                    ),
                    num_links=self.tt_ccl.get_num_links(ax),
                    memory_config=MC,
                    topology=self._topology,
                    chunks_per_sync=1,
                    num_workers_per_link=1,
                    num_buffers_per_channel=2,
                )
            else:
                gathered = ttnn.all_gather(
                    out, dim=1, cluster_axis=ax, memory_config=MC
                )
            out.deallocate(True)
            out = ttnn.sum(gathered, dim=1, keepdim=True, memory_config=MC)
            gathered.deallocate(True)
        return out


class TTGraniteMoeHybridForCausalLM:
    """
    The main Granite hybrid model class.
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
        fabric_1d: bool = False,
        use_conv1d_kernel: bool = True,
        use_ssm_kernel: bool = True,
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
        self.moe_weight_dtype = (
            moe_weight_dtype if moe_weight_dtype is not None else ttnn.bfloat8_b
        )
        self.mamba_weight_dtype = mamba_weight_dtype
        self.use_conv1d_kernel = use_conv1d_kernel
        self.use_ssm_kernel = use_ssm_kernel

        self.embed_tokens = hf_model.model.embed_tokens

        if verbose:
            print("  Initializing LM head...")

        lm_head_weight = hf_model.lm_head.weight.T.contiguous()
        vocab_size = lm_head_weight.shape[1]
        self.logits_scaling = config.logits_scaling

        padded_vocab_size = ((vocab_size + 31) // 32) * 32
        if vocab_size < padded_vocab_size:
            lm_head_weight = torch.cat(
                [
                    lm_head_weight,
                    torch.zeros(
                        lm_head_weight.shape[0],
                        padded_vocab_size - vocab_size,
                        dtype=lm_head_weight.dtype,
                    ),
                ],
                dim=-1,
            )
        lm_head_lazy = LazyWeight(
            source=lm_head_weight,
            device=device,
            dtype=config.get_ttnn_dtype(),
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
        self.embedding = Embedding1D(
            embedding_lazy, embed_scale=config.embedding_multiplier
        )
        self._use_tt_embeddings = is_mesh

        if is_mesh:
            if verbose:
                print(
                    f"  Mesh device detected ({device.get_num_devices()} devices) - tensor parallelism enabled"
                )
            self.tt_ccl = get_tt_ccl(device) if device.get_num_devices() > 1 else None
            _ms = device.shape
            # FABRIC_1D never triggers composite all-gather (no dynamic alloc), so
            # multirow meshes are trace-safe when fabric_1d=True.
            self._is_multirow_mesh = _ms[0] > 1 and _ms[1] > 1 and not fabric_1d
        else:
            self.tt_ccl = None
            self._is_multirow_mesh = False

        if verbose:
            print("  Initializing final norm (RMSNorm)...")

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

        self.layers = []
        for layer_idx, hf_layer in enumerate(hf_model.model.layers):
            is_attention = layer_idx in config.attention_layer_indices

            hf_mlp = hf_layer.shared_mlp

            input_linear_weight_hf = hf_mlp.input_linear.weight
            output_linear_weight_hf = hf_mlp.output_linear.weight

            input_linear_t = input_linear_weight_hf.T.contiguous()
            actual_intermediate_size = input_linear_t.shape[1] // 2

            gate_weight = input_linear_t[:, :actual_intermediate_size].contiguous()
            up_weight = input_linear_t[:, actual_intermediate_size:].contiguous()
            down_weight = output_linear_weight_hf.T.contiguous()

            _num_dev = device.get_num_devices() if is_mesh else 1
            act_dtype = config.get_ttnn_dtype()
            # If hidden_size > 2048, use bfloat8 for MoE weights to save memory
            # otherwise, use the same dtype as activations to avoid precision loss.
            wt_dtype = self.moe_weight_dtype if config.hidden_size > 2048 else act_dtype

            # Tensor parallelism is used on meshes with more than 4 devices and more than 1 column.
            mesh_shape = device.shape if is_mesh else ttnn.MeshShape(1, 1)
            num_cols = mesh_shape[1] if is_mesh else 1
            use_tensor_parallel = is_mesh and num_cols > 1 and _num_dev > 4

            if use_tensor_parallel:
                num_tp, cluster_axis = num_cols, 1
                fwd_mapper = ttnn.ShardTensor2dMesh(
                    device, dims=(None, 1), mesh_shape=mesh_shape
                )
                rev_mapper = ttnn.ShardTensor2dMesh(
                    device, dims=(None, 0), mesh_shape=mesh_shape
                )
                F = gate_weight.shape[1]  # The intermediate dimension of the MLP
                if F % num_tp == 0:

                    def _upload_fwd(t):
                        return ttnn.from_torch(
                            t,
                            device=device,
                            dtype=wt_dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=fwd_mapper,
                        )

                    def _upload_rev(t):
                        return ttnn.from_torch(
                            t,
                            device=device,
                            dtype=wt_dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=rev_mapper,
                        )

                    layer_mlp = ColumnParallelMLP(
                        w1=_upload_fwd(gate_weight),
                        w2=_upload_rev(down_weight),
                        w3=_upload_fwd(up_weight),
                        num_cols=num_tp,
                        dtype=act_dtype,
                        tt_ccl=self.tt_ccl,
                        cluster_axis=cluster_axis,
                    )
                else:
                    # If the intermediate dimension is not divisible by the number of columns, disable tensor parallelism
                    use_tensor_parallel = False

            if not use_tensor_parallel:

                def _upload_replicated(t):
                    return ttnn.from_torch(
                        t,
                        device=device,
                        dtype=wt_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=(
                            ttnn.ReplicateTensorToMesh(device) if is_mesh else None
                        ),
                    )

                layer_mlp = ReplicatedMLP(
                    w1=_upload_replicated(gate_weight),
                    w2=_upload_replicated(down_weight),
                    w3=_upload_replicated(up_weight),
                    dtype=act_dtype,
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
                use_conv1d_kernel=self.use_conv1d_kernel,
                use_ssm_kernel=self.use_ssm_kernel,
            )

            self.layers.append(layer)

            layer_type = "Attention" if is_attention else "Mamba"
            print(f"  layer {layer_idx:2d} ({layer_type}) done", flush=True)

        self.cache_manager = MambaCacheManager(
            num_layers=config.num_hidden_layers,
            batch_size=config.batch_size,
            attention_layer_indices=config.attention_layer_indices,
        )

        self._decode_trace_id = None
        self._decode_trace_input = None
        self._decode_trace_output = None
        self._trace_zeros = None
        self._in_trace = False
        if self.tt_ccl is not None:
            self.tt_ccl._in_trace = False

        _trace_supported_init = self.tt_ccl is not None and self.is_mesh
        if _trace_supported_init:
            # Group specific sub-devices together so they stall (or wait) synchronously,
            # ensuring proper hardware serialization and preventing race conditions during parallel execution
            self.device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

        del self.hf_model
        self.hf_model = None

        if verbose:
            print(f"\nModel initialized with {len(self.layers)} layers")
            print(f"  - {len(config.attention_layer_indices)} attention layers")
            print(
                f"  - {config.num_hidden_layers - len(config.attention_layer_indices)} mamba layers"
            )
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
        The forward pass.
        """
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, f"Only batch_size=1 supported, got {batch_size}"

        chunk_sz = self.mamba_chunk_size
        if chunk_sz is not None and seq_len > chunk_sz and seq_len > 1:
            return self._forward_chunked(input_ids, use_cache, chunk_sz)

        if position_ids is None:
            position_ids = torch.arange(
                self.cache_manager.get_position(),
                self.cache_manager.get_position() + seq_len,
                dtype=torch.long,
            ).unsqueeze(0)

        hidden_states_cpu = self.embed_tokens(input_ids).float().to(torch.bfloat16)
        if self.config.embedding_multiplier != 1.0:
            hidden_states_cpu = hidden_states_cpu * self.config.embedding_multiplier

        hs4d = hidden_states_cpu.reshape(1, 1, seq_len, self.config.hidden_size)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

        position_embeddings = None  # Granite 4H uses NoPE - no RoPE needed

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
        if cache_position[0].item() > 0 or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            mamba_mask = None

        is_decode = seq_len == 1
        start_pos = position_ids[0, 0].item()

        if is_decode:
            for layer in self.layers:
                if layer.is_attention_layer and layer.simple_attention is not None:
                    layer.simple_attention.update_decode_pos(start_pos)

        _trace_supported = (
            self.tt_ccl is not None
            and self.moe_use_all_gather
            and not self._is_multirow_mesh
        )
        if (
            is_decode
            and not _state_only
            and _trace_supported
            and self._decode_trace_id is not None
        ):
            new_input = ttnn.from_torch(
                hs4d,
                device=self.device,
                dtype=self.config.get_ttnn_dtype(),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            ttnn.assign(new_input, self._decode_trace_input)
            new_input.deallocate(True)
            self._in_trace = True
            if self.tt_ccl is not None:
                self.tt_ccl._in_trace = True
                if hasattr(self.tt_ccl, "reset_semaphore_indices"):
                    self.tt_ccl.reset_semaphore_indices()
                else:
                    self.tt_ccl.barrier_semaphore_idx = [0, 0, 0]
                    self.tt_ccl.ag_semaphores_idx = [0, 0, 0]
                    self.tt_ccl.rs_semaphores_idx = [0, 0, 0]
            ttnn.execute_trace(
                self.device, self._decode_trace_id, cq_id=0, blocking=False
            )
            ttnn.synchronize_device(self.device)
            self._in_trace = False
            if self.tt_ccl is not None:
                self.tt_ccl._in_trace = False
            trace_logits = self._extract_logits(self._decode_trace_output)
            if use_cache:
                self.cache_manager.increment_position(seq_len)
            return trace_logits

        hidden_tt = ttnn.from_torch(
            hs4d,
            device=self.device,
            dtype=self.config.get_ttnn_dtype(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        for layer_idx, layer in enumerate(self.layers):
            layer_mask = (
                mamba_mask
                if layer_idx not in self.config.attention_layer_indices
                else attention_mask
            )
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

        if _state_only:
            hidden_tt.deallocate(True)
            return None

        mode = Mode.DECODE if seq_len == 1 else Mode.PREFILL
        hidden_tt = self.norm.forward(hidden_tt, mode=mode)
        logits_tt = self.lm_head.forward(hidden_tt)
        hidden_tt.deallocate(True)

        if self.logits_scaling != 1.0:
            logits_tt = ttnn.multiply(logits_tt, 1.0 / self.logits_scaling)

        return self._extract_logits(logits_tt)

    def capture_decode_trace(self):
        """Capture the decode trace for fast decode replay."""
        _trace_supported = (
            self.tt_ccl is not None
            and self.moe_use_all_gather
            and not self._is_multirow_mesh
        )
        if not _trace_supported:
            print(
                f"  [trace] SKIPPED: tt_ccl={self.tt_ccl is not None} "
                f"moe_all_gather={self.moe_use_all_gather} multirow={self._is_multirow_mesh}",
                flush=True,
            )
            return
        if self._decode_trace_id is not None:
            print(f"  [trace] already captured, skipping", flush=True)
            return

        start_pos = self.cache_manager.get_position()
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None

        for layer in self.layers:
            if layer.is_attention_layer and layer.simple_attention is not None:
                layer.simple_attention.update_decode_pos(start_pos)

        ttnn.synchronize_device(self.device)

        dummy_hs = torch.zeros(1, 1, 1, self.config.hidden_size, dtype=torch.bfloat16)

        self._decode_trace_input = ttnn.from_torch(
            dummy_hs,
            device=self.device,
            dtype=self.config.get_ttnn_dtype(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._trace_zeros = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.config.hidden_size, dtype=torch.bfloat16),
            device=self.device,
            dtype=self.config.get_ttnn_dtype(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        self._in_trace = True
        if self.tt_ccl is not None:
            self.tt_ccl._in_trace = True
            # Reset semaphore cycle counters so the trace always bakes in index-0 handles.
            if hasattr(self.tt_ccl, "reset_semaphore_indices"):
                self.tt_ccl.reset_semaphore_indices()
            else:
                self.tt_ccl.barrier_semaphore_idx = [0, 0, 0]
                self.tt_ccl.ag_semaphores_idx = [0, 0, 0]
                self.tt_ccl.rs_semaphores_idx = [0, 0, 0]
        _trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        hidden_tt = ttnn.add(
            self._decode_trace_input,
            self._trace_zeros,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        position_ids = torch.tensor([[start_pos]], dtype=torch.long)
        cache_position = torch.arange(start_pos, start_pos + 1)
        has_previous_state = start_pos > 0

        if not hasattr(self.cache_manager, "hybrid_cache"):
            from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
                HybridMambaAttentionDynamicCache,
            )

            self.cache_manager.hybrid_cache = HybridMambaAttentionDynamicCache(
                config=self.hf_config,
                batch_size=1,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
        self.cache_manager.hybrid_cache.has_previous_state = has_previous_state

        for layer_idx, layer in enumerate(self.layers):
            hidden_tt = layer.forward(
                hidden_tt,
                self.cache_manager,
                position_ids,
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
            if hasattr(self.tt_ccl, "reset_semaphore_indices"):
                self.tt_ccl.reset_semaphore_indices()
            else:
                self.tt_ccl.barrier_semaphore_idx = [0, 0, 0]
                self.tt_ccl.ag_semaphores_idx = [0, 0, 0]
                self.tt_ccl.rs_semaphores_idx = [0, 0, 0]
        print(f"  [trace] capture done, trace_id={_trace_id}", flush=True)

    def _extract_logits(self, logits_tt):
        """Convert TTNN logits tensor to CPU torch tensor, gathering sharded vocab dim."""
        if self.is_mesh:
            logits = ttnn.to_torch(
                logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=3)
            )[0:1]
        else:
            logits = logits_tt.cpu().to_torch()
        if logits_tt is not self._decode_trace_output:
            logits_tt.deallocate(True)
        while logits.dim() > 3:
            logits = logits[0]
        return logits

    def _forward_chunked(self, input_ids, use_cache, chunk_sz):
        """Prefill long sequences in chunks to reduce peak DRAM allocation."""
        batch_size, seq_len = input_ids.shape
        start_pos = self.cache_manager.get_position()

        chunks = []
        for offset in range(0, seq_len, chunk_sz):
            end = min(offset + chunk_sz, seq_len)
            chunk_ids = input_ids[:, offset:end]
            chunk_pos = torch.arange(
                start_pos + offset, start_pos + end, dtype=torch.long
            ).unsqueeze(0)
            chunks.append((chunk_ids, chunk_pos, offset))

        n = len(chunks)
        last_logits = None
        for i, (chunk_ids, chunk_pos, _) in enumerate(chunks):
            is_last = i == n - 1
            last_logits = self.forward(
                chunk_ids,
                position_ids=chunk_pos,
                attention_mask=None,
                use_cache=use_cache,
                _state_only=not is_last,
            )

        return last_logits

    def reset_cache(self):
        """Reset KV and Mamba caches for a new sequence."""
        self.cache_manager.reset()

        if hasattr(self.cache_manager, "hybrid_cache"):
            delattr(self.cache_manager, "hybrid_cache")

        for layer in self.layers:
            if hasattr(layer, "reset_cache"):
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
        fabric_1d: bool = False,
        use_conv1d_kernel: bool = True,
        use_ssm_kernel: bool = True,
    ):
        """Load HuggingFace weights and initialize the TT model."""
        if verbose:
            print(f"\n=== Loading {model_name} ===")

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        hf_model.eval()

        if verbose:
            total_params = sum(p.numel() for p in hf_model.parameters())
            print(f"Loaded HF model: {total_params / 1e6:.1f}M parameters")

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
            fabric_1d=fabric_1d,
            use_conv1d_kernel=use_conv1d_kernel,
            use_ssm_kernel=use_ssm_kernel,
        )

        return tt_model

    @classmethod
    def from_hf_model(
        cls,
        hf_model,
        device,
        dtype: str = "bfloat16",
        verbose: bool = True,
        use_tt_attention: bool = False,
        use_tt_mamba: bool = False,
        use_tt_moe: bool = True,
        mamba_chunk_size: int = None,
        max_cache_length: int = None,
        moe_weight_dtype=None,
        mamba_weight_dtype=None,
        moe_use_all_gather=True,
        fabric_1d: bool = False,
        use_conv1d_kernel: bool = True,
        use_ssm_kernel: bool = True,
    ):
        """Initialize from an already-loaded HF model - skips HF download/load cost."""
        config_kwargs = {"dtype": dtype}
        if max_cache_length is not None:
            config_kwargs["max_cache_length"] = max_cache_length
        tt_config = TTGraniteConfig.from_hf_config(hf_model.config, **config_kwargs)
        if verbose:
            tt_config.print_summary()
        return cls(
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
            fabric_1d=fabric_1d,
            use_conv1d_kernel=use_conv1d_kernel,
            use_ssm_kernel=use_ssm_kernel,
        )
