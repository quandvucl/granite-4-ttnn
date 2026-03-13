import ttnn
import torch
from typing import Optional
from transformers import AutoModelForCausalLM
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tt_ops.base import to_tt_tensor, to_torch_tensor
from tt_ops.mlp import TTSharedMLP, split_combined_mlp_weight
from tt_ops.normalization import TTRMSNorm
from tt_ops.cache import HybridKVCacheManager
from tt_model.config import TTGraniteConfig
from tt_model.weight_cache import WeightCache, convert_hf_weights_to_cache
from tt_model.decoder_layer import TTGraniteDecoderLayer
from tt_ops.tt_ccl import SimpleTTCCL


class TTGraniteMoeHybridForCausalLM:
    """
    Tenstorrent-accelerated Granite MoE Hybrid model for causal language modeling.

    Implements hybrid execution:
    - Attention layers (5, 15, 25, 35): Run on TT hardware
    - Mamba layers (all others): Run on CPU
    - Shared MLP: Run on TT hardware
    - Embeddings and LM head: Run on CPU
    """

    def __init__(
        self,
        device,
        hf_model,
        config: TTGraniteConfig,
        verbose: bool = True
    ):
        self.device = device
        self.hf_model = hf_model
        self.config = config
        self.verbose = verbose

        # Extract HF components
        self.embed_tokens = hf_model.model.embed_tokens
        self.lm_head = hf_model.lm_head

        # Convert weights to TT format
        if verbose:
            print("\n=== Initializing TTGraniteMoeHybridForCausalLM ===")

        # For mesh devices, skip weight cache creation (weight distribution not yet implemented)
        is_mesh = hasattr(device, 'get_num_devices')
        if is_mesh:
            if verbose:
                num_devs = device.get_num_devices()
                print(f"  Mesh device detected ({num_devs} devices) - using weight sharding with tensor parallelism")
            self.weight_cache = WeightCache(device, dtype=config.get_ttnn_dtype())  # Empty cache
            # Initialize TT_CCL for collective operations (all-reduce)
            self.tt_ccl = SimpleTTCCL(device) if device.get_num_devices() > 1 else None
        else:
            self.weight_cache = convert_hf_weights_to_cache(
                hf_model, device, config.get_ttnn_dtype(), verbose=verbose
            )
            self.tt_ccl = None

        # Initialize final norm
        # For mesh devices, skip weight cache conversion and use HF weights directly
        is_mesh = hasattr(device, 'get_num_devices')

        if is_mesh:
            # Use HF weights directly for mesh (weight distribution not yet implemented)
            final_norm_weight_torch = hf_model.model.norm.weight
        else:
            # Single device: use weight cache if available
            final_norm_weight = self.weight_cache.get("norm.weight")
            if final_norm_weight is not None:
                final_norm_weight_torch = final_norm_weight.to_torch()
            else:
                final_norm_weight_torch = hf_model.model.norm.weight

        self.norm = TTRMSNorm(
            device,
            final_norm_weight_torch,
            eps=config.rms_norm_eps,
            dtype=config.get_ttnn_dtype()
        )

        # Initialize decoder layers
        self.layers = []
        for layer_idx, hf_layer in enumerate(hf_model.model.layers):
            is_attention = layer_idx in config.attention_layer_indices

            # Create per-layer MLP (LLaMA-style with separate gate/up projections)
            # Enable tensor parallelism for mesh devices using CPU-based reduction
            # (native all_reduce doesn't work, but manual gather-sum-broadcast does)
            use_tensor_parallel = is_mesh and device.get_num_devices() > 1
            layer_mlp = TTSharedMLP(
                device,
                config.hidden_size,
                config.intermediate_size,
                config.get_ttnn_dtype(),
                tensor_parallel=use_tensor_parallel,
                tt_ccl=None  # Not using TT_CCL since native collectives don't work
            )

            # Load MLP weights
            if is_mesh:
                # For mesh devices, shard weights using ShardTensor2dMesh
                hf_mlp = hf_layer.shared_mlp
                input_linear_weight_hf = hf_mlp.input_linear.weight  # [intermediate*2, hidden]
                output_linear_weight_hf = hf_mlp.output_linear.weight  # [hidden, intermediate]

                # Split combined weight in PyTorch (before converting to TTNN)
                # Transpose first: [intermediate*2, hidden] -> [hidden, intermediate*2]
                input_linear_t = input_linear_weight_hf.T  # [hidden, intermediate*2]
                gate_weight_torch = input_linear_t[:, :config.intermediate_size].contiguous()  # [hidden, intermediate]
                up_weight_torch = input_linear_t[:, config.intermediate_size:].contiguous()  # [hidden, intermediate]
                down_weight_torch = output_linear_weight_hf.T  # [intermediate, hidden]

                # Make weights 4D [1, 1, H, W] as required by ShardTensor2dMesh
                gate_weight_4d = gate_weight_torch.unsqueeze(0).unsqueeze(0)
                up_weight_4d = up_weight_torch.unsqueeze(0).unsqueeze(0)
                down_weight_4d = down_weight_torch.unsqueeze(0).unsqueeze(0)

                # Get mesh shape for sharding
                num_devices = device.get_num_devices()
                mesh_shape = device.shape

                # Shard weights for tensor parallelism or replicate if disabled
                if use_tensor_parallel:
                    # Column-parallel for gate/up (shard along width/output dimension)
                    col_mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(-1, None), mesh_shape=mesh_shape)
                    # Row-parallel for down (shard along height/input dimension)
                    row_mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(-2, None), mesh_shape=mesh_shape)

                    layer_mlp.gate_proj_weight = ttnn.operations.core.as_tensor(
                        gate_weight_4d,
                        dtype=config.get_ttnn_dtype(),
                        device=device,
                        mesh_mapper=col_mesh_mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    layer_mlp.up_proj_weight = ttnn.operations.core.as_tensor(
                        up_weight_4d,
                        dtype=config.get_ttnn_dtype(),
                        device=device,
                        mesh_mapper=col_mesh_mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    layer_mlp.down_proj_weight = ttnn.operations.core.as_tensor(
                        down_weight_4d,
                        dtype=config.get_ttnn_dtype(),
                        device=device,
                        mesh_mapper=row_mesh_mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                else:
                    # Replicate weights (no tensor parallelism)
                    mesh_mapper = ttnn.ReplicateTensorToMesh(device)

                    layer_mlp.gate_proj_weight = ttnn.operations.core.as_tensor(
                        gate_weight_4d,
                        dtype=config.get_ttnn_dtype(),
                        device=device,
                        mesh_mapper=mesh_mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    layer_mlp.up_proj_weight = ttnn.operations.core.as_tensor(
                        up_weight_4d,
                        dtype=config.get_ttnn_dtype(),
                        device=device,
                        mesh_mapper=mesh_mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    layer_mlp.down_proj_weight = ttnn.operations.core.as_tensor(
                        down_weight_4d,
                        dtype=config.get_ttnn_dtype(),
                        device=device,
                        mesh_mapper=mesh_mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
            else:
                # Single device: load from cache (already transposed and in TILE layout)
                input_linear_weight = self.weight_cache.get(f"layers.{layer_idx}.shared_mlp.input_linear.weight")
                output_linear_weight = self.weight_cache.get(f"layers.{layer_idx}.shared_mlp.output_linear.weight")

                # Split combined weight into gate_proj and up_proj using helper
                if input_linear_weight is not None:
                    layer_mlp.gate_proj_weight, layer_mlp.up_proj_weight = split_combined_mlp_weight(
                        input_linear_weight,
                        device,
                        config.get_ttnn_dtype(),
                        config.hidden_size,
                        config.intermediate_size
                    )

                layer_mlp.down_proj_weight = output_linear_weight

            layer = TTGraniteDecoderLayer(
                device=device,
                layer_idx=layer_idx,
                config=config,
                hf_layer=hf_layer,
                shared_mlp=layer_mlp,
                weight_cache=self.weight_cache,
                is_attention_layer=is_attention,
                hf_config=hf_model.config,
                dtype=config.get_ttnn_dtype(),
                tensor_parallel=use_tensor_parallel,
                tt_ccl=self.tt_ccl if use_tensor_parallel else None
            )

            self.layers.append(layer)

            if verbose and (layer_idx < 5 or layer_idx % 10 == 0):
                layer_type = "Attention" if is_attention else "Mamba"
                print(f"  Initialized layer {layer_idx} ({layer_type})")

        # Initialize KV cache manager
        self.cache_manager = HybridKVCacheManager(
            device=device,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_cache_length=config.max_cache_length,
            batch_size=config.batch_size,
            attention_layer_indices=config.attention_layer_indices,
            dtype=config.get_ttnn_dtype()
        )

        if verbose:
            print(f"\n✓ Model initialized with {len(self.layers)} layers")
            print(f"  - {len(config.attention_layer_indices)} attention layers on TT")
            print(f"  - {config.num_hidden_layers - len(config.attention_layer_indices)} mamba layers on CPU")
            if is_mesh and device.get_num_devices() > 1:
                print(f"  - Mesh device: {device.get_num_devices()} devices")
                if use_tensor_parallel:
                    print(f"  - Tensor parallelism: ENABLED")
                    print(f"    • MLP weights sharded across devices (column/row parallel)")
                    print(f"    • Reduction: CPU-based gather-sum-broadcast")
                    print(f"    • Expected speedup: ~20-25x for MLP layers")
                else:
                    print(f"  - Tensor parallelism: DISABLED (using replicated weights)")
                    print(f"  - All devices compute same result (redundant but correct)")
            self.weight_cache.print_summary()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            position_ids: Optional [batch_size, seq_len] position IDs
            attention_mask: Optional attention mask
            use_cache: Whether to use KV cache (for generation)

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                self.cache_manager.get_position(),
                self.cache_manager.get_position() + seq_len,
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens (on CPU)
        hidden_states = self.embed_tokens(input_ids)  # [batch, seq, hidden_size]

        # DEBUG: Track scale through layers
        DEBUG_SCALE = False
        if DEBUG_SCALE and seq_len > 1:
            print(f"[DEBUG] After embeddings (before scaling): range=[{hidden_states.min():.3f}, {hidden_states.max():.3f}], mean={hidden_states.mean():.6f}")

        # Apply embedding multiplier (granite-1b uses 12.0)
        if self.config.embedding_multiplier != 1.0:
            hidden_states = hidden_states * self.config.embedding_multiplier

        if DEBUG_SCALE and seq_len > 1:
            print(f"[DEBUG] After embeddings (scaled by {self.config.embedding_multiplier}): range=[{hidden_states.min():.3f}, {hidden_states.max():.3f}], mean={hidden_states.mean():.6f}")

        # Process through all decoder layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward(
                hidden_states,
                self.cache_manager,
                position_ids,
                attention_mask
            )

            # DEBUG: Track scale at key layers
            if DEBUG_SCALE and seq_len > 1 and layer_idx in [0, 1, 2, 10, 20, 30, 39]:
                layer_type = "Attention" if layer.is_attention_layer else "Mamba"
                abs_mean = hidden_states.abs().mean().item()
                print(f"[DEBUG] Layer {layer_idx:2d} ({layer_type:9s}): range=[{hidden_states.min():7.3f}, {hidden_states.max():7.3f}], abs_mean={abs_mean:7.3f}")

            # Only print during prefill (seq_len > 1), not during decode
            if self.verbose and layer_idx < 3 and seq_len > 1:
                layer_type = "Attention" if layer.is_attention_layer else "Mamba"
                print(f"    Layer {layer_idx} ({layer_type}): shape={hidden_states.shape}")

        # Final norm (on TT) - single conversion round-trip
        if DEBUG_SCALE and seq_len > 1:
            print(f"[DEBUG] Before final norm: range=[{hidden_states.min():7.3f}, {hidden_states.max():7.3f}], abs_mean={hidden_states.abs().mean():7.3f}")

        # Mesh mapper for input replication
        mesh_mapper = None
        if hasattr(self.device, 'get_num_devices') and self.device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)

        hidden_states_tt = to_tt_tensor(
            hidden_states,
            self.device,
            self.config.get_ttnn_dtype(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper
        )
        hidden_states_tt = self.norm(hidden_states_tt)
        hidden_states = to_torch_tensor(hidden_states_tt, target_shape=hidden_states.shape)

        if DEBUG_SCALE and seq_len > 1:
            print(f"[DEBUG] After final norm: range=[{hidden_states.min():7.3f}, {hidden_states.max():7.3f}], abs_mean={hidden_states.abs().mean():7.3f}")

        # LM head (on CPU - large vocab matmul)
        logits = self.lm_head(hidden_states)  # [batch, seq, vocab_size]

        if DEBUG_SCALE and seq_len > 1:
            print(f"[DEBUG] After LM head (before scaling): range=[{logits[0,-1].min():7.3f}, {logits[0,-1].max():7.3f}], abs_mean={logits[0,-1].abs().mean():7.3f}")

        # Apply logits scaling (granite-1b uses scaling factor of 6.0)
        if self.config.logits_scaling != 1.0:
            logits = logits / self.config.logits_scaling

        if DEBUG_SCALE and seq_len > 1:
            print(f"[DEBUG] After logits scaling (/{self.config.logits_scaling}): abs_mean={logits[0,-1].abs().mean():7.3f}")

        # Update position tracker if using cache
        if use_cache:
            self.cache_manager.increment_position(seq_len)

        return logits

    def reset_cache(self):
        """Reset KV cache for new generation."""
        self.cache_manager.reset()
        # Reset hybrid cache if it exists
        if hasattr(self.cache_manager, 'hybrid_cache'):
            delattr(self.cache_manager, 'hybrid_cache')

    def print_cache_stats(self):
        """Print cache statistics."""
        self.cache_manager.print_summary()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device,
        dtype: str = "bfloat16",
        torch_dtype=torch.bfloat16,
        trust_remote_code: bool = True,
        verbose: bool = True
    ):
        """
        Load model from HuggingFace and convert to TT format.

        Args:
            model_name: HuggingFace model name (e.g., "ibm-granite/granite-4.0-h-1b")
            device: TTNN device
            dtype: TT dtype ("bfloat16" or "bfloat8")
            torch_dtype: PyTorch dtype for loading HF model
            trust_remote_code: Trust remote code when loading
            verbose: Print progress

        Returns:
            TTGraniteMoeHybridForCausalLM instance
        """
        if verbose:
            print(f"\n=== Loading {model_name} ===")

        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        )
        hf_model.eval()  # Set to evaluation mode

        if verbose:
            total_params = sum(p.numel() for p in hf_model.parameters())
            print(f"Loaded HF model: {total_params / 1e6:.1f}M parameters")

        # Create TT config from HF config
        tt_config = TTGraniteConfig.from_hf_config(hf_model.config, dtype=dtype)

        if verbose:
            tt_config.print_summary()

        # Create TT model
        tt_model = cls(device, hf_model, tt_config, verbose=verbose)

        return tt_model
