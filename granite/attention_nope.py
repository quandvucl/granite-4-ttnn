"""TTNN Attention for Granite (NoPE - No Position Embedding)."""

import torch
import ttnn


class AttentionNoPE:
    """On-device attention without position embeddings, with paged KV cache."""

    def __init__(
        self,
        device,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        max_seq_len: int = 2048,
        dtype=ttnn.bfloat16,
        layer_idx: int = 0,
    ):
        self.device = device
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = 1.0 / (head_dim**0.5)
        self.dtype = dtype
        self.layer_idx = layer_idx

        is_mesh = hasattr(device, "get_num_devices") and device.get_num_devices() > 1
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

        # Upload weights to device and convert to TTNN tensors.
        def _upload(t, dt=dtype):
            return ttnn.from_torch(
                t,
                device=device,
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

        # Combine Q/K/V weights into a single tensor for efficiency.
        qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim
        qkv_weight = torch.zeros((qkv_size, hidden_size), dtype=torch.bfloat16)
        q_end = num_heads * head_dim
        k_end = q_end + num_kv_heads * head_dim
        qkv_weight[:q_end, :] = q_weight
        qkv_weight[q_end:k_end, :] = k_weight
        qkv_weight[k_end:, :] = v_weight

        # Upload weights to device.
        self.wqkv = _upload(qkv_weight.T.contiguous())
        self.wo = _upload(o_weight.to(torch.bfloat16).T.contiguous())

        # Initialize KV cache on device.
        cache_k = torch.zeros((1, num_kv_heads, max_seq_len, head_dim))
        cache_v = torch.zeros((1, num_kv_heads, max_seq_len, head_dim))
        self.cache_k = _upload(cache_k, dt=ttnn.bfloat16)
        self.cache_v = _upload(cache_v, dt=ttnn.bfloat16)

        self._mesh_mapper = mesh_mapper
        self._is_mesh = is_mesh

        # Pre-allocated on-device position scalar; updated outside any TTNN trace via copy_host_to_device_tensor.
        self._cur_pos_tt = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def update_decode_pos(self, pos: int):
        """Write the current decode position to the on-device tensor.

        Must be called before forward() and outside any TTNN trace capture,
        because copy_host_to_device_tensor is not traceable.
        """
        host_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_tt, self._cur_pos_tt)

    def forward(
        self,
        hidden_states,
        position_ids: torch.Tensor = None,
    ):
        """Run prefill or decode attention (NoPE, on-device KV cache)."""
        from utils import to_tt_tensor

        if isinstance(hidden_states, torch.Tensor):
            batch_size, seq_len, _ = hidden_states.shape
            hidden_tt = to_tt_tensor(
                hidden_states,
                self.device,
                self.dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self._mesh_mapper,
            )
        else:
            hidden_tt = hidden_states
            batch_size = 1
            seq_len = (
                hidden_tt.shape[1] if len(hidden_tt.shape) == 3 else hidden_tt.shape[-2]
            )

        # Determine if this is a decode step (seq_len == 1)
        # or a prefill step (seq_len > 1).
        is_decode = seq_len == 1
        # If position_ids is provided, use it to determine the starting position
        # otherwise, default to 0.
        start_pos = position_ids[0, 0].item() if position_ids is not None else 0

        # Compute Q/K/V heads for 2 cases: decode (seq_len=1) and prefill (seq_len>1).
        xqkv_fused = ttnn.linear(
            hidden_tt,
            self.wqkv,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if is_decode:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])
            q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, batch_size * seq_len, -1])
            q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(
            xqkv_fused
        )  # Free the fused QKV tensor after splitting into heads.

        keys, values = self.cache_k, self.cache_v

        # Update the KV cache with the new keys and values.
        if is_decode:
            # _cur_pos_tt must be updated by caller (update_decode_pos) before this
            # forward call so that copy_host_to_device_tensor runs outside any trace.
            ttnn.experimental.paged_update_cache(
                keys, k_heads, update_idxs_tensor=self._cur_pos_tt, batch_offset=0
            )
            ttnn.experimental.paged_update_cache(
                values, v_heads, update_idxs_tensor=self._cur_pos_tt, batch_offset=0
            )
            ttnn.deallocate(k_heads)
            ttnn.deallocate(v_heads)

            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads,
                keys,
                values,
                cur_pos_tensor=self._cur_pos_tt,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_heads)

            attn_output = ttnn.transpose(attn_output, 1, 2)
            attn_output_concat = ttnn.experimental.nlp_concat_heads(
                attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(attn_output)

        else:
            block_size = 32
            total_len = start_pos + seq_len
            num_blocks = (total_len + block_size - 1) // block_size

            if start_pos == 0:
                page_table_torch = torch.arange(
                    num_blocks, dtype=torch.int32
                ).unsqueeze(0)
                page_table = ttnn.from_torch(
                    page_table_torch,
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self._mesh_mapper,
                )
                ttnn.experimental.paged_fill_cache(
                    keys, k_heads, page_table, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    values, v_heads, page_table, batch_idx=0
                )
                ttnn.deallocate(page_table)
                attn_output = ttnn.transformer.scaled_dot_product_attention(
                    q_heads,
                    k_heads,
                    v_heads,
                    is_causal=True,
                    scale=self.scale,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(q_heads)
                ttnn.deallocate(k_heads)
                ttnn.deallocate(v_heads)
            else:
                first_new_block = start_pos // block_size
                new_blocks = num_blocks - first_new_block
                page_table_torch = torch.arange(
                    first_new_block, first_new_block + new_blocks, dtype=torch.int32
                ).unsqueeze(0)
                page_table = ttnn.from_torch(
                    page_table_torch,
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self._mesh_mapper,
                )
                ttnn.experimental.paged_fill_cache(
                    keys, k_heads, page_table, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    values, v_heads, page_table, batch_idx=0
                )
                ttnn.deallocate(page_table)
                ttnn.deallocate(k_heads)
                ttnn.deallocate(v_heads)

                row = torch.arange(seq_len).unsqueeze(1)
                col = torch.arange(total_len).unsqueeze(0)
                mask_torch = (col <= (start_pos + row)).to(torch.bfloat16)
                # Create a mask tensor for attention, where positions beyond the current decode position are masked with -inf.
                mask_add = (
                    torch.where(
                        mask_torch.bool(),
                        torch.zeros_like(mask_torch),
                        torch.full_like(mask_torch, float("-inf")),
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                mask_tt = ttnn.from_torch(
                    mask_add,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self._mesh_mapper,
                )
                k_full = self.cache_k[:, :, :total_len, :]
                v_full = self.cache_v[:, :, :total_len, :]
                attn_output = ttnn.transformer.scaled_dot_product_attention(
                    q_heads,
                    k_full,
                    v_full,
                    attn_mask=mask_tt,
                    is_causal=False,
                    scale=self.scale,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(mask_tt)
                ttnn.deallocate(q_heads)
                ttnn.deallocate(k_full)
                ttnn.deallocate(v_full)

            attn_output = ttnn.reshape(
                attn_output, [1, self.num_heads, -1, self.head_dim]
            )
            attn_output_concat = ttnn.experimental.nlp_concat_heads(
                attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(attn_output)

        output_tt = ttnn.linear(
            attn_output_concat,
            self.wo,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_concat)

        output_tt = ttnn.reshape(output_tt, [1, 1, seq_len, self.hidden_size])
        return output_tt
