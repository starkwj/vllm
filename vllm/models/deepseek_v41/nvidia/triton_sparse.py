# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v41.nvidia.ops.sparse_mla_sm80 import sparse_mla_sm80
from vllm.models.deepseek_v41.nvidia.triton_indexer import DeepseekV41TritonIndexer
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
)


class DeepseekV41TritonSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


class DeepseekV41TritonSparseMLABackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV41TritonSparseMLAMetadataBuilder]:
        return DeepseekV41TritonSparseMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == (8, 0)


class DeepseekV41TritonSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    use_flashmla_tile_scheduler = False


class DeepseekV41TritonSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41TritonSWAMetadataBuilder]:
        return DeepseekV41TritonSWAMetadataBuilder


class DeepseekV41TritonSparseMLAAttention(DeepseekV4Attention):
    backend_cls = DeepseekV41TritonSparseMLABackend
    swa_backend_cls = DeepseekV41TritonSWABackend

    def _indexer_cls(
        self, k_cache: DeepseekV4IndexerCache | None
    ) -> type[DeepseekV41TritonIndexer]:
        return DeepseekV41TritonIndexer

    @staticmethod
    def validate_config(vllm_config: VllmConfig) -> None:
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        attention = vllm_config.attention_config
        name = DeepseekV41TritonSparseMLABackend.get_name()
        if not model.enforce_eager:
            compilation = vllm_config.compilation_config
            if (
                not vllm_config.use_v2_model_runner
                or compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
                or compilation.mode != CompilationMode.NONE
            ):
                raise ValueError(
                    f"{name} requires --enforce-eager or V2 with "
                    "FULL_DECODE_ONLY CUDA graphs and compilation mode NONE."
                )
            if (
                parallel.data_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or parallel.use_ubatching
                or vllm_config.cache_config.swa_bounded_replay
            ):
                raise ValueError(
                    f"{name} decode CUDA graphs require DP=PP=1, no ubatching, "
                    "and --no-swa-bounded-replay."
                )
        if model.dtype != torch.bfloat16:
            raise ValueError(f"{name} requires bfloat16 activations.")
        if (
            parallel.prefill_context_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
        ):
            raise ValueError(f"{name} requires PCP and DCP sizes of 1.")
        if vllm_config.speculative_config is not None:
            raise ValueError(f"{name} does not support speculative decoding.")
        if attention.use_non_causal:
            raise ValueError(f"{name} requires causal attention.")
        if vllm_config.lora_config is not None:
            raise ValueError(f"{name} does not support LoRA.")
        multimodal = model.multimodal_config
        if multimodal is not None and multimodal.get_limit_per_prompt("image") > 0:
            raise ValueError(f"{name} requires --language-model-only for text serving.")
        if not DeepseekV41TritonSparseMLABackend.supports_kv_cache_dtype(
            vllm_config.cache_config.cache_dtype
        ):
            raise ValueError(f"{name} requires --kv-cache-dtype auto or fp8.")
        if attention.indexer_sparse_logits:
            raise ValueError(f"{name} requires indexer_sparse_logits=False.")
        if attention.resolve_indexer_kv_dtype("fp8") != "fp8":
            raise ValueError(f"{name} requires indexer_kv_dtype='fp8'.")
        if attention.resolve_indexer_logits_dtype("float32") != "float32":
            raise ValueError(f"{name} requires indexer_logits_dtype='float32'.")
        topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        if topk_backend not in ("auto", "per_row", "torch"):
            raise ValueError(f"{name} requires indexer top-k auto, per_row or torch.")
        config = model.hf_config
        if (config.head_dim, config.qk_rope_head_dim, config.index_head_dim) != (
            512,
            64,
            128,
        ):
            raise ValueError(
                f"{name} requires head dimensions 512/64/128 (MLA/RoPE/indexer)."
            )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # The shared CUDA RoPE/cache writer instantiates these head counts.
        if num_heads > 128:
            raise ValueError("SM80 DeepSeek V4.1 supports up to 128 local Q heads.")
        return max(8, next_power_of_2(num_heads))

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grouped = self.wo_a.grouped_input(
            attn_out[:, : self.n_local_heads],
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
        )
        weight = self.wo_a.weight
        # ModelOpt's grouped MXFP8 emulation normally dequantizes at load time.
        if weight.dtype == torch.float8_e4m3fn:
            if self._o_proj_block_size != 32:
                raise NotImplementedError("SM80 FP8 WO-A requires MXFP8 scales.")
            weight = dequant_mxfp8_to_bf16(weight, self.wo_a.weight_scale)
        weight = weight.view(self.n_local_groups, self.o_lora_rank, -1)
        z = torch.bmm(grouped.transpose(0, 1), weight.transpose(1, 2))
        return self._wo_b_proj(z.transpose(0, 1).flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            output.zero_()
            return
        assert isinstance(metadata, dict)
        swa = cast(DeepseekSparseSWAMetadata, metadata[self.swa_cache_layer.prefix])
        if swa.prefill_right_visible is not None:
            raise NotImplementedError("TRITON_MLA_SPARSE_DSV41 supports text inputs.")
        num_tokens = swa.num_decode_tokens + swa.num_prefill_tokens
        if num_tokens == 0:
            output.zero_()
            return

        compressed_cache = compressed_indices = compressed_lens = None
        if self.compress_ratio:
            assert self.compressed_cache_prefix is not None
            compressed = cast(
                DeepseekV4FlashMLAMetadata, metadata[self.compressed_cache_prefix]
            )
            assert self.topk_indices_buffer is not None
            assert (
                swa.token_to_req_indices is not None and swa.is_valid_token is not None
            )
            compressed_cache = self._compressed_kv_cache()
            compressed_indices, compressed_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_tokens],
                swa.token_to_req_indices,
                compressed.block_table,
                compressed.block_size // self.compress_ratio,
                swa.is_valid_token,
            )

        for is_decode, rows, indices, lens in (
            (
                True,
                slice(0, swa.num_decode_tokens),
                swa.decode_swa_indices,
                swa.decode_swa_lens,
            ),
            (
                False,
                slice(swa.num_decode_tokens, num_tokens),
                swa.prefill_swa_indices,
                swa.prefill_swa_lens,
            ),
        ):
            if rows.start == rows.stop:
                continue
            assert indices is not None and lens is not None
            sparse_mla_sm80(
                q[rows, : self.n_local_heads],
                output[rows, : self.n_local_heads],
                self.attn_sink,
                positions[rows],
                self.swa_cache_layer.kv_cache,
                indices.squeeze(1),
                lens,
                self.scale,
                compressed_cache,
                compressed_indices[rows] if compressed_indices is not None else None,
                compressed_lens[rows] if compressed_lens is not None else None,
                self.compress_ratio,
                is_decode=is_decode,
            )
        if self.padded_heads > self.n_local_heads:
            output[:num_tokens, self.n_local_heads :].zero_()
        output[num_tokens:].zero_()
