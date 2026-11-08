# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    mask_invalid_topk,
    select_candidate_blocks,
)
from vllm.model_executor.kernels.attention.dsa.triton_mqa_logits import paged_mqa_logits
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import max_decode_query_len
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    dense_logits_rows,
)


class DeepseekV41TritonSparseAttnIndexer(SparseAttnIndexer):
    requires_deep_gemm = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        scheduler = config.scheduler_config
        max_rows = scheduler.max_num_batched_tokens
        width = self.max_model_len
        prefill_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024**2
        self.decode_budget = envs.VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB * 1024**2
        # A budget always admits at least one row, even at the maximum width.
        prefill_elements = min(max_rows * width, max(width, prefill_budget // 4))
        decode_rows = min(
            max_rows, scheduler.max_num_seqs * max_decode_query_len(config)
        )
        decode_elements = decode_rows * width
        if self.decode_budget:
            decode_elements = min(decode_elements, max(width, self.decode_budget // 4))
        logits_elements = max(prefill_elements, decode_elements)
        extra_per_row = self.topk_tokens * 3  # int64 indices and fp32 scores
        if self.candidate_blocks is not None:
            # Candidate selection also holds block scores and torch.topk output.
            logits_elements += cdiv(logits_elements, self.candidate_block_size)
            extra_per_row = max(extra_per_row, self.candidate_blocks.shape[1] * 3) + 1
        self.profile_elements = logits_elements + max_rows * extra_per_row

    def _score_and_select(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        rows: slice,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        width: int,
        starts: torch.Tensor | None = None,
        key_to_request: torch.Tensor | None = None,
    ) -> None:
        ragged = self.topk_backend != "torch"
        logits = paged_mqa_logits(
            q[rows],
            self.k_cache.kv_cache,
            weights[rows],
            block_table,
            lengths,
            width,
            starts,
            key_to_request,
            mask_tail=not ragged,
        )
        if self.candidate_blocks is not None:
            candidates = self.candidate_blocks[rows]
            if self.candidate_write:
                select_candidate_blocks(
                    logits,
                    None,
                    lengths,
                    candidates.shape[1],
                    self.candidate_block_size,
                    candidates,
                    ragged_topk=ragged,
                )
            else:
                apply_candidate_mask(
                    logits,
                    None,
                    lengths,
                    candidates,
                    self.candidate_block_size,
                    mask_tail=not ragged,
                )
        topk = self.topk_indices_buffer[rows, : self.topk_tokens]
        if self.topk_backend == "torch":
            k = min(self.topk_tokens, width)
            result = logits.topk(k, dim=-1)
            topk[:, :k] = result.indices.masked_fill(result.values == -torch.inf, -1)
        else:
            get_indexer_topk("per_row")(
                logits, lengths, 1, topk, self.topk_tokens, width
            )
            # The selector can return masked columns when there are fewer
            # candidates than topk. Preserve the missing-entry sentinel.
            mask_invalid_topk(logits, lengths, topk)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(q_quant, torch.Tensor) and k is None
        assert self.topk_indices_buffer is not None
        assert not self.use_fp4_cache and self.skip_k_cache_insert
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            torch.empty(
                self.profile_elements, dtype=torch.float32, device=q_quant.device
            )
            return self.topk_indices_buffer

        indexer = metadata[self.k_cache.prefix]
        assert isinstance(indexer, DeepseekV32IndexerMetadata)
        num_decode = indexer.num_decode_tokens
        num_tokens = num_decode + indexer.num_prefill_tokens
        self.topk_indices_buffer[:num_tokens].fill_(-1)
        if self.candidate_write:
            assert self.candidate_blocks is not None
            self.candidate_blocks[:num_tokens].fill_(-1)

        if indexer.prefill is not None:
            for chunk in indexer.prefill.chunks:
                width = chunk.max_seqlen_k
                rows_per_call = dense_logits_rows(width)
                for start in range(chunk.token_start, chunk.token_end, rows_per_call):
                    end = min(start + rows_per_call, chunk.token_end)
                    local = slice(start - chunk.token_start, end - chunk.token_start)
                    starts = chunk.cu_seqlen_ks[local]
                    lengths = chunk.cu_seqlen_ke[local] - starts
                    self._score_and_select(
                        q_quant,
                        weights,
                        slice(start, end),
                        chunk.block_table,
                        lengths,
                        width,
                        starts,
                        chunk.token_to_seq,
                    )

        if (decode := indexer.decode) is not None and num_decode:
            # SM80 expands DSpark verify metadata into one page-table row and
            # one causal length per query. Padding rows carry a zero length.
            assert decode.block_table.shape[0] == decode.seq_lens.numel() == num_decode
            width = indexer.max_seq_len // self.compress_ratio
            if width:
                rows_per_call = (
                    dense_logits_rows(width, self.decode_budget)
                    if self.decode_budget
                    else num_decode
                )
                lengths = decode.seq_lens.view(-1)
                for start in range(0, num_decode, rows_per_call):
                    rows = slice(start, min(start + rows_per_call, num_decode))
                    self._score_and_select(
                        q_quant,
                        weights,
                        rows,
                        decode.block_table[rows],
                        lengths[rows],
                        width,
                    )

        return self.topk_indices_buffer


class DeepseekV41TritonIndexer(DeepseekV4Indexer):
    attn_cls = DeepseekV41TritonSparseAttnIndexer
