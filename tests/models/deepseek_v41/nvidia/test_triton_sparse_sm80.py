# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm import LLM, SamplingParams
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.attention.dsa.triton_mqa_logits import paged_mqa_logits
from vllm.models.deepseek_v4.common.ops import fused_indexer_q_rope_quant
from vllm.models.deepseek_v41.common.ops.cache_utils import (
    V4_BYTES_PER_TOKEN,
    dequantize_and_gather_k_cache_triton,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v41.nvidia.model import _select_dsv4_attn_cls
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(80)),
    reason="requires an SM80 GPU",
)

BLOCK_SIZE = 128
WINDOW = 128
TOPK = 19
CANDIDATE_BLOCK_SIZE = 8
CANDIDATES = 2
HEADS = 24


@pytest.fixture
def model_path(tmp_path):
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "text_config": {
            "vocab_size": 256,
            "hidden_size": 512,
            "num_hidden_layers": 5,
            "num_attention_heads": HEADS,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 128,
            "o_lora_rank": 128,
            "o_groups": 1,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 1536,
            "rope_theta": 10000,
            "compress_rope_theta": 160000,
            "sliding_window": WINDOW,
            "compress_ratios": [0, 2, 1, 1, 1],
            "kv_source_layer_ids": [1, 2],
            "index_source_layer_ids": [1, 2, 3],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": TOPK,
            "candidate_source_layer_id": 2,
            "candidate_topk_blocks": CANDIDATES,
            "candidate_block_size": CANDIDATE_BLOCK_SIZE,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "intermediate_size": 256,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _engine_args(model_path, **kwargs):
    return EngineArgs(
        model=model_path,
        skip_tokenizer_init=True,
        language_model_only=True,
        block_size=BLOCK_SIZE,
        max_model_len=1536,
        max_num_batched_tokens=384,
        max_num_seqs=4,
        **kwargs,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_triton_backend_selects_sm80_eager(model_path, explicit):
    args = _engine_args(
        model_path,
        enforce_eager=True,
        attention_config={"backend": "TRITON_MLA_SPARSE_DSV41"} if explicit else {},
    )
    config = args.create_engine_config()
    backend = AttentionBackendEnum["TRITON_MLA_SPARSE_DSV41"].get_class()
    assert _select_dsv4_attn_cls(config).backend_cls is backend


def test_triton_backend_requires_eager(model_path):
    config = _engine_args(model_path, enforce_eager=False).create_engine_config()
    with pytest.raises(ValueError, match="--enforce-eager"):
        _select_dsv4_attn_cls(config)


@torch.inference_mode()
def test_sm80_indexer_query_quantization():
    torch.manual_seed(43)
    q = torch.randn((9, 32, 128), dtype=torch.bfloat16, device="cuda")[1:]
    w = torch.randn(q.shape[:2], dtype=torch.bfloat16, device=q.device)
    positions = torch.arange(q.shape[0], device=q.device)
    # Quarter-turns exercise GPT-J interleaving without rounding ambiguity
    # from different FMA contraction choices in the two implementations.
    quarter = (positions[:, None] + torch.arange(32, device=q.device)) % 4
    cos = (quarter == 0).float() - (quarter == 2).float()
    sin = (quarter == 1).float() - (quarter == 3).float()
    q_fp8, weights = fused_indexer_q_rope_quant(
        positions, q, torch.cat((cos, sin), dim=-1), w, 0.125, 0.25
    )

    x, y = q[..., 64::2].float(), q[..., 65::2].float()
    cos, sin = cos[:, None], sin[:, None]
    even, odd = x * cos - y * sin, y * cos + x * sin
    rope = torch.stack((even, odd), dim=-1).flatten(-2).bfloat16()
    rotated = torch.cat((q[..., :64], rope), dim=-1).float()
    scale = (rotated.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
    expected = (rotated / scale[..., None]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(q_fp8.float(), expected.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        weights, w.float() * scale * 0.125 * 0.25, atol=0, rtol=0
    )


def _paged_cache(pages, size, width, device):
    return torch.empty((pages, size * width + 256), dtype=torch.uint8, device=device)[
        :, : size * width
    ].view(pages, size, width)


def _indexer_cache(table, page_size, device):
    pages = table.numel()
    cache = _paged_cache(pages, page_size, 132, device)
    raw = torch.randn((pages, page_size, 128), device=device).to(torch.float8_e4m3fn)
    scales = torch.rand((pages, page_size), device=device) * 0.25 + 0.5
    flat = cache.flatten(1)
    flat[:, : page_size * 128].copy_(raw.view(torch.uint8).flatten(1))
    flat[:, page_size * 128 :].view(torch.float32).copy_(scales)
    return cache, raw.float()[table].flatten(1, 2), scales[table].flatten(1)


@pytest.mark.parametrize("prefill", [False, True])
@torch.inference_mode()
def test_sm80_paged_indexer_logits_match_reference(prefill):
    torch.manual_seed(19)
    device = torch.device("cuda")
    table = torch.tensor([[3, 0], [1, 2]], device=device, dtype=torch.int32)
    cache, keys, scales = _indexer_cache(table, 64, device)
    req = torch.tensor([1, 0, 0], device=device)
    lengths = torch.tensor([65, 0, 71], device=device, dtype=torch.int32)
    q = torch.randn((4, 32, 128), device=device).to(torch.float8_e4m3fn)[1:]
    w = torch.randn(q.shape[:2], device=device)
    width = 71
    if prefill:
        starts = torch.tensor([71, 0, 0], device=device, dtype=torch.int32)
        key_to_request = torch.repeat_interleave(
            table.new_tensor([0, 1]), table.new_tensor([71, 65])
        )
        actual = paged_mqa_logits(
            q, cache, w, table, lengths, width, starts, key_to_request
        )
    else:
        actual = paged_mqa_logits(q, cache, w, table[req], lengths, width)
    dots = torch.einsum("thd,tkd->thk", q.float(), keys[req, :width]).relu()
    expected = (dots * w[:, :, None]).sum(1) * scales[req, :width]
    expected.masked_fill_(
        torch.arange(width, device=device) >= lengths[:, None], -torch.inf
    )
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-5)


def _attention_cache(table, page_size, lengths):
    pages = table.numel()
    device = table.device
    cache = _paged_cache(pages, page_size, V4_BYTES_PER_TOKEN, device)
    values = torch.randn((pages * page_size, 512), device=device, dtype=torch.bfloat16)
    slots = torch.arange(values.shape[0], device=device)
    quantize_and_insert_k_cache(values, cache.flatten(1), slots, page_size)
    gathered = torch.zeros(
        (table.shape[0], table.shape[1] * page_size, 512),
        device=device,
        dtype=torch.bfloat16,
    )
    dequantize_and_gather_k_cache_triton(
        gathered, cache, lengths, None, table.contiguous(), page_size, offset=0
    )
    return cache, gathered.float()


def _logical_topk(logits):
    scores, indices = logits.topk(TOPK)
    return indices.masked_fill(scores == -torch.inf, -1)


@pytest.mark.parametrize(
    ("ratio", "topk_backend"), [(1, "auto"), (2, "per_row"), (1, "torch")]
)
@torch.inference_mode()
def test_mixed_sparse_mla_uses_request_local_candidates(
    model_path, tmp_path, monkeypatch, ratio, topk_backend
):
    # Force a prefill request to span two indexer chunks (ratio 1) and mix
    # requests with different causal boundaries and physical page numbers.
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    config = _engine_args(
        model_path,
        enforce_eager=True,
        attention_config={"backend": "TRITON_MLA_SPARSE_DSV41"},
        kernel_config={"sparse_indexer_topk_backend": topk_backend},
    ).create_engine_config()
    device = torch.device("cuda:0")
    torch.manual_seed(71)
    spec = BatchSpec(seq_lens=[1, 267, 913, 161], query_lens=[1, 1, 290, 5])
    common = create_common_attn_metadata(
        spec, BLOCK_SIZE, device, arange_block_indices=True
    )
    table = common.block_table_tensor
    # Keep logical neighboring pages physically separated, and pad table rows.
    table = torch.empty((4, table.shape[1] + 3), device=device, dtype=torch.int32)[
        :, :-3
    ]
    table.copy_(common.block_table_tensor.flatten().flip(0).view_as(table))
    common.block_table_tensor = table
    swa_common = create_common_attn_metadata(
        spec, 32, device, arange_block_indices=True
    )
    positions = torch.cat(
        [
            torch.arange(s - q, s, device=device)
            for s, q in zip(spec.seq_lens, spec.query_lens)
        ]
    )
    common.positions = swa_common.positions = positions
    num_tokens = positions.numel()
    topk = torch.empty((num_tokens, TOPK), device=device, dtype=torch.int32)
    candidates = torch.empty((num_tokens, CANDIDATES), device=device, dtype=torch.int32)

    try:
        with set_current_vllm_config(config), torch.device(device):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{tmp_path / 'dist'}",
                backend="nccl",
            )
            ensure_model_parallel_initialized(1, 1)
            cls = _select_dsv4_attn_cls(config)
            source = cls(
                config,
                f"model.layers.{1 if ratio == 2 else 2}.attn",
                topk,
                candidate_block_buffer=candidates,
            )
            if ratio == 1:
                scorer = cls(
                    config,
                    "model.layers.3.attn",
                    topk,
                    candidate_block_buffer=candidates,
                )
                layer = cls(config, "model.layers.4.attn", topk)
            else:
                scorer = layer = source
            assert source.indexer is not None and scorer.indexer is not None
            index_cache = source.indexer.k_cache
            index_cache_tensor, keys, key_scales = _indexer_cache(
                table, BLOCK_SIZE // ratio, device
            )
            index_cache.bind_kv_cache(index_cache_tensor.unsqueeze(1))
            index_meta = (
                index_cache.get_attn_backend()
                .get_builder_cls()(
                    index_cache.get_kv_cache_spec(config),
                    [index_cache.prefix],
                    config,
                    device,
                    block_table_width=table.shape[1],
                )
                .build(0, common)
            )
            if ratio == 1:
                assert len(index_meta.prefill.chunks) >= 2
            assert (index_meta.num_decodes, index_meta.num_prefills) == (2, 2)

            swa_layer = layer.swa_cache_layer
            swa_meta = (
                swa_layer.get_attn_backend()
                .get_builder_cls()(
                    swa_layer.get_kv_cache_spec(config),
                    [swa_layer.prefix],
                    config,
                    device,
                )
                .build(0, swa_common)
            )
            compressed_meta = (
                source.get_attn_backend()
                .get_builder_cls()(
                    source.get_kv_cache_spec(config), [source.prefix], config, device
                )
                .build(0, common)
            )
            swa_cache, swa_kv = _attention_cache(
                swa_common.block_table_tensor, 32, common.seq_lens
            )
            comp_cache, comp_kv = _attention_cache(
                table, BLOCK_SIZE // ratio, common.seq_lens // ratio
            )
            swa_layer.bind_kv_cache(swa_cache.unsqueeze(1))
            source.bind_kv_cache(comp_cache.unsqueeze(1))

            q = torch.randn(
                (num_tokens, layer.padded_heads, 512),
                device=device,
                dtype=torch.bfloat16,
            )
            out = torch.empty_like(q)
            sink = layer.attn_sink[:HEADS]
            sink.copy_(torch.linspace(-1, 1, HEADS, device=device))
            layer.attn_sink[0] = -torch.inf
            index_q = torch.randn((num_tokens, 32, 128), device=device).to(
                torch.float8_e4m3fn
            )
            index_w = torch.randn((num_tokens, 32), device=device) * 0.125

            with set_forward_context(
                {
                    index_cache.prefix: index_meta,
                    swa_layer.prefix: swa_meta,
                    source.prefix: compressed_meta,
                },
                config,
                num_tokens=num_tokens,
            ):
                if ratio == 1:
                    source.indexer.indexer_op(q, index_q, None, index_w)
                scorer.indexer.indexer_op(
                    q, index_q, None, -index_w if ratio == 1 else index_w
                )
                layer.forward_mqa(q, q, positions, out)

            assert torch.count_nonzero(out[:, HEADS:]) == 0
            start = 0
            for req, (seq_len, q_len) in enumerate(zip(spec.seq_lens, spec.query_lens)):
                rows = slice(start, start + q_len)
                width = max(TOPK, seq_len // ratio)
                cols = torch.arange(width, device=device)
                lengths = (positions[rows] + 1) // ratio
                scores = (
                    torch.einsum(
                        "thd,kd->thk", index_q[rows].float(), keys[req, :width]
                    ).relu()
                    * index_w[rows, :, None]
                ).sum(1) * key_scales[req, :width]
                visible = cols[None, :] < lengths[:, None]
                if ratio == 1:
                    block_scores = scores.masked_fill(~visible, -torch.inf)
                    padding = (-width) % CANDIDATE_BLOCK_SIZE
                    block_scores = torch.nn.functional.pad(
                        block_scores, (0, padding), value=-torch.inf
                    )
                    block_scores = block_scores.view(
                        q_len, -1, CANDIDATE_BLOCK_SIZE
                    ).amax(-1)
                    block_scores.scatter_(
                        1, ((lengths - 1) // CANDIDATE_BLOCK_SIZE)[:, None], torch.inf
                    )
                    picked = block_scores.topk(min(CANDIDATES, block_scores.shape[1]))
                    expected_blocks = picked.indices.masked_fill(
                        picked.values == -torch.inf, -1
                    )
                    torch.testing.assert_close(
                        candidates[rows].sort(dim=-1).values,
                        expected_blocks.sort(dim=-1).values.to(torch.int32),
                    )
                    visible &= (
                        cols[None, :, None] // CANDIDATE_BLOCK_SIZE
                        == expected_blocks[:, None, :]
                    ).any(-1)
                    scores = -scores
                expected_topk = _logical_topk(scores.masked_fill(~visible, -torch.inf))
                torch.testing.assert_close(
                    topk[rows].sort(dim=-1).values,
                    expected_topk.sort(dim=-1).values.to(torch.int32),
                )

                pos = positions[rows]
                window_positions = pos[:, None] - torch.arange(WINDOW, device=device)
                swa_values = swa_kv[req, window_positions.clamp_min(0)]
                selected_values = comp_kv[req, expected_topk.clamp_min(0)]
                kv = torch.cat((swa_values, selected_values), dim=1)
                valid = torch.cat((window_positions >= 0, expected_topk >= 0), dim=1)
                logits = (
                    torch.einsum("thd,tkd->thk", q[rows, :HEADS].float(), kv)
                    * layer.scale
                )
                logits.masked_fill_(~valid[:, None], -torch.inf)
                sink_logits = sink[None, :, None].expand(q_len, -1, -1)
                probs = torch.cat((logits, sink_logits), dim=-1).softmax(-1)[..., :-1]
                expected_out = torch.einsum("thk,tkd->thd", probs, kv)
                torch.testing.assert_close(
                    out[rows, :HEADS].float(), expected_out, atol=0.012, rtol=0.015
                )
                start += q_len
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(not has_tilelang(), reason="DeepSeek V4.1 uses TileLang mHC")
def test_sm80_llm_eager_chunked_prefill(model_path):
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        language_model_only=True,
        skip_tokenizer_init=True,
        load_format="dummy",
        max_model_len=1024,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        block_size=BLOCK_SIZE,
        kv_cache_memory_bytes=256 << 20,
        attention_config={"backend": "TRITON_MLA_SPARSE_DSV41"},
    )
    lengths = [7, 180, 523]
    prompts = [
        {"prompt_token_ids": [1 + i % 200 for i in range(length)]} for length in lengths
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        detokenize=False,
        logprobs=1,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    for output, length in zip(outputs, lengths, strict=True):
        assert output.prompt_token_ids is not None
        assert len(output.prompt_token_ids) == length
        completion = output.outputs[0]
        assert len(completion.token_ids) == 4
        assert completion.cumulative_logprob is not None
        assert math.isfinite(completion.cumulative_logprob)
