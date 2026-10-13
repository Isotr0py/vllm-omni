# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Anima DiT tensor-parallel and fused-projection unit tests (CPU-only).

The TP world size/rank and the row-parallel all-reduce are mocked so a single
process can simulate each rank's shard in isolation (same approach as the
flux2-klein local-TP tests).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.utils import default_unquantized_gemm

from vllm_omni.diffusion.attention import selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.anima import anima_transformer
from vllm_omni.diffusion.models.anima.anima_transformer import (
    AnimaTransformer3DModel,
    CosmosAttention,
    validate_anima_tp_constraints,
)
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _cpu_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *_args, **_kwargs: SDPABackend)


def _mock_tp(monkeypatch: pytest.MonkeyPatch, tp: int, rank: int) -> None:
    """Mock a TP world of ``tp`` at ``rank`` with an identity all-reduce, so a
    single process can build and run one rank's shard in isolation."""
    monkeypatch.setattr(anima_transformer, "get_tensor_model_parallel_world_size", lambda: tp)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda tp=tp: tp)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda *_args, **_kwargs: default_unquantized_gemm)
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda tensor: tensor)


def _tiny_transformer_config() -> dict:
    return {
        "in_channels": 1,
        "out_channels": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "num_layers": 1,
        "mlp_ratio": 1.0,
        "text_embed_dim": 4,
        "adaln_lora_dim": 3,
        "max_size": (1, 2, 2),
        "patch_size": (1, 1, 1),
        "rope_scale": (1.0, 1.0, 1.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    }


def _build_transformer(monkeypatch: pytest.MonkeyPatch, tp: int = 1, rank: int = 0) -> AnimaTransformer3DModel:
    _mock_tp(monkeypatch, tp, rank)
    return AnimaTransformer3DModel(**_tiny_transformer_config())


def _tiny_forward_inputs() -> dict:
    return {
        "hidden_states": torch.randn(1, 1, 1, 2, 2),
        "timestep": torch.tensor([0.5]),
        "encoder_hidden_states": torch.randn(1, 3, 4),
        "padding_mask": torch.zeros(1, 1, 2, 2),
    }


# ── TP constraint validation ──


def test_validate_anima_tp_constraints_reference_config() -> None:
    supported = validate_anima_tp_constraints(
        num_attention_heads=16,
        hidden_size=2048,
        ffn_inner_dim=8192,
        tensor_parallel_size=16,
    )
    assert supported == [1, 2, 4, 8, 16]


def test_validate_anima_tp_constraints_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValueError, match=r"num_attention_heads=16.*tp=3"):
        validate_anima_tp_constraints(
            num_attention_heads=16,
            hidden_size=2048,
            ffn_inner_dim=8192,
            tensor_parallel_size=3,
        )


def test_validate_anima_tp_constraints_rejects_non_divisible_ffn() -> None:
    with pytest.raises(ValueError, match=r"ffn_inner_dim=30.*tp=4"):
        validate_anima_tp_constraints(
            num_attention_heads=8,
            hidden_size=96,
            ffn_inner_dim=30,
            tensor_parallel_size=4,
        )


def test_validate_anima_tp_constraints_rejects_non_positive_tp() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size must be > 0"):
        validate_anima_tp_constraints(
            num_attention_heads=16,
            hidden_size=2048,
            ffn_inner_dim=8192,
            tensor_parallel_size=0,
        )


def test_model_init_rejects_non_divisible_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_tp(monkeypatch, 3, 0)
    with pytest.raises(ValueError, match=r"num_attention_heads=2.*tp=3"):
        AnimaTransformer3DModel(**_tiny_transformer_config())


# ── Layer selection and tp=1 behavior ──


def test_fused_layer_selection_tp1(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _build_transformer(monkeypatch)
    block = transformer.transformer_blocks[0]

    assert isinstance(block.attn1.to_qkv, QKVParallelLinear)
    assert not hasattr(block.attn1, "to_q")
    assert isinstance(block.attn2.to_q, ColumnParallelLinear)
    assert type(block.attn2.to_q) is ColumnParallelLinear
    assert isinstance(block.attn2.to_kv, MergedColumnParallelLinear)
    for attn in (block.attn1, block.attn2):
        assert isinstance(attn.to_out[0], RowParallelLinear)
    assert isinstance(block.ff.net[0].proj, ColumnParallelLinear)
    assert type(block.ff.net[0].proj) is ColumnParallelLinear
    assert isinstance(block.ff.net[2], RowParallelLinear)

    # The LoRA manager discovers packed->sublayer mappings from the model.
    assert transformer.stacked_params_mapping is anima_transformer._ANIMA_STACKED_PARAMS_MAPPING


def test_tiny_anima_transformer_forward_tp1(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch).eval()
    inputs = _tiny_forward_inputs()

    with torch.inference_mode():
        first = transformer(**inputs, return_dict=False)[0]
        second = transformer(**inputs, return_dict=False)[0]
    assert first.shape == (1, 1, 1, 2, 2)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


# ── Fused weight loading ──


def _separate_projection_checkpoint(config: dict) -> dict[str, torch.Tensor]:
    """Build a checkpoint dict with separate to_q/to_k/to_v weights, as the
    Anima checkpoint (and its converted form) stores them."""
    torch.manual_seed(0)
    heads = config["num_attention_heads"]
    hidden = heads * config["attention_head_dim"]
    text_dim = config["text_embed_dim"]
    return {
        "transformer_blocks.0.attn1.to_q.weight": torch.randn(hidden, hidden),
        "transformer_blocks.0.attn1.to_k.weight": torch.randn(hidden, hidden),
        "transformer_blocks.0.attn1.to_v.weight": torch.randn(hidden, hidden),
        "transformer_blocks.0.attn2.to_q.weight": torch.randn(hidden, hidden),
        "transformer_blocks.0.attn2.to_k.weight": torch.randn(hidden, text_dim),
        "transformer_blocks.0.attn2.to_v.weight": torch.randn(hidden, text_dim),
    }


def test_load_weights_packs_qkv_and_kv_in_order_tp1(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_transformer_config()
    transformer = _build_transformer(monkeypatch)
    checkpoint = _separate_projection_checkpoint(config)
    block = transformer.transformer_blocks[0]

    loaded = transformer.load_weights(checkpoint.items())

    assert "transformer_blocks.0.attn1.to_qkv.weight" in loaded
    assert "transformer_blocks.0.attn2.to_kv.weight" in loaded
    torch.testing.assert_close(
        block.attn1.to_qkv.weight,
        torch.cat(
            [
                checkpoint["transformer_blocks.0.attn1.to_q.weight"],
                checkpoint["transformer_blocks.0.attn1.to_k.weight"],
                checkpoint["transformer_blocks.0.attn1.to_v.weight"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        block.attn2.to_kv.weight,
        torch.cat(
            [
                checkpoint["transformer_blocks.0.attn2.to_k.weight"],
                checkpoint["transformer_blocks.0.attn2.to_v.weight"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(block.attn2.to_q.weight, checkpoint["transformer_blocks.0.attn2.to_q.weight"])


def test_load_weights_shards_fused_projections_tp2(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_transformer_config()
    checkpoint = _separate_projection_checkpoint(config)
    qkv_parts = ["to_q", "to_k", "to_v"]

    for rank in (0, 1):
        transformer = _build_transformer(monkeypatch, tp=2, rank=rank)
        block = transformer.transformer_blocks[0]
        hidden = config["num_attention_heads"] * config["attention_head_dim"]
        assert block.attn1.to_qkv.weight.shape == (3 * hidden // 2, hidden)
        assert block.attn2.to_kv.weight.shape == (hidden, config["text_embed_dim"])
        assert block.attn1.to_out[0].weight.shape == (hidden, hidden // 2)

        loaded = transformer.load_weights(checkpoint.items())

        assert "transformer_blocks.0.attn1.to_qkv.weight" in loaded
        expected_qkv = torch.cat(
            [checkpoint[f"transformer_blocks.0.attn1.{part}.weight"].chunk(2, dim=0)[rank] for part in qkv_parts],
            dim=0,
        )
        torch.testing.assert_close(block.attn1.to_qkv.weight, expected_qkv)
        expected_kv = torch.cat(
            [checkpoint[f"transformer_blocks.0.attn2.{part}.weight"].chunk(2, dim=0)[rank] for part in ("to_k", "to_v")],
            dim=0,
        )
        torch.testing.assert_close(block.attn2.to_kv.weight, expected_kv)
        torch.testing.assert_close(
            block.attn2.to_q.weight, checkpoint["transformer_blocks.0.attn2.to_q.weight"].chunk(2, dim=0)[rank]
        )
        # Replicated layers keep their full (unsharded) shape on every rank.
        assert block.norm1.linear_1.weight.shape == (config["adaln_lora_dim"], hidden)
        assert transformer.proj_out.weight.shape == (config["out_channels"], hidden)


def test_load_weights_full_checkpoint_no_missing_tp2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full (tp=1-style) checkpoint loads under tp=2 with every parameter covered."""
    _mock_tp(monkeypatch, 1, 0)
    dense = AnimaTransformer3DModel(**_tiny_transformer_config())
    full_state = {name: tensor.detach().clone() for name, tensor in dense.state_dict().items()}
    # Expand the fused entries back into checkpoint-style separate sublayers.
    checkpoint: dict[str, torch.Tensor] = {}
    for name, tensor in full_state.items():
        if ".attn1.to_qkv.weight" in name:
            q, k, v = tensor.chunk(3, dim=0)
            checkpoint[name.replace(".attn1.to_qkv.", ".attn1.to_q.")] = q
            checkpoint[name.replace(".attn1.to_qkv.", ".attn1.to_k.")] = k
            checkpoint[name.replace(".attn1.to_qkv.", ".attn1.to_v.")] = v
        elif ".attn2.to_kv.weight" in name:
            k, v = tensor.chunk(2, dim=0)
            checkpoint[name.replace(".attn2.to_kv.", ".attn2.to_k.")] = k
            checkpoint[name.replace(".attn2.to_kv.", ".attn2.to_v.")] = v
        else:
            checkpoint[name] = tensor

    for rank in (0, 1):
        transformer = _build_transformer(monkeypatch, tp=2, rank=rank)
        loaded = transformer.load_weights(checkpoint.items())
        assert loaded == set(dict(transformer.named_parameters()))


# ── TP2 vs TP1 numeric equivalence (single attention module) ──


def _fill_attention(attn: CosmosAttention) -> dict[str, torch.Tensor]:
    """Deterministically fill an attention module; returns its dense weights."""
    torch.manual_seed(0)
    weights: dict[str, torch.Tensor] = {}
    for name, param in sorted(attn.named_parameters()):
        param.data.normal_()
        weights[name] = param.detach().clone()
    return weights


def _load_attention_shard(attn: CosmosAttention, dense_weights: dict[str, torch.Tensor]) -> None:
    """Load dense per-sublayer weights into a (possibly sharded) module via the
    same weight_loader protocol load_weights() uses."""
    for name, tensor in dense_weights.items():
        if name == "to_qkv.weight":
            q, k, v = tensor.chunk(3, dim=0)
            param = attn.to_qkv.weight
            for shard_id, part in zip(("q", "k", "v"), (q, k, v)):
                param.weight_loader(param, part, shard_id)
        elif name == "to_kv.weight":
            k, v = tensor.chunk(2, dim=0)
            param = attn.to_kv.weight
            for shard_id, part in ((0, k), (1, v)):
                param.weight_loader(param, part, shard_id)
        elif name == "to_q.weight":
            attn.to_q.weight.weight_loader(attn.to_q.weight, tensor)
        elif name == "to_out.0.weight":
            attn.to_out[0].weight.weight_loader(attn.to_out[0].weight, tensor)
        else:
            dict(attn.named_parameters())[name].data.copy_(tensor)


@pytest.mark.parametrize("self_attention", [True, False])
def test_attention_tp2_partial_sums_match_dense(monkeypatch: pytest.MonkeyPatch, self_attention: bool) -> None:
    """Per-head sharding is exact: summing both TP2 ranks' row-parallel partial
    outputs (identity all-reduce) reproduces the dense TP1 attention output."""
    heads, dim_head = 4, 8
    query_dim = heads * dim_head
    cross_dim = None if self_attention else 16
    kwargs = dict(query_dim=query_dim, cross_attention_dim=cross_dim, heads=heads, dim_head=dim_head)

    _mock_tp(monkeypatch, 1, 0)
    dense = CosmosAttention(**kwargs).eval()
    dense_weights = _fill_attention(dense)

    torch.manual_seed(1)
    hidden = torch.randn(2, 5, query_dim)
    context = None if self_attention else torch.randn(2, 7, cross_dim)
    rotary = (torch.randn(5, dim_head), torch.randn(5, dim_head)) if self_attention else None

    with torch.inference_mode():
        expected = dense(hidden, encoder_hidden_states=context, image_rotary_emb=rotary)

    partials = []
    for rank in (0, 1):
        _mock_tp(monkeypatch, 2, rank)
        shard = CosmosAttention(**kwargs).eval()
        _load_attention_shard(shard, dense_weights)
        assert shard.heads_local == heads // 2
        with torch.inference_mode():
            partials.append(shard(hidden, encoder_hidden_states=context, image_rotary_emb=rotary))

    torch.testing.assert_close(partials[0] + partials[1], expected, rtol=1e-5, atol=1e-5)


# ── LoRA on fused layers ──


def _lora_weights(name: str, lora_a: torch.Tensor, lora_b: torch.Tensor) -> LoRALayerWeights:
    rank = lora_a.shape[0]
    assert lora_b.shape[1] == rank
    return LoRALayerWeights(name, rank, rank, lora_a, lora_b).optimize()


def _lora_request(adapter_id: int) -> LoRARequest:
    return LoRARequest(lora_name=f"a{adapter_id}", lora_int_id=adapter_id, lora_path=f"/tmp/a{adapter_id}")


def _build_manager(monkeypatch: pytest.MonkeyPatch, transformer: AnimaTransformer3DModel) -> DiffusionLoRAManager:
    pipeline = nn.Module()
    pipeline.transformer = transformer
    return DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=4,
    )


def _fake_adapter_loader(manager: DiffusionLoRAManager, models: dict[int, LoRAModel]):
    def _fake_load(request: LoRARequest):
        model = models[request.lora_int_id]
        peft_helper = PEFTHelper.from_dict(
            {
                "r": model.rank,
                "lora_alpha": model.rank,
                "target_modules": sorted({name.split(".")[-1] for name in model.loras}),
            }
        )
        return model, peft_helper

    return _fake_load


def test_single_lora_binds_to_fused_layers_via_sublayers(monkeypatch: pytest.MonkeyPatch) -> None:
    """kohya-style per-sublayer adapters bind to the fused to_qkv/to_kv layers
    through the manager's packed mapping derived from stacked_params_mapping."""
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch)
    manager = _build_manager(monkeypatch, transformer)
    hidden = 24  # num_attention_heads * attention_head_dim

    rank, scale = 2, 0.5
    prefix = "transformer.transformer_blocks.0"
    loras = {}
    for suffix, out_dim, in_dim in (
        ("attn1.to_q", hidden, hidden),
        ("attn1.to_k", hidden, hidden),
        ("attn1.to_v", hidden, hidden),
        ("attn2.to_q", hidden, hidden),
        ("attn2.to_k", hidden, 4),
        ("attn2.to_v", hidden, 4),
    ):
        name = f"{prefix}.{suffix}"
        loras[name] = _lora_weights(name, torch.randn(rank, in_dim), torch.randn(out_dim, rank))
    models = {1: LoRAModel(1, rank=rank, loras=loras)}
    monkeypatch.setattr(manager, "_load_adapter", _fake_adapter_loader(manager, models))

    manager.set_active_adapters([_lora_request(1)], [scale])

    block = transformer.transformer_blocks[0]
    from vllm_omni.diffusion.lora.layers import (
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionMergedQKVParallelLinearWithLoRA,
    )

    assert isinstance(block.attn1.to_qkv, DiffusionMergedQKVParallelLinearWithLoRA)
    assert isinstance(block.attn2.to_kv, DiffusionMergedColumnParallelLinearWithLoRA)
    assert block.attn1.to_qkv.n_slices == 3
    assert block.attn2.to_kv.n_slices == 2

    x = torch.randn(3, hidden)
    with torch.inference_mode():
        base_qkv = F.linear(x, block.attn1.to_qkv.base_layer.weight)
        delta = torch.cat(
            [scale * (x @ loras[f"{prefix}.attn1.to_{part}"].lora_a.t() @ loras[f"{prefix}.attn1.to_{part}"].lora_b.t()) for part in ("q", "k", "v")],
            dim=-1,
        )
        torch.testing.assert_close(block.attn1.to_qkv(x), base_qkv + delta)

        ctx = torch.randn(3, 4)
        base_kv = F.linear(ctx, block.attn2.to_kv.base_layer.weight)
        delta_kv = torch.cat(
            [scale * (ctx @ loras[f"{prefix}.attn2.to_{part}"].lora_a.t() @ loras[f"{prefix}.attn2.to_{part}"].lora_b.t()) for part in ("k", "v")],
            dim=-1,
        )
        torch.testing.assert_close(block.attn2.to_kv(ctx), base_kv + delta_kv)

        manager.set_active_adapters(None)
        torch.testing.assert_close(block.attn1.to_qkv(x), base_qkv)


def test_composed_loras_on_fused_qkv_match_manual_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two kohya-style adapters compose (rank-sum) and bind to the fused QKV
    layer; the wrapped output equals the manual two-adapter sum."""
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch)
    manager = _build_manager(monkeypatch, transformer)
    hidden = 24
    name = "transformer.transformer_blocks.0.attn1.to_q"
    r1, r2, s1, s2 = 2, 3, 0.5, 2.0
    a1, b1 = torch.randn(r1, hidden), torch.randn(hidden, r1)
    a2, b2 = torch.randn(r2, hidden), torch.randn(hidden, r2)
    models = {
        1: LoRAModel(1, rank=r1, loras={name: _lora_weights(name, a1, b1)}),
        2: LoRAModel(2, rank=r2, loras={name: _lora_weights(name, a2, b2)}),
    }
    monkeypatch.setattr(manager, "_load_adapter", _fake_adapter_loader(manager, models))

    manager.set_active_adapters([_lora_request(1), _lora_request(2)], [s1, s2])

    wrapped = transformer.transformer_blocks[0].attn1.to_qkv
    x = torch.randn(3, hidden)
    with torch.inference_mode():
        base_qkv = F.linear(x, wrapped.base_layer.weight)
        delta_q = s1 * (x @ a1.t()) @ b1.t() + s2 * (x @ a2.t()) @ b2.t()
        expected = base_qkv + torch.cat([delta_q, torch.zeros_like(delta_q), torch.zeros_like(delta_q)], dim=-1)
        torch.testing.assert_close(wrapped(x), expected)

        manager.set_active_adapters(None)
        torch.testing.assert_close(wrapped(x), base_qkv)


# ── PackedLoRALayerWeights composition ──


def _packed_weights(name: str, rank: int, in_dim: int, out_dims: tuple[int, ...]) -> PackedLoRALayerWeights:
    return PackedLoRALayerWeights(
        module_name=name,
        rank=rank,
        lora_alphas=[rank] * len(out_dims),
        lora_a=[torch.randn(rank, in_dim) for _ in out_dims],
        lora_b=[torch.randn(out_dim, rank) for out_dim in out_dims],
        scaling=[1.0] * len(out_dims),
    )


def _bare_manager() -> DiffusionLoRAManager:
    return object.__new__(DiffusionLoRAManager)


def test_compose_packed_lora_weights_matches_manual_sum() -> None:
    torch.manual_seed(0)
    name = "transformer.transformer_blocks.0.attn1.to_qkv"
    w1 = _packed_weights(name, rank=2, in_dim=8, out_dims=(4, 4, 4))
    w2 = _packed_weights(name, rank=3, in_dim=8, out_dims=(4, 4, 4))
    m1 = LoRAModel(1, rank=2, loras={name: w1})
    m2 = LoRAModel(2, rank=3, loras={name: w2})

    composed, peft_helper = _bare_manager()._compose_lora_models(99, [(m1, 0.5), (m2, 2.0)])

    packed = composed.loras[name]
    assert isinstance(packed, PackedLoRALayerWeights)
    assert packed.rank == 5
    assert composed.rank == 5
    assert peft_helper.r == 5
    for shard_idx in range(3):
        torch.testing.assert_close(packed.lora_a[shard_idx], torch.cat([w1.lora_a[shard_idx], w2.lora_a[shard_idx]], dim=0))
        torch.testing.assert_close(
            packed.lora_b[shard_idx],
            torch.cat([w1.lora_b[shard_idx] * 0.5, w2.lora_b[shard_idx] * 2.0], dim=1),
        )
    # Per-shard semantics: B_cat @ A_cat == sum(scale_i * B_i @ A_i).
    x = torch.randn(3, 8)
    for shard_idx in range(3):
        manual = 0.5 * (x @ w1.lora_a[shard_idx].t()) @ w1.lora_b[shard_idx].t() + 2.0 * (
            x @ w2.lora_a[shard_idx].t()
        ) @ w2.lora_b[shard_idx].t()
        torch.testing.assert_close((x @ packed.lora_a[shard_idx].t()) @ packed.lora_b[shard_idx].t(), manual)


def test_compose_packed_lora_weights_requires_same_shard_coverage() -> None:
    torch.manual_seed(0)
    name = "transformer.transformer_blocks.0.attn1.to_qkv"
    w1 = _packed_weights(name, rank=2, in_dim=8, out_dims=(4, 4, 4))
    w2 = _packed_weights(name, rank=2, in_dim=8, out_dims=(4, 4, 4))
    w2.lora_a[1] = None
    w2.lora_b[1] = None
    m1 = LoRAModel(1, rank=2, loras={name: w1})
    m2 = LoRAModel(2, rank=2, loras={name: w2})

    with pytest.raises(ValueError, match="same set of packed shards"):
        _bare_manager()._compose_lora_models(99, [(m1, 1.0), (m2, 1.0)])


def test_compose_rejects_mixed_packed_and_plain_layouts() -> None:
    torch.manual_seed(0)
    name = "transformer.transformer_blocks.0.attn1.to_qkv"
    m1 = LoRAModel(1, rank=2, loras={name: _packed_weights(name, rank=2, in_dim=8, out_dims=(4, 4, 4))})
    m2 = LoRAModel(2, rank=2, loras={name: _lora_weights(name, torch.randn(2, 8), torch.randn(12, 2))})

    with pytest.raises(ValueError, match="packed vs non-packed"):
        _bare_manager()._compose_lora_models(99, [(m1, 1.0), (m2, 1.0)])


# ── Pipeline feature gate ──


def test_pipeline_no_longer_rejects_tensor_parallel() -> None:
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    config = OmniDiffusionConfig(
        model="anima.safetensors",
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
    )
    AnimaPipeline(od_config=config, device=torch.device("cpu"))
