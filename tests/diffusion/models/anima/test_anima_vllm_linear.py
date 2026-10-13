# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Anima DiT/text-conditioner use vLLM linear layers.

Diffusion LoRA replacement (``from_layer_diffusion``) only matches vLLM
``LinearBase`` subclasses, so every Anima projection must be a vLLM linear
(``ReplicatedLinear`` for replicated layers, ``ColumnParallelLinear``/
``RowParallelLinear``/``QKVParallelLinear``/``MergedColumnParallelLinear``
for the TP-sharded ones) rather than ``nn.Linear``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention import selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.anima.anima_text_conditioner import AnimaTextConditioner
from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_transformer_config() -> dict:
    return {
        "in_channels": 1,
        "out_channels": 1,
        "num_attention_heads": 1,
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


def _tiny_conditioner_config() -> dict:
    return {
        "source_dim": 4,
        "target_dim": 6,
        "model_dim": 4,
        "num_layers": 1,
        "num_attention_heads": 1,
        "mlp_ratio": 1.0,
        "target_vocab_size": 8,
        "min_sequence_length": 4,
    }


def _build_transformer(monkeypatch: pytest.MonkeyPatch) -> AnimaTransformer3DModel:
    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *_args, **_kwargs: SDPABackend)
    return AnimaTransformer3DModel(**_tiny_transformer_config())


def _build_conditioner(monkeypatch: pytest.MonkeyPatch) -> AnimaTextConditioner:
    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *_args, **_kwargs: SDPABackend)
    return AnimaTextConditioner(**_tiny_conditioner_config())


def _assert_replicated_linear(module: nn.Module) -> None:
    assert isinstance(module, ReplicatedLinear)
    assert isinstance(module, LinearBase)


def test_anima_transformer_uses_vllm_linear_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _build_transformer(monkeypatch)
    block = transformer.transformer_blocks[0]

    # attn1 (self-attn) fuses q/k/v; attn2 (cross-attn) fuses k/v only.
    assert isinstance(block.attn1.to_qkv, QKVParallelLinear)
    assert isinstance(block.attn2.to_q, ColumnParallelLinear)
    assert isinstance(block.attn2.to_kv, MergedColumnParallelLinear)
    for attn in (block.attn1, block.attn2):
        assert isinstance(attn.to_out[0], RowParallelLinear)
    assert isinstance(block.ff.net[0].proj, ColumnParallelLinear)
    assert isinstance(block.ff.net[2], RowParallelLinear)
    for norm in (block.norm1, block.norm2, block.norm3):
        _assert_replicated_linear(norm.linear_1)
        _assert_replicated_linear(norm.linear_2)
    _assert_replicated_linear(transformer.patch_embed.proj)
    _assert_replicated_linear(transformer.time_embed.t_embedder.linear_1)
    _assert_replicated_linear(transformer.time_embed.t_embedder.linear_2)
    _assert_replicated_linear(transformer.norm_out.linear_1)
    _assert_replicated_linear(transformer.norm_out.linear_2)
    _assert_replicated_linear(transformer.proj_out)

    # No plain nn.Linear may survive: diffusion LoRA cannot wrap it.
    assert not any(type(module) is nn.Linear for module in transformer.modules())


def test_anima_text_conditioner_uses_vllm_linear_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    conditioner = _build_conditioner(monkeypatch)
    block = conditioner.blocks[0]

    for attn in (block.self_attn, block.cross_attn):
        _assert_replicated_linear(attn.q_proj)
        _assert_replicated_linear(attn.k_proj)
        _assert_replicated_linear(attn.v_proj)
        _assert_replicated_linear(attn.o_proj)
    _assert_replicated_linear(block.mlp[0])
    _assert_replicated_linear(block.mlp[2])
    _assert_replicated_linear(conditioner.in_proj)  # target_dim != model_dim
    _assert_replicated_linear(conditioner.out_proj)

    assert not any(type(module) is nn.Linear for module in conditioner.modules())


def test_anima_vllm_linears_match_nn_linear_numerics(monkeypatch: pytest.MonkeyPatch) -> None:
    """At tp=1 every vLLM linear variant must compute exactly F.linear(x, weight, bias)."""
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch)
    conditioner = _build_conditioner(monkeypatch)

    for module in list(transformer.modules()) + list(conditioner.modules()):
        if not isinstance(module, LinearBase):
            continue
        x = torch.randn(2, 3, module.input_size)
        with torch.inference_mode():
            expected = F.linear(x, module.weight, module.bias)
            torch.testing.assert_close(module(x), expected)


def test_tiny_anima_transformer_forward_with_vllm_linears(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch).eval()

    hidden_states = torch.randn(1, 1, 1, 2, 2)
    timestep = torch.tensor([0.5])
    encoder_hidden_states = torch.randn(1, 3, 4)
    padding_mask = torch.zeros(1, 1, 2, 2)
    with torch.inference_mode():
        first = transformer(
            hidden_states,
            timestep,
            encoder_hidden_states,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        second = transformer(
            hidden_states,
            timestep,
            encoder_hidden_states,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
    assert first.shape == (1, 1, 1, 2, 2)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_tiny_anima_text_conditioner_forward_with_vllm_linears(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    conditioner = _build_conditioner(monkeypatch).eval()

    source_hidden_states = torch.randn(1, 3, 4)
    target_input_ids = torch.ones(1, 2, dtype=torch.long)
    with torch.inference_mode():
        output = conditioner(source_hidden_states, target_input_ids)
    # min_sequence_length pads the sequence dimension to 4; out dim is target_dim.
    assert output.shape == (1, 4, 6)
    assert torch.isfinite(output).all()


def _lora_weights(name: str, lora_a: torch.Tensor, lora_b: torch.Tensor) -> LoRALayerWeights:
    rank = lora_a.shape[0]
    assert lora_b.shape[1] == rank
    return LoRALayerWeights(name, rank, rank, lora_a, lora_b).optimize()


def _lora_request(adapter_id: int) -> LoRARequest:
    return LoRARequest(lora_name=f"a{adapter_id}", lora_int_id=adapter_id, lora_path=f"/tmp/a{adapter_id}")


def test_diffusion_lora_replaces_and_stacks_on_anima_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through DiffusionLoRAManager: the Anima DiT's vLLM linear
    layers are wrapped by the diffusion LoRA layers, and two adapters are
    composed into one rank-sum adapter applied to the real model."""
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch)
    pipeline = nn.Module()
    pipeline.transformer = transformer

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_cached_adapters=4,
    )
    # vLLM LinearBase leaves (including ModuleList-indexed ones) are supported.
    assert "to_q" in manager._supported_lora_modules
    assert "0" in manager._supported_lora_modules

    to_q_dim = 12  # num_attention_heads * attention_head_dim
    r1, r2 = 2, 3
    s1, s2 = 0.5, 2.0
    a1 = torch.randn(r1, to_q_dim)
    b1 = torch.randn(to_q_dim, r1)
    a2 = torch.randn(r2, to_q_dim)
    b2 = torch.randn(to_q_dim, r2)
    a3 = torch.randn(r1, to_q_dim)
    b3 = torch.randn(to_q_dim, r1)
    target_to_q = "transformer.transformer_blocks.0.attn2.to_q"
    target_to_out = "transformer.transformer_blocks.0.attn1.to_out.0"

    models = {
        1: LoRAModel(
            1,
            rank=r1,
            loras={
                target_to_q: _lora_weights(target_to_q, a1, b1),
                target_to_out: _lora_weights(target_to_out, a3, b3),
            },
        ),
        2: LoRAModel(
            2,
            rank=r2,
            loras={target_to_q: _lora_weights(target_to_q, a2, b2)},
        ),
    }

    def _fake_load(req: LoRARequest):
        model = models[req.lora_int_id]
        peft_helper = PEFTHelper.from_dict(
            {
                "r": model.rank,
                "lora_alpha": model.rank,
                "target_modules": sorted({name.split(".")[-1] for name in model.loras}),
            }
        )
        return model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter", _fake_load)

    manager.set_active_adapters([_lora_request(1), _lora_request(2)], [s1, s2])

    block = transformer.transformer_blocks[0]
    wrapped_q = block.attn2.to_q
    wrapped_out = block.attn1.to_out[0]
    assert isinstance(wrapped_q, BaseLayerWithLoRA)
    assert isinstance(wrapped_out, BaseLayerWithLoRA)
    assert manager._active_adapter_id not in (1, 2)

    x = torch.randn(3, to_q_dim)
    with torch.inference_mode():
        base_q = F.linear(x, wrapped_q.base_layer.weight)
        expected_q = base_q + s1 * (x @ a1.t()) @ b1.t() + s2 * (x @ a2.t()) @ b2.t()
        torch.testing.assert_close(wrapped_q(x), expected_q)
        # to_out.0 is only covered by adapter 1.
        base_out = F.linear(x, wrapped_out.base_layer.weight)
        torch.testing.assert_close(wrapped_out(x), base_out + s1 * (x @ a3.t()) @ b3.t())

        # Deactivation restores the base outputs.
        manager.set_active_adapters(None)
        torch.testing.assert_close(wrapped_q(x), base_q)
        torch.testing.assert_close(wrapped_out(x), base_out)
