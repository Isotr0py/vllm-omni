# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offload declaration and plan-resolution tests for AnimaPipeline.

Covers the layerwise/DLO integration surface: the DiT block-container
declaration, SupportsComponentDiscovery module lists, the pipeline
``_offload_plan`` (encoder streaming + DLO weight-replication safety), and the
ControlNet-LLLite attachment invariants that make block streaming safe
(correction modules are not block submodules and stay resident).

Plan resolution runs against a tiny fake DiT so these tests need no checkpoint,
no GPU, and no vLLM linear layers; the real DiT class is checked by its
declaration only (instantiation is covered by test_anima_controlnet.py).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel
from vllm_omni.diffusion.models.anima.controlnet import AnimaControlNetLLLite
from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.config import DLOTransfer, OffloadStrategy
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 3
HIDDEN = 24

_TINY_LLLITE = {
    "num_blocks": NUM_LAYERS,
    "hidden_dim": HIDDEN,
    "cond_emb_dim": 8,
    "mlp_dim": 8,
    "cond_dim": 16,
    "cond_resblocks": 1,
    "aspp_dilations": (1, 2),
}


class _TinyAttention(nn.Module):
    """Attention stand-in exposing the ``lllite`` hook point ControlNet writes to."""

    def __init__(self) -> None:
        super().__init__()
        self.to_qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _TinyAttention()
        self.ff = nn.Linear(HIDDEN, HIDDEN)


class _TinyDiT(nn.Module):
    """Fake DiT mirroring AnimaTransformer3DModel's offload declaration."""

    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(self, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(HIDDEN, HIDDEN)
        self.transformer_blocks = nn.ModuleList(_TinyBlock() for _ in range(num_layers))
        self.norm_out = nn.LayerNorm(HIDDEN)
        self.proj_out = nn.Linear(HIDDEN, HIDDEN)


class _TinyTextEncoder(nn.Module):
    """Stands in for the Qwen3 text encoder: exposes a ``layers`` block list."""

    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(num_layers))
        self.norm = nn.LayerNorm(4)


class _TinyVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Linear(4, 4)


def _make_offload_probe(with_controlnet: bool = False) -> AnimaPipeline:
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _TinyDiT()
    pipeline.text_encoder = _TinyTextEncoder()
    pipeline.vae = _TinyVae()
    pipeline.controlnet = None
    if with_controlnet:
        controlnet = AnimaControlNetLLLite(**_TINY_LLLITE)
        controlnet.apply_to(pipeline.transformer)
        pipeline.controlnet = controlnet
    return pipeline


# ── Declarations ──


def test_transformer_declares_layerwise_offload_blocks() -> None:
    # The offload framework resolves the DiT block ring through this class
    # attribute (via get_blocks_from_dit) when the plan declares no block_attrs.
    assert AnimaTransformer3DModel._layerwise_offload_blocks_attrs == ["transformer_blocks"]


def test_pipeline_declares_component_discovery() -> None:
    pipeline = AnimaPipeline.__new__(AnimaPipeline)

    assert isinstance(pipeline, SupportsComponentDiscovery)
    assert AnimaPipeline._dit_modules == ["transformer"]
    assert AnimaPipeline._encoder_modules == ["text_encoder"]
    assert AnimaPipeline._vae_modules == ["vae"]

    plan = AnimaPipeline._offload_plan
    assert isinstance(plan, OffloadPlan)
    assert "transformer" in plan.resident_dit_paths
    assert plan.encoder_block_attrs["text_encoder"] == ("layers",)
    assert plan.encoder_component_types["text_encoder"] == "text_encoder"
    # The Qwen3 text encoder is from_pretrained'd identically on every rank.
    assert "text_encoder" in plan.encoder_dlo_weight_replication


def test_controlnet_stays_out_of_offload_discovery() -> None:
    pipeline = _make_offload_probe(with_controlnet=True)

    discovered = ModuleDiscovery.discover(pipeline)

    assert discovered.dit_names == ["transformer"]
    assert discovered.encoder_names == ["text_encoder"]
    assert discovered.vae_names == ["vae"]
    all_modules = discovered.dits + discovered.encoders + discovered.vaes + discovered.resident_modules
    assert all(pipeline.controlnet is not module for module in all_modules)


# ── Plan resolution ──


def test_resolve_layerwise_streams_dit_blocks_and_encoder_layers() -> None:
    pipeline = _make_offload_probe()
    config = OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=False)

    resolved = resolve_offload_plan(pipeline, config)

    dit = resolved.dits[0]
    assert dit.path == "transformer"
    assert len(dit.stacks) == 1
    stack = dit.stacks[0]
    assert stack.attrs == ("transformer_blocks",)
    assert list(stack.blocks) == list(pipeline.transformer.transformer_blocks)
    assert stack.resident_head == 0

    encoder = resolved.encoders[0]
    assert encoder.path == "text_encoder"
    assert len(encoder.stacks) == 1
    assert list(encoder.stacks[0].blocks) == list(pipeline.text_encoder.layers)


def test_resolve_dlo_rank_local_supports_resident_layers() -> None:
    pipeline = _make_offload_probe()
    config = OffloadConfig(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        dlo_use_allgather=False,
        dlo_resident_layers=1,
    )

    resolved = resolve_offload_plan(pipeline, config)

    stack = resolved.dits[0].stacks[0]
    assert stack.resident_head == 1
    assert list(stack.resident) == [pipeline.transformer.transformer_blocks[0]]
    assert list(stack.streaming) == list(pipeline.transformer.transformer_blocks[1:])


def test_resolve_dlo_encoder_allgather_allowed_by_replication_declaration() -> None:
    pipeline = _make_offload_probe()
    config = OffloadConfig(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        dp_size=2,
        dlo_transfers={"dit": DLOTransfer.RANK_LOCAL, "text_encoder": DLOTransfer.ALLGATHER},
    )

    resolved = resolve_offload_plan(pipeline, config)

    assert list(resolved.encoders[0].stacks[0].blocks) == list(pipeline.text_encoder.layers)


def test_resolve_dlo_encoder_allgather_rejected_without_replication_declaration() -> None:
    pipeline = _make_offload_probe()
    pipeline._offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("layers",)},
    )
    config = OffloadConfig(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        dp_size=2,
        dlo_transfers={"dit": DLOTransfer.RANK_LOCAL, "text_encoder": DLOTransfer.ALLGATHER},
    )

    with pytest.raises(ValueError, match="not declared replicated"):
        resolve_offload_plan(pipeline, config)


# ── ControlNet vs block streaming ──


def test_controlnet_correction_modules_are_not_block_submodules() -> None:
    """LLLite modules attach as a plain attribute tuple, so block streaming never moves them."""
    pipeline = _make_offload_probe(with_controlnet=True)
    transformer = pipeline.transformer
    controlnet = pipeline.controlnet

    block = transformer.transformer_blocks[0]
    # The hook point is a plain attribute tuple, not a registered submodule...
    assert isinstance(block.attn1.lllite, tuple)
    assert "lllite" not in block.attn1._modules
    # ...so neither the block's nor the whole DiT's parameter set owns the
    # correction weights; offload hooks moving block storage leave them alone.
    transformer_param_ids = {id(param) for param in transformer.parameters()}
    for param in controlnet.parameters():
        assert id(param) not in transformer_param_ids

    # Simulated block offload: streaming the blocks off-device does not move
    # any ControlNet weight, and the tuple references stay intact.
    for streamed_block in transformer.transformer_blocks:
        streamed_block.to("meta")
    assert isinstance(block.attn1.lllite, tuple)
    for param in controlnet.parameters():
        assert param.device.type == "cpu"

    controlnet.restore(transformer)
    assert block.attn1.lllite is None
