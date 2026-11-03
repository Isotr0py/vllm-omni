# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image text-encoder online weight-only quantization.

The TE swaps its tp_plan-covered linears for vLLM parallel classes carrying a
quant method (bnb nf4/fp4, or vLLM int8 on sm80+): shards stay lazy until their
weights stream in, then quantize in place per layer — shard first, quantize
second. CPU coverage: component-config routing, replacement wiring per TP
topology, DLO/method rejection rules, and a numeric nf4 forward comparison
against the fp32 reference (the exact quantize/matmul calls the production
linear method makes; bnb runs them on CPU).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_mossimage_te_vllm_tp import (
    _iter_checkpoint,
    _mock_tp,
    _tiny_config,
    _tiny_text_model,
    _write_checkpoint,
)
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_NF4_TE = {"text_encoder": {"method": "bitsandbytes", "quant_type": "nf4"}}


def _make_probe(te_tp_size: int, quantization_config: dict | None) -> MOSSImagePipeline:
    probe = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(probe)
    probe.text_encoder_tp_size = te_tp_size
    probe._te_rank = 0
    probe._text_encoder_device = torch.device("cpu")
    probe.transformer = nn.Linear(2, 2)
    probe.od_config = OmniDiffusionConfig(model="moss-image", quantization_config=quantization_config)
    return probe


def _mock_cuda_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """bnb online quant guards on a CUDA platform; pretend we have one."""
    import vllm_omni.quantization.bitsandbytes_config as bnb_cfg

    monkeypatch.setattr(
        bnb_cfg,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: True, is_npu=lambda: False),
    )


def _bnb_linears(model: nn.Module) -> list[nn.Module]:
    from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

    return [m for m in model.modules() if isinstance(getattr(m, "quant_method", None), BnBOnlineLinearMethod)]


# ── config routing / rejection rules ──


def test_component_config_routes_te_only() -> None:
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.component_config import resolve_component_quant_config

    config = build_quant_config(_NF4_TE)
    te = resolve_component_quant_config(config, "text_encoder")
    assert te is not None and te.get_name() == "bitsandbytes" and te.quant_type == "nf4"
    # The DiT resolves to the (absent) default: never quantized by this spec.
    assert resolve_component_quant_config(config, "transformer") is None


def test_quant_rejects_text_encoder_offload() -> None:
    probe = _make_probe(1, None)
    probe.od_config = OmniDiffusionConfig(
        model="moss-image",
        quantization_config=_NF4_TE,
        diffusion_offload_config={"mode": "layer", "components": ["dit", "text_encoder"]},
    )
    with pytest.raises(ValueError, match="incompatible with text_encoder offload"):
        probe._init_text_encoder(_tiny_config(), torch.float32)


def test_quant_rejects_unsupported_method_on_te() -> None:
    probe = _make_probe(1, {"text_encoder": {"method": "fp8"}})
    with pytest.raises(ValueError, match="online weight-only"):
        probe._init_text_encoder(_tiny_config(), torch.float32)


# ── replacement wiring per topology ──


def test_quant_tp1_swaps_linears_and_keeps_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda_platform(monkeypatch)
    _mock_tp(monkeypatch, size=1, rank=0)
    from vllm.model_executor.layers.linear import LinearBase

    probe = _make_probe(1, _NF4_TE)
    model = probe._init_text_encoder(_tiny_config(), torch.float32)

    q_proj = model.language_model.layers[1].self_attn.q_proj
    # TP1 + quant: plan-covered linears become vLLM linear classes carrying
    # the quant method (full-size weights at tp_size=1).
    assert isinstance(q_proj, LinearBase) and type(q_proj) is not nn.Linear
    reference = _tiny_text_model()
    ref_w = reference.language_model.layers[1].self_attn.q_proj.weight
    assert tuple(q_proj.weight.shape) == tuple(ref_w.shape)  # full size, no sharding
    assert _bnb_linears(model), "no bnb quant methods installed"
    # Embeddings and norms are never quantized.
    assert isinstance(model.language_model.embed_tokens, nn.Embedding)


def test_quant_tp2_keeps_megatron_sharding(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda_platform(monkeypatch)
    _mock_tp(monkeypatch, size=2, rank=0)
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    probe = _make_probe(2, _NF4_TE)
    model = probe._init_text_encoder(_tiny_config(), torch.float32)

    lm = model.language_model
    assert isinstance(lm.layers[0].linear_attn.in_proj_qkv, ColumnParallelLinear)
    assert lm.layers[0].linear_attn.in_proj_qkv.gather_output is True
    assert isinstance(lm.layers[1].self_attn.o_proj, RowParallelLinear)
    assert _bnb_linears(model), "no bnb quant methods installed"


def test_quant_tp2_shard_shape_and_streamed_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sharded-then-quantized loading: each rank's lazy shard receives its slice."""
    _mock_cuda_platform(monkeypatch)
    _mock_tp(monkeypatch, size=2, rank=0)
    reference = _tiny_text_model()
    checkpoint = tmp_path / "text_encoder.safetensors"
    _write_checkpoint(reference, checkpoint)
    original = {name: param.detach().clone() for name, param in reference.named_parameters()}

    probe = _make_probe(2, _NF4_TE)
    probe.text_encoder = probe._init_text_encoder(_tiny_config(), torch.float32)
    probe.load_weights(_iter_checkpoint(checkpoint, "text_encoder."))

    w_qkv = original["language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    shard = probe.text_encoder.language_model.layers[0].linear_attn.in_proj_qkv.weight.data
    assert shard.shape[0] == w_qkv.shape[0] // 2  # shard first
    torch.testing.assert_close(shard, w_qkv.narrow(0, 0, shard.shape[0]))  # rank 0 slice intact


# ── nf4 numerics (the exact calls BnBOnlineLinearMethod makes) ──


class _Nf4Linear(nn.Module):
    """Mirror of BnBOnlineLinearMethod process/apply on CPU (bnb 0.50 CPU backend)."""

    def __init__(self, linear: nn.Linear, quant_type: str = "nf4") -> None:
        super().__init__()
        import bitsandbytes.functional as bnb_F

        self.qweight, self.quant_state = bnb_F.quantize_4bit(
            linear.weight.data.half().contiguous(), quant_type=quant_type, compress_statistics=True
        )
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.half())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import bitsandbytes as bnb

        ori_dtype = x.dtype
        x_2d = x.half().reshape(-1, x.shape[-1])
        out = bnb.matmul_4bit(x_2d, self.qweight.t(), quant_state=self.quant_state, bias=self.bias)
        return out.reshape(*x.shape[:-1], -1).to(ori_dtype)


def _nf4_swap(model: nn.Module) -> nn.Module:
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                setattr(parent, name, _Nf4Linear(child))
    return model


def test_nf4_forward_matches_fp32_reference() -> None:
    torch.manual_seed(0)
    reference = _tiny_text_model()
    quantized = _nf4_swap(_tiny_text_model())

    input_ids = torch.randint(0, 256, (2, 11))
    attention_mask = torch.ones(2, 11, dtype=torch.long)
    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        actual = quantized(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    cos = torch.nn.functional.cosine_similarity(expected.flatten().double(), actual.flatten().double(), dim=0)
    assert cos.item() > 0.99, f"nf4 TE forward diverged from fp32 reference: cos={cos.item()}"
