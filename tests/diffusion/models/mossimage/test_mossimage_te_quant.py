# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image text-encoder online weight-only quantization.

The TE swaps its tp_plan-covered linears for vLLM parallel classes carrying a
quant method (convrot_int8 / int8 via vLLM INT8 kernels, or bnb nf4/fp4):
shards stay lazy until their weights stream in, then quantize in place per
layer — shard first, quantize second. CPU coverage: component-config routing,
replacement wiring per TP topology, DLO/method combination rules, and a DLO
transport roundtrip for int8-shaped state. ConvRot rotation numerics
(orthogonality, group tiling, outlier spreading) are covered by the framework
tests in tests/diffusion/quantization/test_convrot_config.py.
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

_CONVROT_TE = {"text_encoder": {"method": "convrot_int8"}}
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
    """convrot/int8 quant guards on a CUDA platform; pretend we have one."""
    import vllm_omni.quantization.convrot_config as convrot_cfg
    import vllm_omni.quantization.int8_config as int8_cfg

    fake = SimpleNamespace(is_cuda=lambda: True, is_npu=lambda: False)
    monkeypatch.setattr(convrot_cfg, "current_omni_platform", fake)
    monkeypatch.setattr(int8_cfg, "current_omni_platform", fake)


def _convrot_linears(model: nn.Module) -> list[nn.Module]:
    from vllm_omni.quantization.convrot_config import ConvRotInt8OnlineLinearMethod

    return [m for m in model.modules() if isinstance(getattr(m, "quant_method", None), ConvRotInt8OnlineLinearMethod)]


# ── config routing / combination rules ──


def test_component_config_routes_te_only() -> None:
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.component_config import resolve_component_quant_config

    config = build_quant_config(_CONVROT_TE)
    te = resolve_component_quant_config(config, "text_encoder")
    assert te is not None and te.get_name() == "convrot_int8"
    # The DiT resolves to the (absent) default: never quantized by this spec.
    assert resolve_component_quant_config(config, "transformer") is None


def test_bnb_quant_rejects_text_encoder_offload() -> None:
    probe = _make_probe(1, None)
    probe.od_config = OmniDiffusionConfig(
        model="moss-image",
        quantization_config=_NF4_TE,
        diffusion_offload_config={"mode": "layer", "components": ["dit", "text_encoder"]},
    )
    with pytest.raises(ValueError, match="incompatible with text_encoder offload"):
        probe._init_text_encoder(_tiny_config(), torch.float32)


def test_convrot_int8_allows_text_encoder_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    """int8-family state lives in plain Parameters/buffers, so DLO may move it."""
    _mock_cuda_platform(monkeypatch)
    probe = _make_probe(1, None)
    probe.od_config = OmniDiffusionConfig(
        model="moss-image",
        quantization_config=_CONVROT_TE,
        diffusion_offload_config={"mode": "layer", "components": ["dit", "text_encoder"]},
    )
    probe._init_text_encoder(_tiny_config(), torch.float32)  # must not raise


def test_quant_rejects_unsupported_method_on_te() -> None:
    probe = _make_probe(1, {"text_encoder": {"method": "fp8"}})
    with pytest.raises(ValueError, match="online weight-only"):
        probe._init_text_encoder(_tiny_config(), torch.float32)


# ── replacement wiring per topology ──


def test_quant_tp1_swaps_linears_and_keeps_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda_platform(monkeypatch)
    _mock_tp(monkeypatch, size=1, rank=0)
    from vllm.model_executor.layers.linear import LinearBase

    probe = _make_probe(1, _CONVROT_TE)
    model = probe._init_text_encoder(_tiny_config(), torch.float32)

    q_proj = model.language_model.layers[1].self_attn.q_proj
    # TP1 + quant: plan-covered linears become vLLM linear classes carrying
    # the quant method (full-size weights at tp_size=1).
    assert isinstance(q_proj, LinearBase) and type(q_proj) is not nn.Linear
    reference = _tiny_text_model()
    ref_w = reference.language_model.layers[1].self_attn.q_proj.weight
    assert tuple(q_proj.weight.shape) == tuple(ref_w.shape)  # full size, no sharding
    assert _convrot_linears(model), "no convrot quant methods installed"
    # Embeddings and norms are never quantized.
    assert isinstance(model.language_model.embed_tokens, nn.Embedding)


def test_quant_tp2_keeps_megatron_sharding(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda_platform(monkeypatch)
    _mock_tp(monkeypatch, size=2, rank=0)
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    probe = _make_probe(2, _CONVROT_TE)
    model = probe._init_text_encoder(_tiny_config(), torch.float32)

    lm = model.language_model
    assert isinstance(lm.layers[0].linear_attn.in_proj_qkv, ColumnParallelLinear)
    assert lm.layers[0].linear_attn.in_proj_qkv.gather_output is True
    assert isinstance(lm.layers[1].self_attn.o_proj, RowParallelLinear)
    assert _convrot_linears(model), "no convrot quant methods installed"


def test_quant_tp2_shard_shape_and_streamed_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sharded-then-quantized loading: each rank's lazy shard receives its slice."""
    _mock_cuda_platform(monkeypatch)
    _mock_tp(monkeypatch, size=2, rank=0)
    reference = _tiny_text_model()
    checkpoint = tmp_path / "text_encoder.safetensors"
    _write_checkpoint(reference, checkpoint)
    original = {name: param.detach().clone() for name, param in reference.named_parameters()}

    probe = _make_probe(2, _CONVROT_TE)
    probe.text_encoder = probe._init_text_encoder(_tiny_config(), torch.float32)
    probe.load_weights(_iter_checkpoint(checkpoint, "text_encoder."))

    w_qkv = original["language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    shard = probe.text_encoder.language_model.layers[0].linear_attn.in_proj_qkv.weight.data
    assert shard.shape[0] == w_qkv.shape[0] // 2  # shard first
    torch.testing.assert_close(shard, w_qkv.narrow(0, 0, shard.shape[0]))  # rank 0 slice intact


# ── DLO transport of int8-family state ──


class _QuantizedBlock(nn.Module):
    """A block carrying int8-family quantized state (int8 weight, fp32 scale,
    non-persistent fp16 rotation buffer), as produced by convrot_int8/int8."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.weight = nn.Parameter(torch.randint(-127, 127, (4, 4), dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.randn(4, 1), requires_grad=False)
        self.register_buffer("convrot_hadamard", torch.randn(4, 4), persistent=False)

    def forward(self, x, **kwargs):
        return x + x @ self.weight.float() * self.weight_scale.mean() + self.convrot_hadamard.sum()


def test_layerwise_offload_roundtrips_quantized_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """DLO/layerwise storage relocation must not lose int8 params or rotation buffers."""
    import contextlib

    import vllm_omni.diffusion.offloader.layerwise_backend as lw

    class DummyStream:
        def wait_stream(self, _s):
            return None

        def wait_event(self, _e):
            return None

    class DummyEvent:
        def record(self, _s):
            return None

        def synchronize(self):
            return None

    @contextlib.contextmanager
    def dummy_stream(_s):
        yield None

    monkeypatch.setattr(lw.current_omni_platform, "Stream", DummyStream, raising=False)
    monkeypatch.setattr(lw.current_omni_platform, "Event", DummyEvent, raising=False)
    monkeypatch.setattr(lw.current_omni_platform, "current_stream", lambda: DummyStream(), raising=False)
    monkeypatch.setattr(lw.current_omni_platform, "stream", dummy_stream, raising=False)
    monkeypatch.setattr(lw.current_omni_platform, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(lw.current_omni_platform, "empty_cache", lambda: None, raising=False)

    pipeline = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = nn.Module()
    pipeline.transformer.layers = nn.ModuleList([_QuantizedBlock(), _QuantizedBlock()])
    pipeline.text_encoder = nn.Module()
    pipeline.text_encoder.language_model = nn.Module()
    pipeline.text_encoder.language_model.layers = nn.ModuleList([_QuantizedBlock(), _QuantizedBlock()])
    pipeline.vae = nn.Identity()

    original = {name: t.clone() for name, t in pipeline.state_dict().items()}
    hadamards = [m.convrot_hadamard.clone() for m in pipeline.modules() if hasattr(m, "convrot_hadamard")]

    from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy

    backend = lw.LayerWiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.LAYER_WISE,
            pin_cpu_memory=False,
            components=frozenset({"dit", "text_encoder"}),
        ),
        torch.device("cpu"),
    )
    backend.enable(pipeline)
    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        for block in pipeline.text_encoder.language_model.layers:
            x = block(x)
    backend.disable()

    for name, tensor in pipeline.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    quantized = [m for m in pipeline.modules() if hasattr(m, "convrot_hadamard")]
    for module, expected in zip(quantized, hadamards):
        torch.testing.assert_close(module.convrot_hadamard, expected, rtol=0, atol=0)
