# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image VAE decode memory levers.

The FLUX.2-klein decoder runs in fp32; decoding a 1024px latent without
tiling peaks at several GiB of activations, which OOMed T4-class cards at
the denoise -> decode transition. The pipeline must therefore honor
``vae_use_slicing`` / ``vae_use_tiling`` / ``vae_patch_parallel_size`` and
return denoise-activation cache blocks to the driver before decoding.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mossimage import pipeline_mossimage as pipeline_mod
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeVAE(nn.Module):
    """Stand-in for AutoencoderKLFlux2: records memory-lever calls."""

    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.bn = SimpleNamespace(running_var=torch.ones(4), running_mean=torch.zeros(4))
        self.config = SimpleNamespace(batch_norm_eps=1e-5)
        self.decoder = nn.Identity()  # patch-parallel wrapper duck-types on this
        self.slicing_enabled = False
        self.tiling_enabled = False

    def enable_slicing(self) -> None:
        self.slicing_enabled = True

    def enable_tiling(self) -> None:
        self.tiling_enabled = True

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        return (z,)


class _DummyTransformer(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


def _build_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vae_use_slicing: bool = False,
    vae_use_tiling: bool = False,
    vae_patch_parallel_size: int = 1,
) -> MOSSImagePipeline:
    from test_mossimage_te_load import _tiny_config

    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    monkeypatch.setattr(pipeline_mod, "from_pretrained_with_prefetch", lambda *a, **k: _FakeVAE())
    monkeypatch.setattr(pipeline_mod.AutoConfig, "from_pretrained", lambda *a, **k: _tiny_config())
    monkeypatch.setattr(pipeline_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.MOSSImageScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "get_transformer_config_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_mod, "MOSSImageTransformer", _DummyTransformer)
    monkeypatch.setattr(pipeline_mod, "get_local_device", lambda: torch.device("cpu"))

    parallel = DiffusionParallelConfig(vae_patch_parallel_size=vae_patch_parallel_size)
    od_config = OmniDiffusionConfig(
        model="moss-image",
        dtype=torch.float16,
        parallel_config=parallel,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
    )
    return MOSSImagePipeline(od_config=od_config)


def test_vae_memory_levers_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch)
    assert pipeline.vae.slicing_enabled is False
    assert pipeline.vae.tiling_enabled is False
    assert getattr(pipeline.vae, "_vllm_vae_patch_parallel_installed", False) is False


def test_vae_slicing_and_tiling_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch, vae_use_slicing=True, vae_use_tiling=True)
    assert pipeline.vae.slicing_enabled is True
    assert pipeline.vae.tiling_enabled is True


def test_vae_patch_parallel_wraps_decode_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch, vae_patch_parallel_size=2)
    assert getattr(pipeline.vae, "_vllm_vae_patch_parallel_installed", False) is True


def test_vae_decode_returns_cache_blocks_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("empty_cache"))
    # Pretend to sit on a CUDA device; the decode itself stays on CPU tensors.
    pipeline.device = torch.device("cuda")

    latents = torch.zeros(1, 4, 2, 2)
    out = pipeline._vae_decode(latents)

    assert calls == ["empty_cache"]
    # BN denormalization + 2x2 unpack still applied before decode.
    assert out.shape == (1, 1, 4, 4)


def test_vae_decode_skips_accelerator_cache_clear_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("empty_cache"))

    out = pipeline._vae_decode(torch.zeros(1, 4, 2, 2))

    assert calls == []
    assert out.shape == (1, 1, 4, 4)
