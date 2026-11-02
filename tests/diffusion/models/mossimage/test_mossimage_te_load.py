# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression test: MOSS-Image HF components must stage on CPU during load.

The diffusion loader constructs the pipeline under a default-device context
(``with torch.device(target_device):``). Under that context HF
``from_pretrained`` takes a direct-to-GPU fast path, so the Qwen3.5 text
encoder (~15.6 GiB bf16) used to materialize straight onto the GPU and OOM
small cards before any offload hook existed. The pipeline now forces a CPU
context around those loads and moves explicitly afterwards.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mossimage import pipeline_mossimage as pipeline_mod
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _DummyTransformer(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _FactoryRecorder:
    """Stand-in for from_pretrained_with_prefetch: records the ambient default device."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, factory, model, **kwargs):
        # Allocate through the ambient default device, like HF's loader does.
        probe = torch.empty(0)
        self.calls.append(
            {
                "subfolder": kwargs.get("subfolder"),
                "default_device": probe.device.type,
                "kwargs": kwargs,
            }
        )
        return nn.Linear(2, 2)


def _build_pipeline(
    monkeypatch: pytest.MonkeyPatch, recorder: _FactoryRecorder, dtype: torch.dtype | None = None
) -> MOSSImagePipeline:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    monkeypatch.setattr(pipeline_mod, "from_pretrained_with_prefetch", recorder)
    monkeypatch.setattr(pipeline_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.MOSSImageScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "get_transformer_config_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_mod, "MOSSImageTransformer", _DummyTransformer)
    monkeypatch.setattr(pipeline_mod, "get_local_device", lambda: torch.device("cpu"))

    kwargs = {"model": "moss-image"}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return MOSSImagePipeline(od_config=OmniDiffusionConfig(**kwargs))


def test_hf_components_stage_on_cpu_inside_device_context(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _FactoryRecorder()

    # The diffusion loader constructs the pipeline under the target device
    # (CUDA in production; meta here stands in for "not CPU").
    with torch.device("meta"):
        pipeline = _build_pipeline(monkeypatch, recorder, dtype=torch.float16)

    assert [call["subfolder"] for call in recorder.calls] == ["text_encoder", "vae"]
    # Without the fix both loads would record "meta" (CUDA in production).
    assert all(call["default_device"] == "cpu" for call in recorder.calls)

    # dtype contract: TE follows od_config.dtype, VAE stays fp32.
    assert recorder.calls[0]["kwargs"]["dtype"] == torch.float16
    assert recorder.calls[1]["kwargs"]["torch_dtype"] == torch.float32

    # With no offload selected, both end up on the pipeline device (CPU here).
    assert next(pipeline.text_encoder.parameters()).device.type == "cpu"
    assert next(pipeline.vae.parameters()).device.type == "cpu"


def test_text_encoder_dtype_defaults_to_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _FactoryRecorder()
    with torch.device("meta"):
        _build_pipeline(monkeypatch, recorder)
    assert recorder.calls[0]["kwargs"]["dtype"] == torch.bfloat16
