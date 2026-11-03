# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression test: MOSS-Image HF components must not load straight to GPU,
and the text encoder must never materialize a full-size host copy.

The diffusion loader constructs the pipeline under a default-device context
(``with torch.device(target_device):``). Under that context HF
``from_pretrained`` takes a direct-to-GPU fast path, which OOMed small cards
(T4) at load time. Two-stage fix:

1. The text encoder is built as an empty CPU skeleton (``AutoConfig`` +
   ``AutoModel.from_config`` under a meta/CPU context); its weights stream in
   through the loader's lazily-iterated safetensors path, straight into the
   (possibly TP-sharded) params, and ``load_weights`` moves the module to its
   runtime device afterwards.
2. The VAE keeps the from_pretrained path but is staged under an explicit CPU
   context and moved afterwards.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mossimage import pipeline_mossimage as pipeline_mod
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    text_cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
        max_position_embeddings=128,
    )
    vision_cfg = Qwen3_5VisionConfig(hidden_size=32, depth=1, num_heads=2, out_hidden_size=64)
    return Qwen3_5Config(text_config=text_cfg.to_dict(), vision_config=vision_cfg.to_dict())


class _DummyTransformer(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _FactoryRecorder:
    """Stand-in for from_pretrained_with_prefetch: records the ambient default device."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, factory, model, **kwargs):
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
    monkeypatch.setattr(pipeline_mod.AutoConfig, "from_pretrained", lambda *a, **k: _tiny_config())
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


def test_text_encoder_is_a_cpu_skeleton_and_vae_stays_on_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _FactoryRecorder()

    # The diffusion loader constructs the pipeline under the target device
    # (CUDA in production; meta here stands in for "not CPU").
    with torch.device("meta"):
        pipeline = _build_pipeline(monkeypatch, recorder, dtype=torch.float16)

    # The text encoder no longer goes through from_pretrained at all: the VAE
    # is the only from_pretrained_with_prefetch call left, staged on CPU.
    assert [call["subfolder"] for call in recorder.calls] == ["vae"]
    assert recorder.calls[0]["default_device"] == "cpu"
    assert recorder.calls[0]["kwargs"]["torch_dtype"] == torch.float32

    # TE skeleton: real fp16 CPU params, no weights loaded yet, declared as a
    # loader source so weights stream in later.
    param = next(pipeline.text_encoder.parameters())
    assert param.device.type == "cpu" and param.dtype == torch.float16
    assert pipeline.text_encoder_tp_size == 1
    # TE streams first (moved off host before the DiT stream starts), DiT second.
    te_source = pipeline.weights_sources[0]
    assert te_source.subfolder == "text_encoder" and te_source.prefix == "text_encoder."
    # Both big components must stream one tensor at a time: the multi-thread
    # loader reads whole shard files (num_weight_load_threads at a time) into
    # host RAM, which is several times the component size on every rank.
    assert all(source.streaming_load for source in pipeline.weights_sources)

    # With no offload selected the VAE moved to the pipeline device (CPU here).
    assert next(pipeline.vae.parameters()).device.type == "cpu"


def test_text_encoder_leaves_host_before_dit_streams() -> None:
    """load_weights must move the TE to its runtime device at the TE/DiT stream boundary.

    The TE (~15 GiB fp16) and the DiT (~9.5 GiB) must never sit in host RAM at
    the same time on a rank; streaming the DiT into CPU params while the TE is
    still staged there doubles the per-rank startup peak.
    """
    pipeline = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.transformer = nn.Linear(2, 2)
    pipeline._text_encoder_device = torch.device("meta")  # marker for "moved off host"

    events: list[str] = []
    module_to = nn.Module.to

    def recording_to(self, *args, **kwargs):
        # At move time the DiT must not have received any streamed weights yet.
        dit_loaded = bool((pipeline.transformer.weight == 1).all())
        events.append(f"MOVE(dit_loaded={dit_loaded})")
        return module_to(self, *args, **kwargs)

    import types

    pipeline.text_encoder.to = types.MethodType(recording_to, pipeline.text_encoder)  # type: ignore[method-assign]

    def weights():
        for name in ("text_encoder.weight", "text_encoder.bias", "transformer.weight", "transformer.bias"):
            yield name, torch.ones(2, 2) if name.endswith("weight") else torch.ones(2)

    loaded = pipeline.load_weights(weights())

    assert loaded == {"text_encoder.weight", "text_encoder.bias", "transformer.weight", "transformer.bias"}
    assert events == ["MOVE(dit_loaded=False)"]  # exactly one move, before the DiT stream
    assert pipeline.text_encoder.weight.is_meta  # moved to the marker device
    assert bool((pipeline.transformer.weight == 1).all())  # DiT weights still landed


def test_text_encoder_dtype_defaults_to_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _FactoryRecorder()
    with torch.device("meta"):
        pipeline = _build_pipeline(monkeypatch, recorder)
    assert next(pipeline.text_encoder.parameters()).dtype == torch.bfloat16


def test_streaming_sources_bypass_multithread_shard_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """streaming_load sources must use the lazy per-tensor safetensors iterator.

    The default multi-thread path loads whole shard files into host RAM,
    ``num_weight_load_threads`` at a time — for the ~15.6 GiB bf16 text encoder
    that is a full-size host copy per rank, which OOMs small-RAM multi-GPU
    nodes. Default (non-streaming) sources must keep the multi-thread path.
    """
    from types import SimpleNamespace

    from vllm_omni.diffusion.model_loader import diffusers_loader as loader_mod
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    load_config = SimpleNamespace(
        use_tqdm_on_load=False,
        safetensors_load_strategy=None,
        download_dir=None,
    )
    od_config = SimpleNamespace(
        quantization_config=None,
        parallel_config=SimpleNamespace(),
        enable_multithread_weight_load=True,
        num_weight_load_threads=4,
    )
    loader = DiffusersPipelineLoader(load_config, od_config)
    loader.counter_before_loading_weights = 0.0
    files = ["/w/model-00001-of-00002.safetensors", "/w/model-00002-of-00002.safetensors"]
    monkeypatch.setattr(loader, "_prepare_weights", lambda *a, **k: ("/w", files, True))

    calls: dict[str, list] = {"multi": [], "single": []}

    def fake_multi(files_, *_args, **_kwargs):
        calls["multi"].append(list(files_))
        return iter([("w", torch.zeros(1))])

    def fake_single(files_, *_args, **_kwargs):
        calls["single"].append(list(files_))
        return iter([("w", torch.zeros(1))])

    monkeypatch.setattr(loader_mod, "multi_thread_safetensors_weights_iterator", fake_multi)
    monkeypatch.setattr(loader_mod, "safetensors_weights_iterator", fake_single)

    streaming = DiffusersPipelineLoader.ComponentSource(
        model_or_path="/w",
        subfolder=None,
        revision=None,
        prefix="transformer.",
        streaming_load=True,
    )
    list(loader._get_weights_iterator(streaming))
    assert calls["single"] == [files]
    assert calls["multi"] == []

    default = DiffusersPipelineLoader.ComponentSource(
        model_or_path="/w",
        subfolder=None,
        revision=None,
        prefix="transformer.",
    )
    list(loader._get_weights_iterator(default))
    assert calls["multi"] == [files]  # default behavior unchanged
