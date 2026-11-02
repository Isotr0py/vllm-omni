# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image text-encoder layer-slice TP (``text_encoder_tp_size``).

CPU coverage: the slice is numerically identical to the full forward (tiny
random Qwen3.5), weights actually halve per rank, the DLO replication
declaration is dropped when sharded, and validation rejects bad topologies.
The collective handoff itself is exercised by the gloo two-process test at
the bottom.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mossimage import pipeline_mossimage as pipeline_mod
from vllm_omni.diffusion.models.mossimage.encoder_tp import slice_text_encoder_for_tp
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_WORKER = Path(__file__).with_name("te_tp_dist_worker.py")


def _tiny_text_model(num_layers: int = 4):
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    text_cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
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
    vision_cfg = Qwen3_5VisionConfig(hidden_size=32, depth=2, num_heads=2, out_hidden_size=64)
    config = Qwen3_5Config(text_config=text_cfg.to_dict(), vision_config=vision_cfg.to_dict())
    return Qwen3_5Model(config).eval()


def _tiny_inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 256, (2, 11))
    attention_mask = torch.ones(2, 11, dtype=torch.long)
    attention_mask[1, 7:] = 0
    return input_ids, attention_mask


# ── slicing ──


def test_slice_halves_weights_and_preserves_math() -> None:
    torch.manual_seed(0)
    reference = _tiny_text_model()
    torch.manual_seed(0)
    sharded = _tiny_text_model()  # same seed -> identical weights

    full_params = sum(p.numel() for p in reference.parameters())
    with torch.no_grad():
        expected = reference(**dict(zip(("input_ids", "attention_mask"), _tiny_inputs())), use_cache=False)
        expected = expected.last_hidden_state

    slice_text_encoder_for_tp(sharded, rank=0, tp_size=2)
    lm0 = sharded.language_model
    assert len(lm0.layers) == 2
    assert isinstance(lm0.norm, nn.Identity)
    assert lm0.embed_tokens is not None
    assert getattr(sharded, "visual", None) is None
    assert lm0.config.layer_types == reference.language_model.config.layer_types[:2]

    rank0_params = sum(p.numel() for p in sharded.parameters())

    torch.manual_seed(0)
    sharded1 = _tiny_text_model()
    slice_text_encoder_for_tp(sharded1, rank=1, tp_size=2)
    lm1 = sharded1.language_model
    assert len(lm1.layers) == 2
    assert not hasattr(lm1, "embed_tokens")
    assert not isinstance(lm1.norm, nn.Identity)
    assert lm1.config.layer_types == reference.language_model.config.layer_types[2:]

    rank1_params = sum(p.numel() for p in sharded1.parameters())
    # Each rank holds roughly half the layer weights; the total shrinks
    # because rank1 drops the embedding table and both drop the vision tower.
    assert rank0_params + rank1_params < full_params

    # Hand the hidden state between shells exactly like the broadcast does.
    input_ids, attention_mask = _tiny_inputs()
    with torch.no_grad():
        mid = lm0(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        actual = lm1(inputs_embeds=mid, attention_mask=attention_mask, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_slice_rejects_indivisible_layer_count() -> None:
    model = _tiny_text_model(num_layers=3)
    with pytest.raises(ValueError, match="must divide"):
        slice_text_encoder_for_tp(model, rank=0, tp_size=2)


# ── pipeline integration ──


def _build_pipeline(monkeypatch: pytest.MonkeyPatch, te_tp_size: int, rank: int = 0) -> MOSSImagePipeline:
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    recorder = {}

    def fake_load(factory, model, **kwargs):
        recorder[kwargs.get("subfolder")] = kwargs
        if kwargs.get("subfolder") == "text_encoder":
            torch.manual_seed(0)
            return _tiny_text_model()
        return nn.Linear(2, 2)

    monkeypatch.setattr(pipeline_mod, "from_pretrained_with_prefetch", fake_load)
    monkeypatch.setattr(pipeline_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.MOSSImageScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "get_transformer_config_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_mod, "MOSSImageTransformer", _DummyTransformer)
    monkeypatch.setattr(pipeline_mod, "get_local_device", lambda: torch.device("cpu"))

    if te_tp_size > 1:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        monkeypatch.setattr(
            pipeline_mod,
            "get_world_group",
            lambda: SimpleGroup(world_size=te_tp_size, device_group=None),
        )

    parallel = DiffusionParallelConfig(text_encoder_tp_size=te_tp_size)
    od_config = OmniDiffusionConfig(model="moss-image", dtype=torch.float16, parallel_config=parallel)
    return MOSSImagePipeline(od_config=od_config)


class _DummyTransformer(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class SimpleGroup:
    def __init__(self, world_size: int, device_group) -> None:
        self.world_size = world_size
        self.device_group = device_group


def test_te_tp1_leaves_encoder_and_plan_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch, te_tp_size=1)

    assert pipeline.text_encoder_tp_size == 1
    assert len(pipeline.text_encoder.language_model.layers) == 4
    assert pipeline.text_encoder.visual is not None
    plan = get_offload_plan(pipeline)
    assert "text_encoder" in plan.encoder_dlo_weight_replication


def test_te_tp2_slices_encoder_and_drops_dlo_replication(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch, te_tp_size=2, rank=0)

    assert len(pipeline.text_encoder.language_model.layers) == 2
    plan = get_offload_plan(pipeline)
    # Sharded TE weights differ per rank: AllGather transfer must be rejected.
    assert plan.encoder_dlo_weight_replication == frozenset()


def test_te_tp_rejects_partial_world(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    monkeypatch.setattr(pipeline_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(pipeline_mod, "get_world_group", lambda: SimpleGroup(world_size=4, device_group=None))

    parallel = DiffusionParallelConfig(text_encoder_tp_size=2)
    with pytest.raises(ValueError, match="world size"):
        MOSSImagePipeline(od_config=OmniDiffusionConfig(model="moss-image", parallel_config=parallel))


def test_te_tp_requires_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    monkeypatch.setattr(pipeline_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "get_local_device", lambda: torch.device("cpu"))

    parallel = DiffusionParallelConfig(text_encoder_tp_size=2)
    with pytest.raises(ValueError, match="torch.distributed"):
        MOSSImagePipeline(od_config=OmniDiffusionConfig(model="moss-image", parallel_config=parallel))


# ── real collective path (gloo, two CPU processes) ──


@pytest.mark.skipif(not _WORKER.is_file(), reason="dist worker helper missing")
def test_te_tp_forward_gloo_two_ranks(tmp_path: Path) -> None:
    """Two real gloo ranks run the sliced encode; both must equal the full forward."""
    import vllm_omni

    repo_root = str(Path(vllm_omni.__path__[0]).resolve().parent)
    store = tmp_path / "gloo_store"
    env = {**os.environ, "TE_TP_STORE": str(store)}
    workers = []
    for rank in (0, 1):
        workers.append(
            subprocess.Popen(
                [sys.executable, str(_WORKER), str(rank), repo_root],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = []
    for worker in workers:
        out, _ = worker.communicate(timeout=180)
        outputs.append(out)
    for rank, (worker, out) in enumerate(zip(workers, outputs)):
        assert worker.returncode == 0, f"rank {rank} failed:\n{out[-2000:]}"
        assert "TE_TP_DIST_OK" in out, out[-2000:]
