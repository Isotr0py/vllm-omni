# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image text-encoder megatron TP (``text_encoder_tp_size``).

The HF checkpoint's own ``tp_plan`` drives the sharding: every covered
``nn.Linear`` becomes the matching vLLM parallel linear, and weights stream
from the checkpoint one tensor at a time into the rank-local shard through
the same ``AutoWeightsLoader`` path the DiT uses — no full-size host copy of
the encoder ever exists.  CPU coverage: skeleton construction, lazy weight
feeding, plan-coverage validation, the DLO replication drop, topology
validation, and a real two-process gloo forward equivalence check.
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
from vllm_omni.diffusion.models.mossimage.encoder_vllm_tp import apply_text_encoder_tp_plan
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_WORKER = Path(__file__).with_name("te_vllm_tp_dist_worker.py")


def _tiny_config(num_layers: int = 4):
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

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
    return Qwen3_5Config(text_config=text_cfg.to_dict(), vision_config=vision_cfg.to_dict())


def _tiny_text_model(num_layers: int = 4):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    torch.manual_seed(0)
    return Qwen3_5Model(_tiny_config(num_layers)).eval()


def _write_checkpoint(model: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    save_file(model.state_dict(), str(path))


def _iter_checkpoint(path: Path, prefix: str):
    """Yield (prefixed_name, tensor) one at a time, like the loader's lazy iterator."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            yield prefix + key, f.get_tensor(key)


def _mock_tp(monkeypatch: pytest.MonkeyPatch, size: int, rank: int) -> None:
    """Point the vLLM TP accessors (used by the parallel linears) at a fake topology."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)


def _make_probe(te_tp_size: int) -> MOSSImagePipeline:
    pipeline = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.text_encoder_tp_size = te_tp_size
    pipeline._text_encoder_device = torch.device("cpu")
    pipeline.transformer = nn.Linear(2, 2)
    return pipeline


# ── loader-fed construction (te_tp_size == 1) ──


def test_loader_fed_text_encoder_matches_from_pretrained(tmp_path: Path) -> None:
    checkpoint = tmp_path / "text_encoder.safetensors"
    reference = _tiny_text_model()
    _write_checkpoint(reference, checkpoint)

    probe = _make_probe(te_tp_size=1)
    probe.text_encoder = probe._init_text_encoder(_tiny_config(), torch.float32)
    # The vision tower is dropped at skeleton-build time.
    assert getattr(probe.text_encoder, "visual", None) is None
    # tp1: no parallel-linears replacement.
    assert isinstance(probe.text_encoder.language_model.layers[1].self_attn.q_proj, nn.Linear)

    loaded = probe.load_weights(_iter_checkpoint(checkpoint, "text_encoder."))
    assert any(name.startswith("text_encoder.language_model.") for name in loaded)

    # Every language_model param matches the from_pretrained reference exactly.
    lm = probe.text_encoder.language_model
    reference_lm = reference.language_model
    for name, param in lm.named_parameters():
        torch.testing.assert_close(param.data, reference_lm.get_parameter(name), rtol=0, atol=0)

    input_ids = torch.randint(0, 256, (2, 11))
    attention_mask = torch.ones(2, 11, dtype=torch.long)
    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        actual = probe.text_encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ── sharding (te_tp_size == 2) ──


@pytest.mark.parametrize("rank", [0, 1])
def test_megatron_shard_streams_into_rank_slice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank: int
) -> None:
    _mock_tp(monkeypatch, size=2, rank=rank)
    reference = _tiny_text_model()
    checkpoint = tmp_path / "text_encoder.safetensors"
    _write_checkpoint(reference, checkpoint)
    original = {name: param.detach().clone() for name, param in reference.named_parameters()}

    probe = _make_probe(te_tp_size=2)
    probe.text_encoder = probe._init_text_encoder(_tiny_config(), torch.float32)

    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    lm = probe.text_encoder.language_model
    assert isinstance(lm.layers[0].linear_attn.in_proj_qkv, ColumnParallelLinear)
    assert lm.layers[0].linear_attn.in_proj_qkv.gather_output is True
    assert isinstance(lm.layers[1].self_attn.o_proj, RowParallelLinear)

    probe.load_weights(_iter_checkpoint(checkpoint, "text_encoder."))

    # Rank-local shard contents match the reference checkpoint slices.
    w_qkv = original["language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    per = w_qkv.shape[0] // 2
    torch.testing.assert_close(lm.layers[0].linear_attn.in_proj_qkv.weight.data, w_qkv.narrow(0, rank * per, per))
    w_down = original["language_model.layers.1.mlp.down_proj.weight"]
    per_in = w_down.shape[1] // 2
    torch.testing.assert_close(lm.layers[1].mlp.down_proj.weight.data, w_down.narrow(1, rank * per_in, per_in))
    # Embeddings and norms are replicated full-size.
    torch.testing.assert_close(
        lm.embed_tokens.weight.data, original["language_model.embed_tokens.weight"]
    )


def test_megatron_shard_rejects_uncovered_linear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    class BrokenPlanModel(Qwen3_5Model):
        @property
        def tp_plan(self):
            plan = dict(super().tp_plan)
            del plan["language_model.layers.*.self_attn.q_proj"]
            return plan

    _mock_tp(monkeypatch, size=2, rank=0)
    torch.manual_seed(0)
    model = BrokenPlanModel(_tiny_config()).eval()

    with pytest.raises(ValueError, match="tp_plan did not cover"):
        apply_text_encoder_tp_plan(model)


def test_streamed_shard_leaves_no_full_size_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Streaming into the rank slice must not retain any full-size checkpoint tensor."""
    import gc

    _mock_tp(monkeypatch, size=2, rank=0)
    reference = _tiny_text_model()
    checkpoint = tmp_path / "text_encoder.safetensors"
    _write_checkpoint(reference, checkpoint)
    full_numel = reference.language_model.layers[0].linear_attn.in_proj_qkv.weight.numel()

    probe = _make_probe(te_tp_size=2)
    probe.text_encoder = probe._init_text_encoder(_tiny_config(), torch.float32)
    # Only half of the streamed tensor is kept.
    assert probe.text_encoder.language_model.layers[0].linear_attn.in_proj_qkv.weight.numel() == full_numel // 2

    gc.collect()
    before = {id(t) for t in gc.get_objects() if isinstance(t, torch.Tensor)}
    probe.load_weights(_iter_checkpoint(checkpoint, "text_encoder."))
    gc.collect()
    leaked = [
        tuple(t.shape)
        for t in gc.get_objects()
        if isinstance(t, torch.Tensor) and id(t) not in before and t.numel() == full_numel
    ]
    assert leaked == []


# ── pipeline integration ──


class _DummyTransformer(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


def _build_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    te_tp_size: int,
    dit_tp_size: int = 1,
    rank: int = 0,
) -> MOSSImagePipeline:
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    monkeypatch.setattr(pipeline_mod.AutoConfig, "from_pretrained", lambda *a, **k: _tiny_config())

    def fake_load(factory, model, **kwargs):
        assert kwargs.get("subfolder") == "vae"  # the text encoder no longer loads this way
        return nn.Linear(2, 2)

    monkeypatch.setattr(pipeline_mod, "from_pretrained_with_prefetch", fake_load)
    monkeypatch.setattr(pipeline_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.MOSSImageScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "get_transformer_config_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_mod, "MOSSImageTransformer", _DummyTransformer)
    monkeypatch.setattr(pipeline_mod, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_mod, "get_tensor_model_parallel_world_size", lambda: dit_tp_size)
    monkeypatch.setattr(pipeline_mod, "get_tensor_model_parallel_rank", lambda: rank)
    _mock_tp(monkeypatch, size=dit_tp_size, rank=rank)

    parallel = DiffusionParallelConfig(text_encoder_tp_size=te_tp_size)
    od_config = OmniDiffusionConfig(model="moss-image", dtype=torch.float16, parallel_config=parallel)
    return MOSSImagePipeline(od_config=od_config)


def test_te_tp1_leaves_encoder_and_plan_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch, te_tp_size=1)

    assert pipeline.text_encoder_tp_size == 1
    layer = pipeline.text_encoder.language_model.layers[1]
    assert isinstance(layer.self_attn.q_proj, nn.Linear)
    # Skeleton is built on CPU with fp16 params; weights stream in later.
    param = next(pipeline.text_encoder.parameters())
    assert param.device.type == "cpu" and param.dtype == torch.float16
    plan = get_offload_plan(pipeline)
    assert "text_encoder" in plan.encoder_dlo_weight_replication


def test_te_tp2_shards_encoder_and_drops_dlo_replication(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline(monkeypatch, te_tp_size=2, dit_tp_size=2, rank=0)

    layer = pipeline.text_encoder.language_model.layers[0]
    assert not isinstance(layer.linear_attn.in_proj_qkv, nn.Linear)  # replaced by a parallel linear
    assert layer.linear_attn.in_proj_qkv.tp_size == 2
    assert getattr(pipeline.text_encoder, "visual", None) is None
    plan = get_offload_plan(pipeline)
    # Sharded TE weights differ per rank: AllGather transfer must be rejected.
    assert plan.encoder_dlo_weight_replication == frozenset()


def test_te_tp_rejects_mismatched_dit_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    # text_encoder_tp_size=2 but the DiT TP group has size 1.
    with pytest.raises(ValueError, match="tensor-parallel size"):
        _build_pipeline(monkeypatch, te_tp_size=2, dit_tp_size=1)


# ── real collective path (gloo, two CPU processes) ──


@pytest.mark.skipif(not _WORKER.is_file(), reason="dist worker helper missing")
def test_te_megatron_tp_forward_gloo_two_ranks(tmp_path: Path) -> None:
    """Two real gloo ranks load-and-shard from a checkpoint; both match full forward."""
    import vllm_omni

    checkpoint = tmp_path / "text_encoder.safetensors"
    _write_checkpoint(_tiny_text_model(), checkpoint)

    repo_root = str(Path(vllm_omni.__path__[0]).resolve().parent)
    env = {
        **os.environ,
        "TE_TP_STORE": str(tmp_path / "gloo_store"),
        "TE_TP_CHECKPOINT": str(checkpoint),
    }
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
        assert "TE_MEGATRON_TP_DIST_OK" in out, out[-2000:]
