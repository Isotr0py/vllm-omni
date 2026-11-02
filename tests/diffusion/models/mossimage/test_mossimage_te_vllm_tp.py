# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image text-encoder megatron TP (``text_encoder_tp_size``).

The HF checkpoint's own ``tp_plan`` drives the sharding: every covered
``nn.Linear`` becomes the matching vLLM parallel linear holding only this
rank's shard.  CPU coverage: shard shapes/content, plan coverage validation,
the DLO replication declaration drop, topology validation, and a real
two-process gloo forward equivalence check at the bottom.
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
from vllm_omni.diffusion.models.mossimage.encoder_vllm_tp import shard_text_encoder_megatron_tp
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_WORKER = Path(__file__).with_name("te_vllm_tp_dist_worker.py")


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


def _mock_tp(monkeypatch: pytest.MonkeyPatch, size: int, rank: int) -> None:
    """Point the vLLM TP accessors (used by the parallel linears) at a fake topology."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)


# ── sharding ──


@pytest.mark.parametrize("rank", [0, 1])
def test_megatron_shard_shapes_and_content(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    _mock_tp(monkeypatch, size=2, rank=rank)
    torch.manual_seed(0)
    model = _tiny_text_model()
    original = {name: param.detach().clone() for name, param in model.named_parameters()}

    shard_text_encoder_megatron_tp(model, rank, 2)

    assert getattr(model, "visual", None) is None  # vision tower dropped (text-only pipeline)
    lm = model.language_model
    assert lm.embed_tokens is not None  # embeddings are replicated, not sharded

    # Every plan-covered linear became the matching parallel class with this
    # rank's half of the checkpoint weight.
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    layer = lm.layers[0]  # linear_attention block: in_proj_*/out_proj are colwise_gather_output
    assert isinstance(layer.linear_attn.in_proj_qkv, ColumnParallelLinear)
    assert layer.linear_attn.in_proj_qkv.gather_output is True
    w0 = original[f"language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    per = w0.shape[0] // 2
    torch.testing.assert_close(layer.linear_attn.in_proj_qkv.weight.data, w0.narrow(0, rank * per, per))

    attn = lm.layers[1].self_attn  # full_attention block
    assert isinstance(attn.q_proj, ColumnParallelLinear)
    assert isinstance(attn.o_proj, RowParallelLinear)
    w_q = original["language_model.layers.1.self_attn.q_proj.weight"]
    per_q = w_q.shape[0] // 2
    torch.testing.assert_close(attn.q_proj.weight.data, w_q.narrow(0, rank * per_q, per_q))
    w_down = original["language_model.layers.1.mlp.down_proj.weight"]
    per_in = w_down.shape[1] // 2
    assert isinstance(lm.layers[1].mlp.down_proj, RowParallelLinear)
    torch.testing.assert_close(lm.layers[1].mlp.down_proj.weight.data, w_down.narrow(1, rank * per_in, per_in))

    # q/k norms are replicated_with_grad_allreduce: not Linear, untouched.
    assert isinstance(attn.q_norm, nn.RMSNorm) or type(attn.q_norm).__name__.endswith("RMSNorm")


def test_megatron_shard_rejects_uncovered_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    class BrokenPlanModel(Qwen3_5Model):
        @property
        def tp_plan(self):
            plan = dict(super().tp_plan)
            del plan["language_model.layers.*.self_attn.q_proj"]
            return plan

    _mock_tp(monkeypatch, size=2, rank=0)
    torch.manual_seed(0)
    config = _tiny_text_model().config
    torch.manual_seed(0)
    model = BrokenPlanModel(config).eval()

    with pytest.raises(ValueError, match="tp_plan did not cover"):
        shard_text_encoder_megatron_tp(model, 0, 2)


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

    def fake_load(factory, model, **kwargs):
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
    assert pipeline.text_encoder.visual is not None
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
    """Two real gloo ranks run the sharded encode; both must match the full forward."""
    import vllm_omni

    repo_root = str(Path(vllm_omni.__path__[0]).resolve().parent)
    env = {**os.environ, "TE_TP_STORE": str(tmp_path / "gloo_store")}
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
