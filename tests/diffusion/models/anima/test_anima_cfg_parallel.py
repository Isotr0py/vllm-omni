# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for AnimaPipeline CFG parallel integration.

These exercise ``AnimaPipeline.diffuse`` against a stub transformer and a real
FlowMatchEulerDiscreteScheduler. The single-rank path works because the shared
CFG mixin treats an uninitialized CFG group as world size 1; the two-rank path
is mocked like tests/diffusion/models/qwen_image_21/test_qwen_image_21_cfg_parallel.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torch import nn

import vllm_omni.diffusion.distributed.cfg_parallel as cfg_dist
from vllm_omni.diffusion.models.anima import pipeline_anima as pipeline_mod
from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline
from vllm_omni.diffusion.offloader.config import OffloadStrategy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_STEPS = 4
TRUE_CFG_SCALE = 4.0
LATENT_SHAPE = (1, 4, 1, 2, 2)


class _StubTransformer:
    """Stand-in for AnimaTransformer3DModel: returns per-branch constant noise."""

    dtype = torch.float32

    def __init__(
        self,
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor | None,
        fill: float,
        events: list[str] | None = None,
    ) -> None:
        self.prompt_embeds = prompt_embeds
        self.negative_embeds = negative_embeds
        self.positive_fill = fill
        self.negative_fill = fill + 1.0
        self.events = events
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        assert return_dict is False
        branch = "pos" if encoder_hidden_states is self.prompt_embeds else "neg"
        assert encoder_hidden_states is self.prompt_embeds or encoder_hidden_states is self.negative_embeds
        if self.events is not None:
            # Logged at call time, after any ControlNet multiplier gating.
            self.events.append("transformer")
        fill = self.positive_fill if branch == "pos" else self.negative_fill
        self.calls.append(
            {
                "branch": branch,
                "timestep": timestep,
                "padding_mask": padding_mask,
            }
        )
        return (torch.full_like(hidden_states, fill),)


class _StubControlNet:
    """Records set_multiplier calls; cond_emb is set so the pipeline gates it."""

    def __init__(self, events: list[str]) -> None:
        self.cond_emb = torch.zeros(1)
        self.events = events

    def set_multiplier(self, value: float) -> None:
        self.events.append(f"multiplier:{value}")


def _make_pipeline(
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    *,
    events: list[str] | None = None,
) -> tuple[AnimaPipeline, _StubTransformer, FlowMatchEulerDiscreteScheduler]:
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    nn.Module.__init__(pipeline)
    pipeline._interrupt = False
    pipeline.transformer = _StubTransformer(prompt_embeds, negative_embeds, fill=1.0, events=events)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    pipeline.scheduler.set_timesteps(NUM_STEPS, device="cpu")
    pipeline.controlnet = None
    pipeline._controlnet_strength = 0.7
    pipeline._controlnet_start = 0.0
    pipeline._controlnet_end = 1.0
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, pipeline.transformer, pipeline.scheduler


def _diffuse_inputs() -> dict[str, Any]:
    torch.manual_seed(0)
    return {
        "prompt_embeds": torch.randn(1, 3, 8),
        "negative_prompt_embeds": torch.randn(1, 3, 8),
        "latents": torch.randn(*LATENT_SHAPE),
        "padding_mask": torch.zeros(1, 1, 4, 4),
        "timesteps": None,  # filled from the pipeline scheduler
        "do_true_cfg": True,
        "true_cfg_scale": TRUE_CFG_SCALE,
    }


def _reference_latents(
    scheduler_config: Any,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
) -> torch.Tensor:
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(NUM_STEPS, device="cpu")
    out = latents.clone()
    for t in scheduler.timesteps:
        out = scheduler.step(noise_pred, t, out, return_dict=False)[0]
    return out


def test_cfg_parallel_size_no_longer_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_mod, "resolve_offload_strategy", lambda _cfg: OffloadStrategy.NONE)
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cfg_parallel_size=2, sequence_parallel_size=1, use_hsdp=False),
        cache_backend="none",
        quantization_config=None,
    )

    pipeline._raise_unsupported_features()

    pipeline.od_config.parallel_config.sequence_parallel_size = 2
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        pipeline._raise_unsupported_features()


def test_diffuse_sequential_cfg_combines_both_branches() -> None:
    inputs = _diffuse_inputs()
    pipeline, transformer, scheduler = _make_pipeline(
        inputs["prompt_embeds"], inputs["negative_prompt_embeds"]
    )
    inputs["timesteps"] = scheduler.timesteps

    latents = pipeline.diffuse(**inputs)

    # Both branches run per step, with padding_mask and /1000-scaled timesteps.
    assert len(transformer.calls) == 2 * NUM_STEPS
    for step, pair in enumerate(zip(transformer.calls[::2], transformer.calls[1::2])):
        pos_call, neg_call = pair
        assert pos_call["branch"] == "pos"
        assert neg_call["branch"] == "neg"
        assert pos_call["padding_mask"] is inputs["padding_mask"]
        assert neg_call["padding_mask"] is inputs["padding_mask"]
        torch.testing.assert_close(pos_call["timestep"], scheduler.timesteps[step].expand(1) / 1000.0)

    expected_noise = torch.full(
        LATENT_SHAPE,
        transformer.negative_fill + TRUE_CFG_SCALE * (transformer.positive_fill - transformer.negative_fill),
    )
    torch.testing.assert_close(latents, _reference_latents(scheduler.config, inputs["latents"], expected_noise))


def test_diffuse_without_cfg_runs_positive_branch_only() -> None:
    inputs = _diffuse_inputs()
    pipeline, transformer, scheduler = _make_pipeline(
        inputs["prompt_embeds"], inputs["negative_prompt_embeds"]
    )
    inputs["timesteps"] = scheduler.timesteps
    inputs["do_true_cfg"] = False

    latents = pipeline.diffuse(**inputs)

    assert [call["branch"] for call in transformer.calls] == ["pos"] * NUM_STEPS
    expected_noise = torch.full(LATENT_SHAPE, transformer.positive_fill)
    torch.testing.assert_close(latents, _reference_latents(scheduler.config, inputs["latents"], expected_noise))


def test_diffuse_true_cfg_without_negative_embeds_degenerates_to_positive_only() -> None:
    inputs = _diffuse_inputs()
    pipeline, transformer, scheduler = _make_pipeline(inputs["prompt_embeds"], None)
    inputs["negative_prompt_embeds"] = None
    inputs["timesteps"] = scheduler.timesteps

    latents = pipeline.diffuse(**inputs)

    assert [call["branch"] for call in transformer.calls] == ["pos"] * NUM_STEPS
    expected_noise = torch.full(LATENT_SHAPE, transformer.positive_fill)
    torch.testing.assert_close(latents, _reference_latents(scheduler.config, inputs["latents"], expected_noise))


def test_diffuse_controlnet_multiplier_gates_before_each_branch() -> None:
    events: list[str] = []
    inputs = _diffuse_inputs()
    pipeline, _, scheduler = _make_pipeline(
        inputs["prompt_embeds"], inputs["negative_prompt_embeds"], events=events
    )
    pipeline.controlnet = _StubControlNet(events)
    # Only the middle sigma window is active.
    pipeline._controlnet_start = 0.25
    pipeline._controlnet_end = 0.75
    inputs["timesteps"] = scheduler.timesteps

    pipeline.diffuse(**inputs)

    sigmas = scheduler.sigmas
    sigma_start = AnimaPipeline._sigma_at_percent(0.25, sigmas)
    sigma_end = AnimaPipeline._sigma_at_percent(0.75, sigmas)
    expected_events: list[str] = []
    for step in range(NUM_STEPS):
        active = sigma_end <= float(sigmas[step]) <= sigma_start
        expected_events.append(f"multiplier:{0.7 if active else 0.0}")
        # The gate is set before both CFG branches of the step.
        expected_events += ["transformer", "transformer"]

    assert events == expected_events
    # Sanity: the window actually toggles within these steps.
    assert "multiplier:0.7" in events
    assert "multiplier:0.0" in events


@pytest.mark.parametrize("rank", [0, 1])
def test_diffuse_cfg_parallel_dispatches_one_branch_per_rank(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    inputs = _diffuse_inputs()
    pipeline, transformer, scheduler = _make_pipeline(
        inputs["prompt_embeds"], inputs["negative_prompt_embeds"]
    )
    inputs["timesteps"] = scheduler.timesteps

    pos_pred = torch.full(LATENT_SHAPE, transformer.positive_fill)
    neg_pred = torch.full(LATENT_SHAPE, transformer.negative_fill)

    class _Gather:
        def all_gather(
            self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
        ) -> list[torch.Tensor]:
            del dim
            assert separate_tensors
            return [pos_pred.clone(), neg_pred.clone()]

    monkeypatch.setattr(cfg_dist, "_get_cfg_world_size_or_one", lambda: 2)
    monkeypatch.setattr(cfg_dist, "get_classifier_free_guidance_rank", lambda: rank)
    monkeypatch.setattr(cfg_dist, "get_cfg_group", lambda: _Gather())

    latents = pipeline.diffuse(**inputs)

    # Each rank computes only its own branch, but the gathered combine is identical.
    expected_branch = "pos" if rank == 0 else "neg"
    assert [call["branch"] for call in transformer.calls] == [expected_branch] * NUM_STEPS
    expected_noise = torch.full(
        LATENT_SHAPE,
        transformer.negative_fill + TRUE_CFG_SCALE * (transformer.positive_fill - transformer.negative_fill),
    )
    torch.testing.assert_close(latents, _reference_latents(scheduler.config, inputs["latents"], expected_noise))


def test_check_cfg_parallel_validity_passes_without_cfg_group() -> None:
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    nn.Module.__init__(pipeline)

    # No CFG group initialized in tests -> treated as single rank, always valid.
    assert pipeline.check_cfg_parallel_validity(TRUE_CFG_SCALE, True)
    assert pipeline.check_cfg_parallel_validity(1.0, False)
