# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for MOSS-Image denoise progress/preview outputs (SSE streaming).

Covers the runner-side gating (``build_progress_output``: opt-in via
extra_args, chunk attached on the primary rank, preview decode on EVERY rank
so patch-parallel VAE collectives stay in lockstep, preview cadence, preview
failures non-fatal) and the pipeline's cheap downscaled latent decode
(``decode_preview``).
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_LOCKSTEP_WORKER = Path(__file__).with_name("preview_lockstep_dist_worker.py")


class _PreviewVAE(nn.Module):
    """FLUX.2-klein VAE stand-in: decode returns a known RGB tensor."""

    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.bn = SimpleNamespace(running_var=torch.ones(4), running_mean=torch.zeros(4))
        self.config = SimpleNamespace(batch_norm_eps=1e-5)

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        # Pretend an 8x-upscaling decoder: fill with a mid-gray RGB image.
        b = z.shape[0]
        h, w = z.shape[-2] * 8, z.shape[-1] * 8
        return (torch.full((b, 3, h, w), 0.25),)


def _make_pipeline() -> MOSSImagePipeline:
    pipeline = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae = _PreviewVAE()
    pipeline.device = torch.device("cpu")
    return pipeline


def test_decode_preview_produces_downscaled_jpeg() -> None:
    pipeline = _make_pipeline()
    # 1024x1024 request -> 64x64 packed latent (bn channels); preview at max_side=256.
    latents = torch.randn(1, 4, 64, 64)

    b64 = pipeline.decode_preview(latents, max_side=256)

    assert b64 is not None
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    image.load()
    assert image.format == "JPEG"
    assert image.size == (256, 256)  # 64 -> pool 16 -> unpack 32 -> 8x decode 256


def test_decode_preview_small_latent_skips_pooling() -> None:
    pipeline = _make_pipeline()
    # 256x256 request -> 16x16 packed latent == preview grid already.
    b64 = pipeline.decode_preview(torch.randn(1, 4, 16, 16), max_side=256)
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert image.size == (256, 256)


def _progress(pipeline, extra_args: dict, *, step_index: int = 3, is_primary: bool = True):
    """Invoke the runner's progress-chunk helper without the runner's import chain."""
    from vllm_omni.diffusion.progress import build_progress_output

    return build_progress_output(
        pipeline,
        extra_args=extra_args,
        step_index=step_index,
        total=50,
        latents=torch.randn(1, 4, 16, 16),
        request_id="req-1",
        is_primary=is_primary,
    )


def test_progress_output_opt_in_and_primary_only() -> None:
    pipeline = _make_pipeline()

    # Not requested -> nothing.
    assert _progress(pipeline, {}) is None
    # Non-primary rank -> no chunk, but the decode still runs on it: the
    # decode may carry TP-group collectives (patch-parallel VAE), so every
    # rank must call it or the group deadlocks.
    calls = []
    pipeline.decode_preview = lambda latents, max_side=256: calls.append(1) or "qux"
    assert _progress(pipeline, {"stream_progress": True, "preview_interval": 1}, is_primary=False) is None
    assert len(calls) == 1

    output = _progress(pipeline, {"stream_progress": True})
    assert output is not None and output.finished is False and output.output is None
    assert output.custom_output["progress"] == {"step": 3, "total": 50}
    assert "preview_b64" not in output.custom_output  # no interval -> progress only


def test_preview_cadence_and_content() -> None:
    pipeline = _make_pipeline()
    extra = {"stream_progress": True, "preview_interval": 5}

    assert "preview_b64" not in _progress(pipeline, extra, step_index=3).custom_output
    output = _progress(pipeline, extra, step_index=5)
    b64 = output.custom_output["preview_b64"]
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert image.format == "JPEG" and image.size == (256, 256)


def test_preview_failure_is_not_fatal() -> None:
    pipeline = _make_pipeline()

    def exploding_preview(latents, max_side=256):
        raise RuntimeError("decode blew up")

    pipeline.decode_preview = exploding_preview

    output = _progress(pipeline, {"stream_progress": True, "preview_interval": 1}, step_index=5)
    assert output is not None
    assert output.custom_output["progress"] == {"step": 5, "total": 50}
    assert "preview_b64" not in output.custom_output


@pytest.mark.skipif(not _LOCKSTEP_WORKER.is_file(), reason="dist worker helper missing")
def test_preview_decode_participates_on_all_ranks(tmp_path: Path) -> None:
    """Two gloo ranks: a collectives-bearing decode_preview must not deadlock.

    Regression for the T4x2 hang: a primary-only preview decode hits the
    patch-parallel VAE wrapper's TP-group broadcast on rank 0 while rank 1
    waits for the next step — a permanent NCCL deadlock (GPU0 spinning, GPU1
    idle). The fix runs the decode on every rank and attaches the result only
    on the primary.
    """
    import vllm_omni

    repo_root = str(Path(vllm_omni.__path__[0]).resolve().parent)
    env = {**os.environ, "PREVIEW_TP_STORE": str(tmp_path / "gloo_store")}
    workers = []
    for rank in (0, 1):
        workers.append(
            subprocess.Popen(
                [sys.executable, str(_LOCKSTEP_WORKER), str(rank), repo_root],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    for rank, worker in enumerate(workers):
        out, _ = worker.communicate(timeout=90)
        assert worker.returncode == 0, f"rank {rank} failed:\n{out[-2000:]}"
        assert "PREVIEW_LOCKSTEP_DIST_OK" in out, out[-2000:]
