# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SSE chunk-sequence tests for streamed /v1/images/generations.

The generator under test (``stream_image_generation_sse``) lazily imports the
engine iterator and image helpers, so tests inject lightweight fakes through
``sys.modules``; the chunk models and the SSE framing are the real code.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.protocol.images import (
    ImageGenerationImageChunk,
    ImageGenerationPreviewChunk,
    ImageGenerationProgressChunk,
    ImageGenerationRequest,
    ImageGenerationStreamError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_output(*, finished: bool, custom_output: dict | None = None, images: list | None = None) -> Any:
    return SimpleNamespace(
        finished=finished,
        custom_output=custom_output or {},
        images=images or [],
        metrics={"stage_metrics": {}},
        stage_durations={"denoise": 1.0},
        peak_memory_mb=128.0,
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch, outputs: list[Any] | Exception) -> None:
    """Inject fake diffusion-iterator and image-helper modules."""
    diffusion_mod = types.ModuleType("vllm_omni.entrypoints.openai.diffusion")

    async def fake_iter(engine_client, gen_params, stage_configs, **kwargs):
        if isinstance(outputs, Exception):
            raise outputs
        for output in outputs:
            yield output

    diffusion_mod._iter_with_async_omni = fake_iter
    monkeypatch.setitem(sys.modules, diffusion_mod.__name__, diffusion_mod)

    helpers_mod = types.ModuleType("vllm_omni.entrypoints.openai.images.helpers")
    helpers_mod._choose_output_format = lambda fmt, background: fmt or "png"
    helpers_mod._extract_images_from_result = lambda result: result.images
    helpers_mod._generated_size_str = lambda images, fallback: (
        f"{images[0].width}x{images[0].height}" if images else fallback
    )
    monkeypatch.setitem(sys.modules, helpers_mod.__name__, helpers_mod)


def _collect(request: ImageGenerationRequest) -> list[str]:
    from vllm_omni.entrypoints.openai.image_stream import stream_image_generation_sse

    async def run() -> list[str]:
        return [
            chunk
            async for chunk in stream_image_generation_sse(
                engine_client=None,
                gen_params=None,
                stage_configs=[SimpleNamespace()],
                model_name="moss-image",
                request=request,
                generate_kwargs={},
            )
        ]

    return asyncio.run(run())


def _parse(chunk: str) -> dict:
    assert chunk.startswith("data: ") and chunk.endswith("\n\n")
    return json.loads(chunk[len("data: ") : -2])


def _png_image() -> Image.Image:
    return Image.new("RGB", (64, 64), (128, 32, 200))


def test_progress_preview_image_done_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    preview_jpeg = base64.b64encode(b"\xff\xd8preview").decode()
    outputs = [
        _make_output(finished=False, custom_output={"progress": {"step": 5, "total": 50}}),
        _make_output(
            finished=False,
            custom_output={"progress": {"step": 10, "total": 50}, "preview_b64": preview_jpeg},
        ),
        _make_output(finished=True, images=[_png_image()]),
    ]
    _install_fakes(monkeypatch, outputs)

    chunks = _collect(ImageGenerationRequest(prompt="x", stream=True))

    # step-5 progress; step-10 progress + preview; final image; [DONE]
    assert len(chunks) == 5
    progress = ImageGenerationProgressChunk(**_parse(chunks[0]))
    assert (progress.object, progress.type, progress.step, progress.total) == (
        "image.generation.chunk",
        "progress",
        5,
        50,
    )
    progress10 = ImageGenerationProgressChunk(**_parse(chunks[1]))
    assert (progress10.step, progress10.total) == (10, 50)
    preview = ImageGenerationPreviewChunk(**_parse(chunks[2]))
    assert preview.type == "preview" and preview.step == 10
    assert base64.b64decode(preview.data[0].b64_json) == b"\xff\xd8preview"
    image_chunk = ImageGenerationImageChunk(**_parse(chunks[3]))
    assert image_chunk.type == "image" and image_chunk.output_format == "png"
    assert image_chunk.size == "64x64"
    # Round-trip: the final image is a real PNG.
    Image.open(io.BytesIO(base64.b64decode(image_chunk.data[0].b64_json))).verify()
    assert chunks[4] == "data: [DONE]\n\n"


def test_stream_without_step_execution_emits_final_image_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engines without step execution yield no intermediates: image + DONE only."""
    _install_fakes(monkeypatch, [_make_output(finished=True, images=[_png_image()])])

    chunks = _collect(ImageGenerationRequest(prompt="x", stream=True))

    assert ImageGenerationImageChunk(**_parse(chunks[0])).type == "image"
    assert chunks[1] == "data: [DONE]\n\n"


def test_mid_stream_failure_yields_error_chunk_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch, RuntimeError("engine exploded"))

    chunks = _collect(ImageGenerationRequest(prompt="x", stream=True))

    error = ImageGenerationStreamError(**_parse(chunks[0]))
    assert error.error["type"] == "server_error" and "engine exploded" in error.error["message"]
    assert chunks[1] == "data: [DONE]\n\n"


def test_stream_never_raises_and_ends_done_without_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream with no image output still terminates with error + [DONE]."""
    _install_fakes(monkeypatch, [])

    chunks = _collect(ImageGenerationRequest(prompt="x", stream=True))

    assert ImageGenerationStreamError(**_parse(chunks[0])).object == "error"
    assert chunks[-1] == "data: [DONE]\n\n"
