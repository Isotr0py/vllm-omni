# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SSE streaming for /v1/images/generations (denoise progress + previews).

Kept in a dedicated module so it stays importable without the full API
server stack (api_server pulls in uvloop and the model registry).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationImageChunk,
    ImageGenerationPreviewChunk,
    ImageGenerationProgressChunk,
    ImageGenerationRequest,
    ImageGenerationStreamError,
)
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)


async def stream_image_generation_sse(
    *,
    engine_client: Any,
    gen_params: Any,
    stage_configs: list[Any],
    model_name: str,
    request: ImageGenerationRequest,
    generate_kwargs: dict[str, Any],
) -> AsyncIterator[str]:
    """Yield SSE chunks for a streamed /v1/images/generations request.

    Event sequence: zero or more image.generation.chunk progress/preview
    chunks (only when the diffusion stage runs with step execution and the
    pipeline supports previews), then one image chunk, then [DONE]. The
    final image chunk is always emitted, so the stream degrades gracefully
    on servers whose engine has no step execution enabled.
    """
    created = int(time.time())
    emitted_image = False
    # Imported lazily: helpers/diffusion pull in the app-state and async
    # engine stacks, which this module otherwise does not need at import time.
    from vllm_omni.entrypoints.openai.image_api_utils import encode_image_base64_with_compression
    from vllm_omni.entrypoints.openai.images.helpers import (
        _choose_output_format,
        _extract_images_from_result,
        _generated_size_str,
    )

    output_format = _choose_output_format(request.output_format or "png", None)
    from vllm_omni.entrypoints.openai.diffusion import _iter_with_async_omni

    try:
        async for output in _iter_with_async_omni(
            engine_client,
            gen_params,
            stage_configs,
            **generate_kwargs,
        ):
            custom_output = getattr(output, "custom_output", None) or {}
            progress = custom_output.get("progress") if isinstance(custom_output, dict) else None
            if progress and not getattr(output, "finished", False):
                step = int(progress.get("step", 0))
                total = int(progress.get("total", 0))
                progress_chunk = ImageGenerationProgressChunk(
                    step=step,
                    total=total,
                    created=created,
                    model=model_name,
                )
                yield f"data: {progress_chunk.model_dump_json()}\n\n"
                preview_b64 = custom_output.get("preview_b64")
                if preview_b64:
                    preview_chunk = ImageGenerationPreviewChunk(
                        step=step,
                        total=total,
                        data=[ImageData(b64_json=preview_b64)],
                        created=created,
                        model=model_name,
                    )
                    yield f"data: {preview_chunk.model_dump_json()}\n\n"
                continue
            if not getattr(output, "finished", False):
                continue
            images = _extract_images_from_result(output)
            if not images:
                continue
            image_data = [
                ImageData(
                    b64_json=encode_image_base64_with_compression(
                        image,
                        format=output_format,
                        output_compression=request.output_compression,
                    ),
                    revised_prompt=None,
                )
                for image in images
            ]
            image_chunk = ImageGenerationImageChunk(
                data=image_data,
                output_format=output_format,
                size=_generated_size_str(images, request.size),
                created=created,
                model=model_name,
                metrics=(
                    _build_image_stream_metrics(
                        response_metrics=getattr(output, "metrics", None),
                        stage_durations=getattr(output, "stage_durations", None),
                        peak_memory_mb=getattr(output, "peak_memory_mb", None),
                    )
                    if request.return_stage_metrics
                    else None
                ),
            )
            yield f"data: {image_chunk.model_dump_json()}\n\n"
            emitted_image = True
        if not emitted_image:
            raise RuntimeError("Streaming image generation completed without a final image output.")
    except OmniClientError as exc:
        logger.info("Client error during streaming image generation: %s", exc)
        error_chunk = ImageGenerationStreamError(
            created=created,
            model=model_name,
            error={"message": exc.message, "type": exc.error_type, "code": exc.status_code},
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    except Exception as exc:
        logger.exception("Streaming image generation failed: %s", exc)
        error_chunk = ImageGenerationStreamError(
            created=created,
            model=model_name,
            error={"message": str(exc), "type": "server_error", "code": 500},
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def _build_image_stream_metrics(
    *,
    response_metrics: Any,
    stage_durations: Any,
    peak_memory_mb: Any,
) -> dict[str, Any]:
    """Merge detailed stage metrics with the legacy image timing fields."""
    metrics = dict(response_metrics) if isinstance(response_metrics, dict) else {}
    metrics["stage_durations"] = stage_durations or None
    metrics["peak_memory_mb"] = float(peak_memory_mb) if peak_memory_mb else None
    return metrics
