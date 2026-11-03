# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Denoise progress/preview chunk construction for SSE image streaming.

Kept separate from the model runner so the logic is importable (and testable)
without the runner's heavy dependency chain.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput

logger = init_logger(__name__)


def build_progress_output(
    pipeline: Any,
    *,
    extra_args: dict[str, Any],
    step_index: int,
    total: int,
    latents: torch.Tensor | None,
    request_id: str,
    is_primary: bool,
) -> DiffusionOutput | None:
    """Build a non-terminal progress/preview chunk, or None when not requested.

    Emitted only when the request opted in via ``extra_args["stream_progress"]``
    (the API layer sets it for ``stream=true`` image requests) and only on the
    primary rank. Preview decode is delegated to the pipeline's optional
    ``decode_preview`` hook; preview failures never fail the request.
    """
    if not extra_args.get("stream_progress") or not is_primary:
        return None
    custom_output: dict[str, Any] = {"progress": {"step": step_index, "total": total}}
    preview_interval = int(extra_args.get("preview_interval", 0) or 0)
    if preview_interval > 0 and step_index % preview_interval == 0 and latents is not None:
        decode_preview = getattr(pipeline, "decode_preview", None)
        if callable(decode_preview):
            try:
                preview_b64 = decode_preview(
                    latents,
                    max_side=int(extra_args.get("preview_max_side", 256) or 256),
                )
                if preview_b64:
                    custom_output["preview_b64"] = preview_b64
            except Exception:
                logger.warning("Preview decode failed for request %s", request_id, exc_info=True)
    return DiffusionOutput(output=None, finished=False, custom_output=custom_output)
