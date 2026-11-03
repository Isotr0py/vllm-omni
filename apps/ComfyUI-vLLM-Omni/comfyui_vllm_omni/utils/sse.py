# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Minimal SSE parsing for vLLM-Omni image generation streams.

Standalone (no aiohttp/comfy imports) so the wire-format handling is unit
testable without a ComfyUI runtime.

Stream layout (server: vllm_omni.entrypoints.openai.image_stream):

    data: {"object":"image.generation.chunk","type":"progress","step":i,"total":N,...}
    data: {"object":"image.generation.chunk","type":"preview","step":i,"total":N,"data":[{"b64_json":...}]}
    data: {"object":"image.generation.chunk","type":"image","data":[{"b64_json":...}],...}
    data: {"object":"error","error":{...}}            # only on failure
    data: [DONE]
"""

import json
from typing import Any

# Event kinds returned by parse_sse_line.
EVENT_PROGRESS = "progress"
EVENT_PREVIEW = "preview"
EVENT_IMAGE = "image"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def parse_sse_line(line: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one SSE line from an image generation stream.

    Returns:
        (kind, payload) for ``data:`` lines — kind is one of EVENT_PROGRESS /
        EVENT_PREVIEW / EVENT_IMAGE / EVENT_ERROR / EVENT_DONE — or None for
        blank lines, comments, and non-data fields.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return (EVENT_DONE, {})
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("object") == "error" or ("error" in payload and "type" not in payload):
        return (EVENT_ERROR, payload)
    event_type = payload.get("type")
    if payload.get("object") == "image.generation.chunk" and event_type in (
        EVENT_PROGRESS,
        EVENT_PREVIEW,
        EVENT_IMAGE,
    ):
        return (event_type, payload)
    return None


def extract_progress(payload: dict[str, Any]) -> tuple[int, int]:
    """(step, total) from a progress or preview event payload."""
    return int(payload.get("step", 0)), int(payload.get("total", 0))


def extract_preview_b64(payload: dict[str, Any]) -> str | None:
    """The b64 JPEG from a preview event, if present."""
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        b64 = data[0].get("b64_json")
        return b64 if isinstance(b64, str) and b64 else None
    return None


def extract_images_b64(payload: dict[str, Any]) -> list[str]:
    """All b64 images from the final image event."""
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [item["b64_json"] for item in data if isinstance(item, dict) and item.get("b64_json")]
