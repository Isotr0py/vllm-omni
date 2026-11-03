# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the SSE wire-format parser (no ComfyUI runtime required).

Loads comfyui_vllm_omni/utils/sse.py by file path because the plugin package
__init__ imports ComfyUI-only modules.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _load_sse():
    path = Path(__file__).resolve().parents[1] / "comfyui_vllm_omni" / "utils" / "sse.py"
    spec = importlib.util.spec_from_file_location("comfyui_vllm_omni_sse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sse = _load_sse()


def test_parse_progress_event() -> None:
    line = 'data: {"object":"image.generation.chunk","type":"progress","step":5,"total":50,"created":1,"model":"m"}\n'
    kind, payload = sse.parse_sse_line(line)
    assert kind == sse.EVENT_PROGRESS
    assert sse.extract_progress(payload) == (5, 50)


def test_parse_preview_event_with_b64() -> None:
    line = (
        'data: {"object":"image.generation.chunk","type":"preview","step":10,"total":50,"data":[{"b64_json":"QUJD"}]}'
    )
    kind, payload = sse.parse_sse_line(line)
    assert kind == sse.EVENT_PREVIEW
    assert sse.extract_preview_b64(payload) == "QUJD"
    assert sse.extract_progress(payload) == (10, 50)


def test_parse_final_image_event() -> None:
    line = 'data: {"object":"image.generation.chunk","type":"image","data":[{"b64_json":"AAAA"},{"b64_json":"BBBB"}]}'
    kind, payload = sse.parse_sse_line(line)
    assert kind == sse.EVENT_IMAGE
    assert sse.extract_images_b64(payload) == ["AAAA", "BBBB"]


def test_parse_error_event() -> None:
    line = (
        'data: {"object":"error","created":1,"model":"m","error":{"message":"boom","type":"server_error","code":500}}'
    )
    kind, payload = sse.parse_sse_line(line)
    assert kind == sse.EVENT_ERROR
    assert payload["error"]["message"] == "boom"


def test_parse_done_sentinel() -> None:
    assert sse.parse_sse_line("data: [DONE]\n\n") == (sse.EVENT_DONE, {})


def test_non_data_lines_are_ignored() -> None:
    assert sse.parse_sse_line("") is None
    assert sse.parse_sse_line("\n") is None
    assert sse.parse_sse_line(": heartbeat comment\n") is None
    assert sse.parse_sse_line("event: message\n") is None
    assert sse.parse_sse_line("data: not-json\n") is None
    assert sse.parse_sse_line('data: {"unrelated": true}\n') is None
