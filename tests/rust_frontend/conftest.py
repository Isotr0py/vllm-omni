# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared helpers for the rust_frontend tests."""

from __future__ import annotations

import time

import msgspec
import pytest

# The Rust client always serializes EngineCoreRequest as a full 21-element
# array (see rust/src/engine-core-client/src/protocol/request.rs).
RUST_REQUEST_ARRAY_LEN = 21

#: Map-shaped subset of SamplingParams as sent by the Rust client.
RUST_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 42,
    "max_tokens": 16,
    "min_tokens": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "stop_token_ids": [],
    "_all_stop_token_ids": [],
}


def make_engine_core_request_payload(
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params: dict | None = None,
    *,
    client_index: int = 0,
    priority: int = 0,
    session_id: str | None = None,
) -> bytes:
    """Encode a Rust-shaped (21-element array) EngineCoreRequest payload."""
    array = [
        request_id,
        list(prompt_token_ids),
        None,  # mm_features
        RUST_SAMPLING_PARAMS if sampling_params is None else sampling_params,
        None,  # pooling_params
        time.time(),  # arrival_time
        None,  # lora_request
        None,  # cache_salt
        None,  # data_parallel_rank
        None,  # prompt_embeds
        None,  # prompt_is_token_ids
        client_index,
        0,  # current_wave
        priority,
        None,  # trace_headers
        False,  # resumable
        None,  # external_req_id
        None,  # reasoning_ended
        None,  # reasoning_parser_kwargs
        False,  # abort_immediately
        session_id,
    ]
    assert len(array) == RUST_REQUEST_ARRAY_LEN
    return msgspec.msgpack.encode(array)


@pytest.fixture
def request_payload_factory():
    """Return :func:`make_engine_core_request_payload`."""
    return make_engine_core_request_payload
