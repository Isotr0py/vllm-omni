# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Protocol-level tests for the Rust-frontend façade."""

from __future__ import annotations

from types import SimpleNamespace

import msgspec
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest, FinishReason
from vllm.v1.serial_utils import MsgpackDecoder

from tests.rust_frontend.conftest import RUST_REQUEST_ARRAY_LEN
from vllm_omni.entrypoints.rust_frontend.bridge import (
    MOCK_ECHO_TOKEN_IDS,
    build_token_frame,
    build_utility_frame,
)
from vllm_omni.entrypoints.rust_frontend.protocol import (
    _RUST_OUTPUT_FIELDS,
    _RUST_OUTPUTS_FIELDS,
    REQUIRED_READY_KEYS,
    VALID_READY_DTYPES,
    decode_add_request,
    mock_ready_dict,
    ready_dict_from_vllm_config,
)


def test_mock_ready_dict_has_exactly_required_keys_and_valid_dtype():
    ready = mock_ready_dict()
    assert set(ready) == set(REQUIRED_READY_KEYS)
    assert ready["dtype"] in VALID_READY_DTYPES


def test_ready_dict_from_vllm_config_covers_required_keys():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096,
            dtype="bfloat16",
            enable_sleep_mode=False,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=512,
            block_size=16,
            mamba_block_size=None,
            kv_cache_size_tokens=None,
            kv_cache_max_concurrency=None,
        ),
        parallel_config=SimpleNamespace(
            world_size=1,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=128,
            max_num_batched_tokens=4096,
        ),
        lora_config=None,
        instance_id="test-instance",
    )
    ready = ready_dict_from_vllm_config(vllm_config, engine_index=0)
    assert REQUIRED_READY_KEYS <= ready.keys()
    assert ready["dtype"] == "bfloat16"
    assert ready["dtype"] in VALID_READY_DTYPES
    assert ready["max_model_len"] == 4096
    assert ready["data_parallel_rank"] == 0
    assert ready["supports_lora"] is False
    assert ready["max_loras"] == 0


def test_token_frame_roundtrips_through_vllm_decoder():
    frame = build_token_frame(0, "req-1", [9707, 11])
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(frame)
    assert decoded.engine_index == 0
    assert len(decoded.outputs) == 1
    assert decoded.outputs[0].request_id == "req-1"
    assert decoded.outputs[0].new_token_ids == [9707, 11]
    assert decoded.outputs[0].finish_reason is None
    assert decoded.utility_output is None
    assert decoded.scheduler_stats is None


def test_final_token_frame_carries_finish_reason():
    frame = build_token_frame(0, "req-1", [], FinishReason.STOP)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(frame)
    assert decoded.outputs[0].finish_reason == FinishReason.STOP
    assert decoded.finished_requests == {"req-1"}


def test_utility_frame_is_never_mixed_with_outputs():
    frame = build_utility_frame(0, 42, "not implemented")
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(frame)
    assert decoded.utility_output is not None
    assert decoded.utility_output.call_id == 42
    assert decoded.utility_output.failure_message == "not implemented"
    assert decoded.utility_output.result is None
    assert decoded.outputs == []
    assert decoded.scheduler_stats is None


def test_token_frame_matches_rust_wire_shape():
    """Encoded frames must use the Rust client's exact array layouts."""
    frame = build_token_frame(0, "req-1", [9707, 11])
    raw = msgspec.msgpack.decode(frame[0] if isinstance(frame, list) else frame)
    assert isinstance(raw, list) and len(raw) == len(_RUST_OUTPUTS_FIELDS)
    assert isinstance(raw[1], list) and len(raw[1]) == 1
    assert len(raw[1][0]) == len(_RUST_OUTPUT_FIELDS)


def test_rust_shaped_21_element_request_array_decodes(request_payload_factory):
    payload = request_payload_factory(
        "req-1",
        list(MOCK_ECHO_TOKEN_IDS),
        client_index=7,
        session_id="session-1",
    )
    request = decode_add_request(payload)
    assert isinstance(request, EngineCoreRequest)
    assert request.request_id == "req-1"
    assert request.prompt_token_ids == list(MOCK_ECHO_TOKEN_IDS)
    assert request.sampling_params is not None
    assert request.sampling_params.max_tokens == 16
    assert request.sampling_params.seed == 42
    assert request.client_index == 7
    # Trailing field present in 0.26.0's dataclass; older decoders drop it.
    if hasattr(request, "session_id"):
        assert request.session_id == "session-1"


def test_rust_request_array_length_is_pinned():
    # Guard the test helper against drifting from the Rust wire shape.
    import msgspec

    from tests.rust_frontend.conftest import make_engine_core_request_payload

    payload = make_engine_core_request_payload("req-1", [1])
    decoded = msgspec.msgpack.decode(payload)
    assert len(decoded) == RUST_REQUEST_ARRAY_LEN == 21
