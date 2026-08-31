# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""End-to-end tests of the façade in ``--mock-echo`` mode.

The test process plays the Rust frontend: it binds the ROUTER input and PULL
output sockets, waits for the façade's registration, and then drives
ADD/ABORT/UTILITY traffic, asserting on the wire-level shapes.
"""

from __future__ import annotations

import signal
import subprocess
import sys

import msgspec
import pytest
import zmq
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.serial_utils import MsgpackDecoder

from tests.rust_frontend.conftest import make_engine_core_request_payload
from vllm_omni.entrypoints.rust_frontend.bridge import MOCK_ECHO_TOKEN_IDS
from vllm_omni.entrypoints.rust_frontend.protocol import (
    REQUIRED_READY_KEYS,
    VALID_READY_DTYPES,
)

_RECV_TIMEOUT_MS = 30_000
ENGINE_INDEX_IDENTITY = b"\x00\x00"


def _recv_frames(sock: zmq.Socket, timeout_ms: int = _RECV_TIMEOUT_MS) -> list[bytes]:
    if not sock.poll(timeout_ms):
        raise TimeoutError("timed out waiting for a ZMQ message")
    return sock.recv_multipart()


def _recv_outputs(pull: zmq.Socket, timeout_ms: int = _RECV_TIMEOUT_MS) -> EngineCoreOutputs:
    (frame,) = _recv_frames(pull, timeout_ms)
    assert frame != b"ENGINE_CORE_DEAD"
    return MsgpackDecoder(EngineCoreOutputs).decode(frame)


def _collect_until_finished(
    pull: zmq.Socket, request_id: str, timeout_ms: int = _RECV_TIMEOUT_MS
) -> tuple[list[int], EngineCoreOutputs]:
    """Collect token deltas for a request until its terminal output."""
    tokens: list[int] = []
    while True:
        outputs = _recv_outputs(pull, timeout_ms)
        assert outputs.engine_index == 0
        assert outputs.utility_output is None
        for output in outputs.outputs:
            if output.request_id != request_id:
                continue
            tokens.extend(output.new_token_ids)
            if output.finish_reason is not None:
                return tokens, outputs


@pytest.fixture
def facade():
    """Launch the mock-echo façade; yield (process, router, pull, identity)."""
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind("tcp://127.0.0.1:0")
    pull = ctx.socket(zmq.PULL)
    pull.bind("tcp://127.0.0.1:0")
    input_address = router.getsockopt_string(zmq.LAST_ENDPOINT)
    output_address = pull.getsockopt_string(zmq.LAST_ENDPOINT)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.rust_frontend",
            "--input-address",
            input_address,
            "--output-address",
            output_address,
            "--mock-echo",
        ],
    )
    try:
        identity, payload = _recv_frames(router)
        yield proc, router, pull, identity, payload
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        router.close()
        pull.close()
        ctx.term()


def test_registration_dict_has_required_keys(facade):
    _, _, _, identity, payload = facade
    assert identity == ENGINE_INDEX_IDENTITY
    ready = msgspec.msgpack.decode(payload)
    assert REQUIRED_READY_KEYS <= ready.keys()
    assert ready["dtype"] in VALID_READY_DTYPES


def test_add_streams_token_deltas_then_finishes(facade):
    _, router, pull, identity, _ = facade
    payload = make_engine_core_request_payload("req-add", [1, 2, 3])
    router.send_multipart([identity, b"\x00", payload])

    tokens, final = _collect_until_finished(pull, "req-add")
    assert tokens == MOCK_ECHO_TOKEN_IDS
    final_output = next(o for o in final.outputs if o.request_id == "req-add")
    assert final_output.finish_reason == FinishReason.STOP
    assert "req-add" in (final.finished_requests or set())


def test_abort_stops_in_flight_request(facade):
    _, router, pull, identity, _ = facade
    payload = make_engine_core_request_payload("req-abort", [1, 2, 3])
    router.send_multipart([identity, b"\x00", payload])

    # Wait for the first delta, then abort while later deltas are pending.
    first = _recv_outputs(pull)
    assert any(o.request_id == "req-abort" for o in first.outputs)
    abort_payload = msgspec.msgpack.encode(["req-abort"])
    router.send_multipart([identity, b"\x01", abort_payload])

    _, final = _collect_until_finished(pull, "req-abort")
    final_output = next(o for o in final.outputs if o.request_id == "req-abort")
    assert final_output.finish_reason == FinishReason.ABORT
    assert "req-abort" in (final.finished_requests or set())

    # No further outputs for the aborted request.
    assert not pull.poll(500)


def test_utility_call_replies_not_implemented(facade):
    _, router, pull, identity, _ = facade
    payload = msgspec.msgpack.encode([0, 42, "is_sleeping", []])
    router.send_multipart([identity, b"\x03", payload])

    outputs = _recv_outputs(pull)
    assert outputs.utility_output is not None
    assert outputs.utility_output.call_id == 42
    assert outputs.utility_output.failure_message == "not implemented"
    assert outputs.utility_output.result is None
    assert outputs.outputs == []


def test_sigterm_sends_engine_core_dead(facade):
    proc, _, pull, _, _ = facade
    proc.send_signal(signal.SIGTERM)
    (frame,) = _recv_frames(pull)
    assert frame == b"ENGINE_CORE_DEAD"
    proc.wait(timeout=15)
    assert proc.returncode == 0
