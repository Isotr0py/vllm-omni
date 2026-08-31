# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ZMQ façade impersonating a single vLLM v1 EngineCore process.

Wire layout (Bootstrapped mode — the Rust frontend binds a ROUTER input and
a PULL output at known addresses, no HELLO handshake):

- Input: a DEALER socket connected to ``--input-address`` whose identity is
  the engine index as 2-byte little-endian. The first message sent is the
  single-frame msgpack registration map (see ``protocol.py``). Requests then
  arrive as ``[type_byte, payload, *aux]`` frames (ADD/ABORT/UTILITY).
- Output: a PUSH socket connected to ``--output-address`` carrying
  single-frame msgpack ``EngineCoreOutputs``. On shutdown a lone raw
  ``b"ENGINE_CORE_DEAD"`` frame is sent before the socket is closed.

pyzmq sockets are not thread-safe, so each socket is confined to one thread:
the receiver thread owns the DEALER, the sender thread owns the PUSH and
drains a queue of already-encoded frames.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Protocol

import zmq
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

from vllm_omni.entrypoints.rust_frontend.bridge import build_utility_frame
from vllm_omni.entrypoints.rust_frontend.protocol import (
    REQUIRED_READY_KEYS,
    decode_abort_request,
    decode_add_request,
    decode_utility_request,
    encode_ready_dict,
)

logger = init_logger(__name__)

#: Sentinel raw frame sent on the PUSH socket when the engine dies.
ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

#: Internal queue item telling the sender thread to emit the dead sentinel.
_SENDER_STOP = object()

_RECV_POLL_MS = 100
_PUSH_LINGER_MS = 5000


class FacadeHandler(Protocol):
    """Backend interface consumed by :class:`EngineCoreFacade`."""

    def get_ready_dict(self) -> dict[str, Any]:
        """Return the registration payload (may block until the engine is up)."""
        ...

    def handle_add(self, request: Any) -> None:
        """Dispatch a decoded ``EngineCoreRequest``."""
        ...

    def handle_abort(self, request_ids: list[str]) -> None:
        """Abort in-flight requests by external request ID."""
        ...

    def shutdown(self) -> None:
        """Shut the backend down before the sockets are closed."""
        ...


class EngineCoreFacade:
    """Owns the ZMQ sockets and the ADD/ABORT/UTILITY dispatch."""

    def __init__(
        self,
        *,
        input_address: str,
        output_address: str,
        engine_index: int,
        handler: FacadeHandler,
        send_queue: queue.Queue,
    ):
        if not 0 <= engine_index <= 0xFFFF:
            raise ValueError(f"engine_index must fit in 2 bytes, got {engine_index}")
        self._input_address = input_address
        self._output_address = output_address
        self._engine_index = engine_index
        self._handler = handler
        self._send_queue = send_queue
        self._stop = threading.Event()
        self._ctx: zmq.Context | None = None
        self._dealer: zmq.Socket | None = None
        self._push: zmq.Socket | None = None
        self._receiver_thread: threading.Thread | None = None
        self._sender_thread: threading.Thread | None = None

    def start(self) -> None:
        """Connect the sockets, send the registration, start the threads."""
        self._ctx = zmq.Context()

        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.LINGER, _PUSH_LINGER_MS)
        self._push.connect(self._output_address)

        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.IDENTITY, self._engine_index.to_bytes(2, "little"))
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.connect(self._input_address)

        ready = self._handler.get_ready_dict()
        missing = REQUIRED_READY_KEYS - ready.keys()
        if missing:
            raise ValueError(f"ready dict is missing required keys: {sorted(missing)}")
        # Must be the first message on the DEALER socket.
        self._dealer.send(encode_ready_dict(ready))
        logger.info(
            "Registered as engine %d (input=%s, output=%s)",
            self._engine_index,
            self._input_address,
            self._output_address,
        )

        self._sender_thread = threading.Thread(target=self._send_loop, name="facade-sender", daemon=True)
        self._receiver_thread = threading.Thread(target=self._recv_loop, name="facade-receiver", daemon=True)
        self._sender_thread.start()
        self._receiver_thread.start()

    def shutdown(self) -> None:
        """Stop the receiver, drain the backend, emit ENGINE_CORE_DEAD."""
        self._stop.set()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=5.0)
        try:
            self._handler.shutdown()
        except Exception:
            logger.exception("Backend shutdown failed")
        # The PUSH linger (> 0) lets the dead sentinel flush before close.
        self._send_queue.put(_SENDER_STOP)
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=10.0)
        if self._dealer is not None:
            self._dealer.close()
        if self._push is not None:
            self._push.close()
        if self._ctx is not None:
            self._ctx.term()

    # ------------------------------------------------------------------
    # Sender thread (owns the PUSH socket)
    # ------------------------------------------------------------------

    def _send_loop(self) -> None:
        assert self._push is not None
        while True:
            item = self._send_queue.get()
            if item is _SENDER_STOP:
                self._push.send(ENGINE_CORE_DEAD)
                return
            try:
                self._push.send_multipart(item)
            except Exception:
                logger.exception("Failed to send output frame")

    # ------------------------------------------------------------------
    # Receiver thread (owns the DEALER socket)
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._dealer is not None
        poller = zmq.Poller()
        poller.register(self._dealer, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(_RECV_POLL_MS):
                continue
            try:
                frames = self._dealer.recv_multipart()
            except zmq.ZMQError:
                if self._stop.is_set():
                    return
                logger.exception("Failed to receive on input socket")
                continue
            try:
                self._dispatch(frames)
            except Exception:
                logger.exception("Failed to dispatch input frames")

    def _dispatch(self, frames: list[bytes]) -> None:
        if len(frames) < 2:
            logger.warning("Ignoring malformed message with %d frame(s)", len(frames))
            return
        type_byte, payload = frames[0], frames[1]
        aux_frames = tuple(frames[2:])

        if type_byte == EngineCoreRequestType.ADD.value:
            self._handler.handle_add(decode_add_request(payload, aux_frames))
        elif type_byte == EngineCoreRequestType.ABORT.value:
            self._handler.handle_abort(decode_abort_request(payload))
        elif type_byte == EngineCoreRequestType.UTILITY.value:
            self._handle_utility(payload)
        elif type_byte == EngineCoreRequestType.START_DP_WAVE.value:
            logger.debug("Ignoring START_DP_WAVE (single-engine façade)")
        else:
            logger.warning("Ignoring unknown request type byte %r", type_byte)

    def _handle_utility(self, payload: bytes) -> None:
        """Phase 1: no utility methods are implemented; always reply so."""
        try:
            _, call_id, method_name, _ = decode_utility_request(payload)
        except Exception:
            logger.exception("Failed to decode UTILITY payload")
            return
        logger.info("Utility call %r (call_id=%d): not implemented", method_name, call_id)
        self._send_queue.put(build_utility_frame(self._engine_index, call_id, "not implemented"))
