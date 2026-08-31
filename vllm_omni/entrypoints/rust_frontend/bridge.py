# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bridge between the ZMQ façade and the request backend.

Two backends:

- :class:`OmniBridge` drives a real ``AsyncOmni`` engine on a dedicated
  asyncio loop thread. ``EngineCoreRequest`` objects arrive from the façade's
  receiver thread via ``asyncio.run_coroutine_threadsafe``; streamed
  ``OmniRequestOutput`` deltas are diffed per request and pushed back as
  encoded ``EngineCoreOutputs`` frames.
- :class:`MockEchoHandler`` streams a fixed canned response without loading
  any model, for GPU-less protocol end-to-end tests.

Both expose the same handler interface consumed by
:class:`~vllm_omni.entrypoints.rust_frontend.facade.EngineCoreFacade`:

``get_ready_dict()``, ``handle_add(request)``, ``handle_abort(request_ids)``
and ``shutdown()``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    FINISH_REASON_STRINGS,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
    UtilityOutput,
)

from vllm_omni.entrypoints.rust_frontend.protocol import (
    encode_outputs,
    mock_ready_dict,
    ready_dict_from_vllm_config,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = init_logger(__name__)

#: Canned token stream emitted by the mock-echo backend.
MOCK_ECHO_TOKEN_IDS = [9707, 11, 1879, 330, 13339]
MOCK_ECHO_STEP_S = 0.005


def _finish_reason_from_string(reason: str | None) -> FinishReason | None:
    if reason is None:
        return None
    try:
        return FinishReason(FINISH_REASON_STRINGS.index(reason))
    except ValueError:
        logger.warning("Unknown finish reason %r; reporting as error", reason)
        return FinishReason.ERROR


def build_token_frame(
    engine_index: int,
    request_id: str,
    delta_token_ids: list[int],
    finish_reason: FinishReason | None = None,
    stop_reason: int | str | None = None,
) -> list[bytes]:
    """Encode one streamed (delta) or terminal output for a request."""
    output = EngineCoreOutput(
        request_id=request_id,
        new_token_ids=list(delta_token_ids),
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )
    outputs = EngineCoreOutputs(
        engine_index=engine_index,
        outputs=[output],
        finished_requests={request_id} if finish_reason is not None else None,
    )
    return encode_outputs(outputs)


def build_utility_frame(
    engine_index: int,
    call_id: int,
    failure_message: str | None = None,
) -> list[bytes]:
    """Encode a standalone utility reply (never mixed with token outputs)."""
    outputs = EngineCoreOutputs(
        engine_index=engine_index,
        utility_output=UtilityOutput(call_id=call_id, failure_message=failure_message),
    )
    return encode_outputs(outputs)


def _diff_token_ids(sent: list[int], token_ids: list[int]) -> tuple[list[int], list[int]]:
    """Diff a possibly-cumulative token list against what was already sent.

    Returns ``(delta, new_sent)``. ``AsyncOmni.generate`` coerces outputs to
    DELTA kind for streaming, but this handles cumulative shapes too: if
    ``token_ids`` extends the already-sent prefix it is treated as cumulative,
    otherwise as a delta.
    """
    if token_ids[: len(sent)] == sent:
        return token_ids[len(sent) :], list(token_ids)
    return list(token_ids), sent + list(token_ids)


class MockEchoHandler:
    """GPU-less backend streaming a fixed canned response per request."""

    def __init__(self, enqueue: Callable[[list[bytes]], None], engine_index: int = 0):
        self._enqueue = enqueue
        self._engine_index = engine_index
        self._lock = threading.Lock()
        self._abort_events: dict[str, threading.Event] = {}
        self._threads: list[threading.Thread] = []

    def get_ready_dict(self) -> dict[str, Any]:
        return mock_ready_dict(self._engine_index)

    def handle_add(self, request: EngineCoreRequest) -> None:
        abort_event = threading.Event()
        with self._lock:
            self._abort_events[request.request_id] = abort_event
        thread = threading.Thread(
            target=self._stream,
            args=(request, abort_event),
            name=f"mock-echo-{request.request_id}",
            daemon=True,
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def handle_abort(self, request_ids: list[str]) -> None:
        with self._lock:
            events = [self._abort_events.get(rid) for rid in request_ids]
        for event in events:
            if event is not None:
                event.set()

    def shutdown(self) -> None:
        with self._lock:
            events = list(self._abort_events.values())
            threads = list(self._threads)
        for event in events:
            event.set()
        for thread in threads:
            thread.join(timeout=2.0)

    def _stream(self, request: EngineCoreRequest, abort_event: threading.Event) -> None:
        request_id = request.request_id
        tokens = MOCK_ECHO_TOKEN_IDS
        if request.sampling_params is not None and request.sampling_params.max_tokens:
            tokens = tokens[: request.sampling_params.max_tokens]
        try:
            for token_id in tokens:
                time.sleep(MOCK_ECHO_STEP_S)
                if abort_event.is_set():
                    self._enqueue(build_token_frame(self._engine_index, request_id, [], FinishReason.ABORT))
                    return
                self._enqueue(build_token_frame(self._engine_index, request_id, [token_id]))
            if abort_event.is_set():
                self._enqueue(build_token_frame(self._engine_index, request_id, [], FinishReason.ABORT))
            else:
                self._enqueue(build_token_frame(self._engine_index, request_id, [], FinishReason.STOP))
        finally:
            with self._lock:
                self._abort_events.pop(request_id, None)


class OmniBridge:
    """Asyncio glue driving a real ``AsyncOmni`` engine.

    The engine is constructed (and all requests are run) on a private event
    loop in a dedicated thread; the façade's receiver thread submits work via
    ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(
        self,
        enqueue: Callable[[list[bytes]], None],
        engine_index: int,
        model: str,
        omni_kwargs: dict[str, Any],
    ):
        self._enqueue = enqueue
        self._engine_index = engine_index
        self._model = model
        self._omni_kwargs = omni_kwargs
        self._loop: asyncio.AbstractEventLoop | None = None
        self._engine: AsyncOmni | None = None
        self._ready_dict: dict[str, Any] | None = None
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._thread = threading.Thread(
            target=self._thread_main,
            name="omni-bridge-loop",
            daemon=True,
        )

    def start(self) -> None:
        """Start the loop thread; engine init proceeds asynchronously."""
        self._thread.start()

    def get_ready_dict(self) -> dict[str, Any]:
        """Block until the engine has loaded and return its ready dict."""
        self._ready.wait()
        if self._init_error is not None:
            raise RuntimeError("AsyncOmni initialization failed") from self._init_error
        assert self._ready_dict is not None
        return self._ready_dict

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._init_engine())
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _init_engine(self) -> None:
        try:
            from vllm_omni.entrypoints.async_omni import AsyncOmni

            self._engine = AsyncOmni(model=self._model, **self._omni_kwargs)
            vllm_config = self._engine.vllm_config
            if vllm_config is not None:
                self._ready_dict = ready_dict_from_vllm_config(vllm_config, engine_index=self._engine_index)
            else:
                logger.warning(
                    "No comprehension-stage vllm_config available; registering with placeholder ready values"
                )
                self._ready_dict = mock_ready_dict(self._engine_index)
        except BaseException as exc:
            self._init_error = exc
            logger.exception("Failed to initialize AsyncOmni")
        finally:
            self._ready.set()

    def handle_add(self, request: EngineCoreRequest) -> None:
        loop = self._loop
        if loop is None or self._engine is None:
            logger.warning("Dropping ADD for %s: engine not ready", request.request_id)
            return
        asyncio.run_coroutine_threadsafe(self._track(request), loop)

    def handle_abort(self, request_ids: list[str]) -> None:
        loop = self._loop
        if loop is None or self._engine is None:
            return
        for request_id in request_ids:
            task = self._tasks.get(request_id)
            if task is not None:
                loop.call_soon_threadsafe(task.cancel)
        # Belt and braces: AsyncOmni.abort() maps external request IDs to
        # internal (<external>-<uuid>) ones via its request_states table, so
        # aborts also work if our streaming task has not been scheduled yet.
        asyncio.run_coroutine_threadsafe(self._abort_safe(request_ids), loop)

    async def _abort_safe(self, request_ids: list[str]) -> None:
        try:
            assert self._engine is not None
            await self._engine.abort(list(request_ids))
        except Exception:
            logger.exception("AsyncOmni abort failed for %s", request_ids)

    def shutdown(self) -> None:
        loop = self._loop
        engine = self._engine
        if loop is not None and engine is not None:
            future = asyncio.run_coroutine_threadsafe(self._shutdown_engine(), loop)
            try:
                future.result(timeout=30.0)
            except Exception:
                logger.exception("AsyncOmni shutdown failed")
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=10.0)

    async def _shutdown_engine(self) -> None:
        assert self._engine is not None
        self._engine.shutdown()

    async def _track(self, request: EngineCoreRequest) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks[request.request_id] = task
        try:
            await self._stream_request(request)
        finally:
            self._tasks.pop(request.request_id, None)

    async def _stream_request(self, request: EngineCoreRequest) -> None:
        from vllm import TokensPrompt

        request_id = request.request_id
        engine_index = self._engine_index
        if request.sampling_params is None:
            logger.warning("Request %s has no sampling params; failing it", request_id)
            self._enqueue(build_token_frame(engine_index, request_id, [], FinishReason.ERROR))
            return

        assert self._engine is not None
        prompt = TokensPrompt(prompt_token_ids=list(request.prompt_token_ids or []))
        # Pass the wire-decoded sampling params as the bare ``sampling_params``
        # argument: generate() expands it into the per-stage list itself,
        # taking the engine's defaults and replacing stage 0 with ours.
        generator = self._engine.generate(
            prompt=prompt,
            sampling_params=request.sampling_params,
            request_id=request_id,
            lora_request=request.lora_request,
            priority=request.priority,
            arrival_time=request.arrival_time,
        )
        sent: list[int] = []
        finish_reason: FinishReason | None = None
        stop_reason: int | str | None = None
        try:
            async for output in generator:
                if getattr(output, "error", None):
                    logger.error("Request %s failed engine-side: %s", request_id, output.error)
                    finish_reason = FinishReason.ERROR
                    break
                for completion in output.outputs or []:
                    delta, sent = _diff_token_ids(sent, list(completion.token_ids))
                    if completion.finish_reason is not None and finish_reason is None:
                        finish_reason = _finish_reason_from_string(completion.finish_reason)
                        stop_reason = completion.stop_reason
                    if delta:
                        self._enqueue(build_token_frame(engine_index, request_id, delta))
                if output.finished:
                    break
        except asyncio.CancelledError:
            # handle_abort() cancelled us; generate() aborts the internal
            # request on GeneratorExit/CancelledError. Report the abort.
            self._enqueue(build_token_frame(engine_index, request_id, [], FinishReason.ABORT))
            raise
        except Exception:
            logger.exception("Request %s failed", request_id)
            finish_reason = FinishReason.ERROR
        finally:
            await generator.aclose()

        if finish_reason is None:
            finish_reason = FinishReason.STOP
        self._enqueue(build_token_frame(engine_index, request_id, [], finish_reason, stop_reason))
