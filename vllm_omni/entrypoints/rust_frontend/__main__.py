# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Entry point for the Rust-frontend EngineCore façade.

Usage:

    python -m vllm_omni.entrypoints.rust_frontend \
        --input-address tcp://127.0.0.1:5555 \
        --output-address tcp://127.0.0.1:5556 \
        [--engine-index 0] [--mock-echo] -- <omni serve args...>

Arguments after ``--`` are parsed with the same argument parser the
``omni serve`` CLI command builds (``OmniServeCommand.subparser_init``), and
forwarded to ``AsyncOmni`` exactly like the OpenAI API server does.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_facade_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split argv on ``--`` and parse the façade's own arguments."""
    if "--" in argv:
        sep = argv.index("--")
        facade_argv, omni_argv = argv[:sep], argv[sep + 1 :]
    else:
        facade_argv, omni_argv = argv, []

    parser = argparse.ArgumentParser(
        prog="python -m vllm_omni.entrypoints.rust_frontend",
        description=(
            "Façade impersonating a vLLM v1 EngineCore for a Bootstrapped "
            "Rust frontend, driving vllm-omni's AsyncOmni engine."
        ),
    )
    parser.add_argument(
        "--input-address",
        required=True,
        help="ZMQ address of the frontend's ROUTER input socket.",
    )
    parser.add_argument(
        "--output-address",
        required=True,
        help="ZMQ address of the frontend's PULL output socket.",
    )
    parser.add_argument(
        "--engine-index",
        type=int,
        default=0,
        help="Engine index; encoded as the 2-byte little-endian DEALER identity.",
    )
    parser.add_argument(
        "--mock-echo",
        action="store_true",
        help="Skip AsyncOmni entirely and stream a canned response per request.",
    )
    return parser.parse_args(facade_argv), omni_argv


def _parse_omni_serve_args(omni_argv: list[str]) -> tuple[str, dict]:
    """Parse args after ``--`` with the omni serve CLI's own parser."""
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand, _ensure_vllm_platform
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    _ensure_vllm_platform()
    parser = TrackingArgumentParser(description="vLLM-Omni serve arguments")
    subparsers = parser.add_subparsers(required=True, dest="subparser")
    cmd = OmniServeCommand()
    cmd.subparser_init(subparsers)
    args = parser.parse_args(["serve", *omni_argv])

    # Mirror OmniServeCommand.cmd / api_server argument handling.
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    kwargs = args.get_explicit_kwargs_dict()
    model = kwargs.pop("model", None) or args.model
    if not model:
        raise ValueError("a model is required after '--' unless --mock-echo is used")
    kwargs.setdefault("log_stats", not args.disable_log_stats)
    return model, kwargs


def main(argv: list[str] | None = None) -> None:
    """Run the façade until SIGINT/SIGTERM."""
    from vllm_omni.entrypoints.rust_frontend.bridge import (
        MockEchoHandler,
        OmniBridge,
    )
    from vllm_omni.entrypoints.rust_frontend.facade import (
        EngineCoreFacade,
        FacadeHandler,
    )

    facade_args, omni_argv = _parse_facade_args(list(sys.argv[1:] if argv is None else argv))

    send_queue: queue.Queue = queue.Queue()
    handler: FacadeHandler
    if facade_args.mock_echo:
        logger.info("Running in --mock-echo mode; no model will be loaded")
        handler = MockEchoHandler(send_queue.put, engine_index=facade_args.engine_index)
    else:
        model, omni_kwargs = _parse_omni_serve_args(omni_argv)
        bridge = OmniBridge(
            send_queue.put,
            engine_index=facade_args.engine_index,
            model=model,
            omni_kwargs=omni_kwargs,
        )
        bridge.start()
        handler = bridge

    facade = EngineCoreFacade(
        input_address=facade_args.input_address,
        output_address=facade_args.output_address,
        engine_index=facade_args.engine_index,
        handler=handler,
        send_queue=send_queue,
    )
    facade.start()

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    logger.info("Façade is up; waiting for shutdown signal")
    stop.wait()
    logger.info("Shutdown signal received")
    facade.shutdown()


if __name__ == "__main__":
    main()
