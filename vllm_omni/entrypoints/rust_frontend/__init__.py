# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Façade that pretends to be a vLLM v1 EngineCore on the wire.

A Rust frontend connecting in Bootstrapped mode (no HELLO handshake) talks to
this process exactly as if it were a vLLM ``EngineCoreProc``: registration on
the input DEALER socket, ADD/ABORT/UTILITY requests in, msgpack
``EngineCoreOutputs`` out on the PUSH socket. Internally, requests are driven
through vllm-omni's ``AsyncOmni`` engine.

Phase 1 is text-only and draft-quality.
"""
