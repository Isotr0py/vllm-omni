# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Wire-protocol helpers for the Rust-frontend EngineCore façade.

The registration message is a hand-rolled msgpack map (not vLLM's
``EngineCoreReadyResponse`` dataclass): the Rust frontend expects the *new*
protocol's field set, which is a superset of what the installed vLLM
dataclass carries. Values mirror ``EngineCoreProc._make_ready_response``.
"""

from __future__ import annotations

from typing import Any

import msgspec
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.version import __version__ as VLLM_VERSION

#: Keys the Rust frontend requires in the registration (ready) map.
REQUIRED_READY_KEYS: frozenset[str] = frozenset(
    {
        "max_model_len",
        "num_gpu_blocks",
        "block_size",
        "dp_stats_address",
        "dtype",
        "vllm_version",
        "world_size",
        "data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "data_parallel_rank",
        "max_num_seqs",
        "max_num_batched_tokens",
        "instance_id",
        "supports_lora",
        "max_loras",
    }
)

#: Optional keys, sent only when the value is known.
OPTIONAL_READY_KEYS: frozenset[str] = frozenset(
    {
        "mamba_block_size",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
        "enable_sleep_mode",
        "supports_draft_weight_updates",
    }
)

#: ``dtype`` values the Rust frontend accepts.
VALID_READY_DTYPES: frozenset[str] = frozenset({"float16", "bfloat16", "float32"})


def mock_ready_dict(engine_index: int = 0) -> dict[str, Any]:
    """Build a ready dict with fake constants for ``--mock-echo`` mode.

    Every value below is fabricated; this mode never touches a real model.
    The dict contains exactly :data:`REQUIRED_READY_KEYS`.
    """
    return {
        "max_model_len": 32768,
        "num_gpu_blocks": 1000,
        "block_size": 16,
        "dp_stats_address": None,
        "dtype": "bfloat16",
        "vllm_version": VLLM_VERSION,
        "world_size": 1,
        "data_parallel_size": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "data_parallel_rank": engine_index,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "instance_id": "vllm-omni-mock-echo",
        "supports_lora": False,
        "max_loras": 0,
    }


def ready_dict_from_vllm_config(
    vllm_config: Any,
    engine_index: int = 0,
    dp_stats_address: str | None = None,
) -> dict[str, Any]:
    """Build the ready dict from a (stage-0) ``VllmConfig``.

    Mirrors ``EngineCoreProc._make_ready_response`` in upstream vLLM. Optional
    keys are included only when their value is not None.
    """
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    lora_config = vllm_config.lora_config

    ready: dict[str, Any] = {
        "max_model_len": vllm_config.model_config.max_model_len,
        "num_gpu_blocks": cache_config.num_gpu_blocks or 0,
        "block_size": cache_config.block_size,
        "dp_stats_address": dp_stats_address,
        "dtype": str(vllm_config.model_config.dtype).removeprefix("torch."),
        "vllm_version": VLLM_VERSION,
        "world_size": parallel_config.world_size,
        "data_parallel_size": parallel_config.data_parallel_size,
        "tensor_parallel_size": parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
        "decode_context_parallel_size": parallel_config.decode_context_parallel_size,
        "data_parallel_rank": engine_index,
        "max_num_seqs": scheduler_config.max_num_seqs,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "instance_id": vllm_config.instance_id,
        "supports_lora": lora_config is not None,
        "max_loras": lora_config.max_loras if lora_config is not None else 0,
    }

    optionals = {
        "mamba_block_size": cache_config.mamba_block_size,
        "kv_cache_size_tokens": cache_config.kv_cache_size_tokens,
        "kv_cache_max_concurrency": cache_config.kv_cache_max_concurrency,
        "enable_sleep_mode": bool(vllm_config.model_config.enable_sleep_mode),
    }
    ready.update({k: v for k, v in optionals.items() if v is not None})
    return ready


def encode_ready_dict(ready: dict[str, Any]) -> bytes:
    """Encode the registration ready map as a single msgpack frame."""
    return msgspec.msgpack.encode(ready)


# Field names of the Rust client's EngineCoreOutput in wire order
# (rust/src/engine-core-client/src/protocol/output.rs, 17 fields). Local
# vllm/vllm-omni classes may carry extra trailing fields (e.g. 19 under
# vllm 0.28.0 + omni patch), which the Rust decoder rejects outright, so
# outgoing outputs are realigned to exactly this layout.
_RUST_OUTPUT_FIELDS = (
    "request_id",
    "new_token_ids",
    "new_logprobs",
    "new_prompt_logprobs_tensors",
    "pooling_output",
    "finish_reason",
    "stop_reason",
    "events",
    "kv_transfer_params",
    "ec_transfer_params",
    "trace_headers",
    "prefill_stats",
    "routed_experts",
    "num_nans_in_logits",
    "mm_cache_miss_hashes",
    "new_sampling_mask",
    "spec_decode_metrics",
)

# Field names of the Rust client's EngineCoreOutputs in wire order.
_RUST_OUTPUTS_FIELDS = (
    "engine_index",
    "outputs",
    "scheduler_stats",
    "timestamp",
    "utility_output",
    "finished_requests",
    "wave_complete",
    "start_wave",
)


def _realign(raw: list, local_fields: list[str], rust_fields: tuple[str, ...]) -> list:
    """Realign a struct-as-array to the Rust wire layout, matching by name."""
    if len(raw) == len(rust_fields) and local_fields[: len(rust_fields)] == list(rust_fields):
        return raw
    values = dict(zip(local_fields, raw))
    return [values.get(name) for name in rust_fields]


def encode_outputs(outputs: EngineCoreOutputs) -> list[bytes]:
    """Encode an ``EngineCoreOutputs`` as a single msgpack frame.

    ``msgspec.to_builtins`` lowers the struct (array_like) to plain lists;
    both the envelope and each per-request output are then realigned by field
    name to the Rust client's wire layout, dropping fields it does not know.
    Text-only: no tensor aux frames are produced.
    """
    raw = msgspec.to_builtins(outputs)
    envelope_fields = [f.name for f in msgspec.structs.fields(type(outputs))]
    raw = _realign(raw, envelope_fields, _RUST_OUTPUTS_FIELDS)
    inner_cls = type(outputs.outputs[0]) if outputs.outputs else None
    if inner_cls is not None and raw[1]:
        output_fields = [f.name for f in msgspec.structs.fields(inner_cls)]
        raw[1] = [_realign(list(o), output_fields, _RUST_OUTPUT_FIELDS) for o in raw[1]]
    return [msgspec.msgpack.encode(raw)]


# The Rust client always sends EngineCoreRequest as a full 21-element array
# (see rust/src/engine-core-client/src/protocol/request.rs).
_RUST_REQUEST_ARRAY_LEN = 21

# Field names of the Rust array in wire order (vllm 0.28.x layout).
_RUST_REQUEST_FIELDS = (
    "request_id",
    "prompt_token_ids",
    "mm_features",
    "sampling_params",
    "pooling_params",
    "arrival_time",
    "lora_request",
    "cache_salt",
    "data_parallel_rank",
    "prompt_embeds",
    "prompt_is_token_ids",
    "client_index",
    "current_wave",
    "priority",
    "trace_headers",
    "resumable",
    "external_req_id",
    "reasoning_ended",
    "reasoning_parser_kwargs",
    "abort_immediately",
    "session_id",
)


def _request_class() -> type[EngineCoreRequest]:
    """Resolve the effective EngineCoreRequest class at decode time.

    vllm_omni's patch layer (``vllm_omni/patch.py``) replaces the
    ``vllm.v1.engine.EngineCoreRequest`` module attribute with
    ``OmniEngineCoreRequest``, which appends omni-only fields; resolve lazily
    so we always decode with whichever class is currently installed.
    """
    from vllm.v1 import engine as v1_engine

    return v1_engine.EngineCoreRequest


def _normalize_request_array(payload: bytes, request_cls: type) -> bytes:
    """Re-encode a Rust request array to match the local struct's layout.

    Field layouts drift across vllm versions and vllm_omni's patched
    ``OmniEngineCoreRequest`` (e.g. ``additional_information`` at index 20
    under vllm 0.26.0, after ``session_id`` under 0.28.x). When the local
    struct's leading fields do not match the Rust wire order, realign by
    field name; omni-only fields get None (they have defaults).
    """
    local_fields = [f.name for f in msgspec.structs.fields(request_cls)]
    if local_fields[: len(_RUST_REQUEST_FIELDS)] == list(_RUST_REQUEST_FIELDS):
        return payload
    raw = msgspec.msgpack.decode(payload)
    if not isinstance(raw, list) or len(raw) != _RUST_REQUEST_ARRAY_LEN:
        return payload
    values = dict(zip(_RUST_REQUEST_FIELDS, raw, strict=True))
    aligned = [values.get(name) for name in local_fields]
    return msgspec.msgpack.encode(aligned)


def decode_add_request(
    payload: bytes,
    aux_frames: tuple[bytes, ...] = (),
) -> EngineCoreRequest:
    """Decode an ADD payload into an ``EngineCoreRequest``.

    The Rust client sends a 21-element array. ``_normalize_request_array``
    realigns it by field name when the locally installed struct (possibly
    vllm_omni's patched ``OmniEngineCoreRequest``) has a different layout.
    """
    request_cls = _request_class()
    payload = _normalize_request_array(payload, request_cls)
    decoder = MsgpackDecoder(request_cls)
    if aux_frames:
        return decoder.decode([payload, *aux_frames])
    return decoder.decode(payload)


def decode_abort_request(payload: bytes) -> list[str]:
    """Decode an ABORT payload: a msgpack list of request IDs."""
    return msgspec.msgpack.decode(payload, type=list[str])


def decode_utility_request(payload: bytes) -> tuple[int, int, str, list[Any]]:
    """Decode a UTILITY payload: ``(client_index, call_id, method, args)``."""
    client_index, call_id, method_name, args = msgspec.msgpack.decode(payload)
    return int(client_index), int(call_id), str(method_name), list(args or [])
