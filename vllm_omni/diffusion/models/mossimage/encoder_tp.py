# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer-slice tensor parallelism for the MOSS-Image text encoder (Qwen3.5).

Unlike MiniMax-H3's megatron-style reimplementation, this module shards the HF
``Qwen3_5Model`` *vertically*: each encoder rank keeps one contiguous slice of
``language_model.layers`` and the hidden state is handed rank to rank with a
broadcast per slice boundary.  The per-layer computation is untouched, so the
split forward is bit-identical to the full forward (covered by tests).

Constraint: every rank of the DiT world must participate (the pipeline
validates ``text_encoder_tp_size in (1, world_size)``), because the broadcasts
run over the world group and every rank consumes the encoder output.
"""

from __future__ import annotations

import copy
import gc

import torch
import torch.distributed as dist
from torch import nn


def slice_text_encoder_for_tp(text_encoder: nn.Module, rank: int, tp_size: int) -> None:
    """Shard ``text_encoder`` in place, keeping only this rank's layer slice.

    The first rank keeps ``embed_tokens``; the last rank keeps the final norm.
    ``config.layer_types`` / ``config.num_hidden_layers`` are narrowed on a
    shallow config copy so ``Qwen3_5TextModel.forward`` indexes the correct
    mask type for each local layer.  The (text-generation only) vision tower
    is dropped on every rank.
    """
    language_model = getattr(text_encoder, "language_model", None) or text_encoder
    layers = language_model.layers
    num_layers = len(layers)
    if num_layers % tp_size != 0:
        raise ValueError(f"text encoder layers ({num_layers}) must divide text_encoder_tp_size ({tp_size})")
    per_rank = num_layers // tp_size
    start, end = rank * per_rank, (rank + 1) * per_rank

    language_model.layers = nn.ModuleList(list(layers)[start:end])
    config = copy.copy(language_model.config)
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = list(config.layer_types)[start:end]
    config.num_hidden_layers = per_rank
    language_model.config = config

    if rank > 0 and getattr(language_model, "embed_tokens", None) is not None:
        del language_model.embed_tokens
    if rank < tp_size - 1 and getattr(language_model, "norm", None) is not None:
        language_model.norm = nn.Identity()
    if getattr(text_encoder, "visual", None) is not None:
        del text_encoder.visual
    gc.collect()


def text_encoder_tp_forward(
    language_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    rank: int,
    tp_size: int,
    device_group: Any = None,
    hidden_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the sliced encoder forward; returns the full last hidden state on every rank.

    All ranks tokenize identical prompts, so the intermediate shape is known
    everywhere and no shape handshake is needed.  Rank ``stage`` computes its
    slice and broadcasts the hidden state; the last slice includes the final
    norm, so the loop's last broadcast delivers the encoder output.
    """
    if tp_size == 1:
        return language_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    if hidden_dtype is None:
        hidden_dtype = next(language_model.parameters()).dtype
    hidden_size = language_model.config.hidden_size
    hidden = None
    for stage in range(tp_size):
        if rank == stage:
            if stage == 0:
                hidden = language_model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                ).last_hidden_state
            else:
                hidden = language_model(
                    inputs_embeds=hidden, attention_mask=attention_mask, use_cache=False
                ).last_hidden_state
            hidden = hidden.contiguous()
        else:
            hidden = torch.empty(
                (input_ids.shape[0], input_ids.shape[1], hidden_size),
                device=input_ids.device,
                dtype=hidden_dtype,
            )
        dist.broadcast(hidden, src=stage, group=device_group)
    return hidden
