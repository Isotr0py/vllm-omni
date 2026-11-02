# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Megatron-style tensor parallelism for the MOSS-Image text encoder (Qwen3.5).

Applies the checkpoint model's own HF ``tp_plan``: every ``nn.Linear`` the
plan covers is swapped for the matching vLLM parallel linear class
(``ColumnParallelLinear`` / ``RowParallelLinear`` / ``ReplicatedLinear``) and
the rank-local weight shard is copied in.  The collectives ride the vLLM TP
group (which vllm-omni binds to the DiT TP group), so the encoder is sharded
over exactly the DiT TP ranks and every layer's math is unchanged — full
attention halves heads per rank, the linear-attention (GDN) projections are
``colwise_gather_output`` so the recurrence still sees the full hidden state.

Unlike the layer-slice variant this is standard megatron sharding, but it can
only shard over the DiT TP group: ``text_encoder_tp_size`` must equal the DiT
tensor-parallel size (validated by the pipeline).
"""

from __future__ import annotations

import gc
import re

import torch
from torch import nn

from vllm_omni.diffusion.models.utils import replace_linear_class

# HF tp_plan style -> vllm-omni replace_linear_class style
_STYLES = {
    "colwise": "colwise",
    "colwise_gather_output": "colwise_rep",
    "colwise_rep": "colwise_rep",
    "rowwise": "rowwise",
    "rowwise_split_input": "rowwise_rep",
    "rowwise_rep": "rowwise_rep",
    "replicate": "replicate",
}

# Plan entries that are not nn.Linear (e.g. per-head RMSNorms) are replicated
# on every rank already, so they need no replacement.
_SKIPPED_STYLES = {"replicated_with_grad_allreduce", "local", "gather", "local_packed_rowwise"}


def _shard_weight(new_linear: nn.Module, old_linear: nn.Linear, style: str, rank: int, tp_size: int) -> None:
    """Copy the rank-local shard of ``old_linear``'s weights into ``new_linear``."""
    weight = old_linear.weight.data
    if style == "colwise":
        if weight.shape[0] % tp_size:
            raise ValueError(f"colwise weight rows {weight.shape[0]} not divisible by tp_size {tp_size}")
        shard = weight.narrow(0, rank * (weight.shape[0] // tp_size), weight.shape[0] // tp_size)
    elif style == "rowwise":
        if weight.shape[1] % tp_size:
            raise ValueError(f"rowwise weight cols {weight.shape[1]} not divisible by tp_size {tp_size}")
        shard = weight.narrow(1, rank * (weight.shape[1] // tp_size), weight.shape[1] // tp_size)
    else:  # replicate
        shard = weight
    new_linear.weight.data.copy_(shard)

    if old_linear.bias is not None:
        if style == "colwise":
            per = old_linear.bias.shape[0] // tp_size
            new_linear.bias.data.copy_(old_linear.bias.data.narrow(0, rank * per, per))
        else:
            # RowParallel adds the full bias after the all-reduce; replicate keeps it whole.
            new_linear.bias.data.copy_(old_linear.bias.data)


def shard_text_encoder_megatron_tp(text_encoder: nn.Module, rank: int, tp_size: int) -> None:
    """Shard ``text_encoder`` in place per its HF ``tp_plan``.

    The checkpoint's ``visual`` tower (unused by this text-only pipeline) is
    dropped.  Any ``nn.Linear`` under ``language_model`` not covered by the
    plan raises instead of silently staying replicated — silently keeping a
    large projection full-size would defeat the point.
    """
    tp_plan = getattr(text_encoder, "tp_plan", None) or {}
    if not tp_plan:
        raise ValueError(f"{type(text_encoder).__name__} does not declare a tp_plan; cannot shard the text encoder")

    if getattr(text_encoder, "visual", None) is not None:
        del text_encoder.visual

    language_model = getattr(text_encoder, "language_model", None) or text_encoder
    patterns = list(tp_plan.items())
    unmatched: list[str] = []

    def _recursive_replace(module: nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            qual_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                style = next((s for pattern, s in patterns if re.match(pattern, qual_name)), None)
                if style is None:
                    unmatched.append(qual_name)
                    continue
                if style in _SKIPPED_STYLES:
                    continue
                new_linear = replace_linear_class(child, _STYLES[style], prefix=qual_name)
                shard_style = "colwise" if style.startswith("colwise") else "rowwise" if style.startswith("rowwise") else "replicate"
                _shard_weight(new_linear, child, shard_style, rank, tp_size)
                setattr(module, child_name, new_linear)
            else:
                _recursive_replace(child, qual_name)

    _recursive_replace(text_encoder, "")

    leaked = [name for name in unmatched if name.split(".")[0] in ("language_model", "layers")]
    if leaked:
        raise ValueError(f"tp_plan did not cover these text-encoder linears: {leaked}")
    gc.collect()
