# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Megatron-style tensor parallelism for the MOSS-Image text encoder (Qwen3.5).

Applies the checkpoint model's own HF ``tp_plan``: every ``nn.Linear`` the
plan covers is swapped for the matching vLLM parallel linear class
(``ColumnParallelLinear`` / ``RowParallelLinear`` / ``ReplicatedLinear``).
Weights are NOT copied here — the parallel params carry vLLM's
``weight_loader`` sharding metadata, so the ordinary loader path
(``AutoWeightsLoader`` over the lazily-iterated safetensors) narrows each
checkpoint tensor to the local rank while streaming it, and no full-size
host copy of the encoder ever exists.

The collectives ride the vLLM TP group (which vllm-omni binds to the DiT TP
group), so the encoder shards over exactly the DiT TP ranks.  Full-attention
blocks split heads per rank; the linear-attention (GDN) projections use the
plan's ``colwise_gather_output`` style so the recurrence still sees the full
hidden state.
"""

from __future__ import annotations

import re
from typing import Any

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


def apply_text_encoder_tp_plan(text_encoder: nn.Module, quant_config: Any | None = None) -> None:
    """Swap tp_plan-covered linears of ``text_encoder`` for vLLM parallel classes.

    The checkpoint's ``visual`` tower (unused by this text-only pipeline) is
    dropped.  Any ``nn.Linear`` under the language model that the plan does
    not cover raises instead of silently staying replicated — keeping a large
    projection full-size would defeat the point.

    With ``quant_config`` the replacement linears carry a quant method (online
    weight-only quantization): each layer's shard loads through the method's
    lazy weight loader and quantizes in place as soon as its weights finish
    streaming in — shard first, quantize second, never a full-size host copy.
    """
    tp_plan = getattr(text_encoder, "tp_plan", None) or {}
    if not tp_plan:
        raise ValueError(f"{type(text_encoder).__name__} does not declare a tp_plan; cannot shard the text encoder")

    if getattr(text_encoder, "visual", None) is not None:
        del text_encoder.visual

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
                setattr(
                    module,
                    child_name,
                    replace_linear_class(child, _STYLES[style], quant_config, prefix=qual_name),
                )
            else:
                _recursive_replace(child, qual_name)

    _recursive_replace(text_encoder, "")

    leaked = [name for name in unmatched if name.split(".")[0] in ("language_model", "layers")]
    if leaked:
        raise ValueError(f"tp_plan did not cover these text-encoder linears: {leaked}")
