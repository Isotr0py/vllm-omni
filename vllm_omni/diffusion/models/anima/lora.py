# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loader for kohya-ss/sd-scripts Anima LoRA checkpoints.

kohya flattens module names to underscores (``lora_unet_blocks_0_cross_attn_k_proj``)
and stores ``lora_down``/``lora_up`` matrices plus a per-module ``alpha`` scalar.
The underscore flattening is lossy, so the restore table below mirrors the
replacement rules of kohya's ``networks/convert_anima_lora_to_comfy.py`` in
their original order.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.diffusion.models.anima.pipeline_anima import _ANIMA_ORIGINAL_TRANSFORMER_RENAMES
from vllm_omni.lora.request import LoRARequest

logger = logging.getLogger(__name__)

_KOHYA_UNET_PREFIX = "lora_unet_"
_KOHYA_TEXT_ENCODER_PREFIX = "lora_te_"

# Inverse of the underscore flattening in kohya's
# networks/convert_anima_lora_to_comfy.py (DiT section), in the same order;
# the replacements are plain substring rules and the sequence matters.
_KOHYA_ANIMA_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("llm.adapter", "llm_adapter"),
    (".linear.", ".linear_"),
    ("t.embedding.norm", "t_embedding_norm"),
    ("x.embedder", "x_embedder"),
    ("adaln.modulation.cross_attn", "adaln_modulation_cross_attn"),
    ("adaln.modulation.mlp", "adaln_modulation_mlp"),
    ("cross.attn", "cross_attn"),
    ("k.proj", "k_proj"),
    ("k.norm", "k_norm"),
    ("q.proj", "q_proj"),
    ("q.norm", "q_norm"),
    ("v.proj", "v_proj"),
    ("o.proj", "o_proj"),
    ("output.proj", "output_proj"),
    ("self.attn", "self_attn"),
    ("final.layer", "final_layer"),
    ("adaln.modulation", "adaln_modulation"),
    ("norm.cross.attn", "norm_cross_attn"),
    ("norm.mlp", "norm_mlp"),
    ("norm.self.attn", "norm_self_attn"),
    ("out.proj", "out_proj"),
)

_ANIMA_LORA_TARGET_PATTERN = (
    r"^transformer\.transformer_blocks\.\d+\.(?:attn[12]\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))$"
)


def kohya_anima_module_to_omni(kohya_module: str) -> str:
    """Map a flattened kohya DiT module name to the AnimaTransformer3DModel path."""
    original = kohya_module.replace("_", ".")
    for old, new in _KOHYA_ANIMA_KEY_REPLACEMENTS:
        original = original.replace(old, new)
    omni = original
    for old, new in _ANIMA_ORIGINAL_TRANSFORMER_RENAMES.items():
        omni = omni.replace(old, new)
    return omni


def _select_lora_file(lora_path: str | Path) -> Path | None:
    path = Path(lora_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None
    if (path / "adapter_config.json").is_file():
        return None
    candidates = sorted(path.glob("*.safetensors"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{path} holds {len(candidates)} safetensors files ({[c.name for c in candidates[:4]]}); "
            "point the LoRA path at one file."
        )
    return None


def load_kohya_anima_lora(
    *,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
) -> tuple[LoRAModel, PEFTHelper] | None:
    """Load a kohya-format Anima DiT LoRA, or ``None`` if the artifact is not one."""
    lora_file = _select_lora_file(lora_path)
    if lora_file is None:
        return None

    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if not any(key.startswith(_KOHYA_UNET_PREFIX) for key in keys):
            return None

        down: dict[str, torch.Tensor] = {}
        up: dict[str, torch.Tensor] = {}
        alphas: dict[str, float] = {}
        skipped_text_encoder = False
        for key in keys:
            if key.startswith(_KOHYA_TEXT_ENCODER_PREFIX):
                skipped_text_encoder = True
                continue
            if not key.startswith(_KOHYA_UNET_PREFIX):
                raise ValueError(f"Unrecognized key {key!r} in kohya Anima LoRA {lora_file}")
            module_name, _, weight_name = key[len(_KOHYA_UNET_PREFIX) :].partition(".")
            if weight_name == "alpha":
                alpha = checkpoint.get_tensor(key)
                if alpha.numel() != 1:
                    raise ValueError(f"kohya Anima LoRA alpha must be a scalar, got {key}={tuple(alpha.shape)}")
                alphas[module_name] = float(alpha.float())
            elif weight_name in ("lora_down.weight", "lora_up.weight"):
                tensor = checkpoint.get_tensor(key)
                if tensor.ndim != 2:
                    raise ValueError(f"kohya Anima LoRA tensors must be matrices, got {key}={tuple(tensor.shape)}")
                side = down if weight_name.startswith("lora_down") else up
                if module_name in side:
                    raise ValueError(f"Duplicate kohya Anima LoRA tensor for {module_name}.{weight_name}")
                side[module_name] = tensor
            else:
                raise ValueError(f"Unsupported tensor {key!r} in kohya Anima LoRA {lora_file}")

    if skipped_text_encoder:
        logger.warning(
            "Ignoring lora_te_* text-encoder entries in %s; only the Anima DiT adapter is served.",
            lora_file,
        )

    missing_up = sorted(set(down) - set(up))
    missing_down = sorted(set(up) - set(down))
    if missing_up or missing_down:
        raise ValueError(
            f"Incomplete kohya Anima LoRA pairs in {lora_file}: "
            f"missing lora_up={missing_up[:5]}, missing lora_down={missing_down[:5]}"
        )
    if not down:
        raise ValueError(f"kohya Anima LoRA {lora_file} contains no lora_unet_* pairs.")

    ranks = {name: tensor.shape[0] for name, tensor in down.items()}
    if len(set(ranks.values())) != 1:
        raise ValueError(f"kohya Anima LoRA {lora_file} mixes per-module ranks: {sorted(ranks.items())[:5]}")
    rank = next(iter(ranks.values()))

    tensors: dict[str, torch.Tensor] = {}
    for module_name, down_tensor in down.items():
        up_tensor = up[module_name]
        if up_tensor.shape[1] != rank:
            raise ValueError(
                f"kohya Anima LoRA pair rank mismatch for {module_name}: "
                f"lora_down={tuple(down_tensor.shape)}, lora_up={tuple(up_tensor.shape)}"
            )
        omni_module = kohya_anima_module_to_omni(module_name)
        if re.search(_ANIMA_LORA_TARGET_PATTERN, f"transformer.{omni_module}") is None:
            raise ValueError(
                f"kohya Anima LoRA target {module_name!r} (->{omni_module!r}) is not a supported Anima module."
            )
        # kohya treats a missing or zero alpha as alpha=rank, i.e. no scaling.
        alpha = alphas.get(module_name) or rank
        lora_b = up_tensor.float()
        if alpha != rank:
            lora_b = lora_b * (alpha / rank)
        tensors[f"transformer.{omni_module}.lora_A.weight"] = down_tensor.float()
        tensors[f"transformer.{omni_module}.lora_B.weight"] = lora_b

    peft_helper = PEFTHelper.from_dict(
        {
            "r": rank,
            # alpha/rank is already folded into lora_B, so the runtime scaling is 1.
            "lora_alpha": rank,
            "target_modules": _ANIMA_LORA_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
    )
    logger.info(
        "Loaded kohya Anima LoRA %s: rank=%d, %d modules",
        lora_file,
        rank,
        len(lora_model.loras),
    )
    return lora_model, peft_helper
