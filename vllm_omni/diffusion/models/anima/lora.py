# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loader for Anima LoRA checkpoints in kohya and diffusers/ComfyUI formats.

kohya flattens module names to underscores (``lora_unet_blocks_0_cross_attn_k_proj``)
and stores ``lora_down``/``lora_up`` matrices plus a per-module ``alpha`` scalar.
The underscore flattening is lossy, so the restore table below mirrors the
replacement rules of kohya's ``networks/convert_anima_lora_to_comfy.py`` in
their original order.

diffusers/ComfyUI exports (e.g. the Anima turbo LoRAs) keep the original
dotted training names under a ``diffusion_model.`` prefix and store
``lora_A``/``lora_B`` matrices, usually without any ``alpha``; absent alpha
means no scaling, matching the kohya/ComfyUI default.
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
_DIFFUSERS_DIT_PREFIX = "diffusion_model."
_DIFFUSERS_TEXT_ENCODER_PREFIX = "text_encoders."
# The turbo LoRAs also adapt the llm_adapter text conditioner; only the DiT is
# served, so those entries are skipped with a warning.
_DIFFUSERS_TEXT_CONDITIONER_PREFIX = "llm_adapter."

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
    r"^transformer\.(?:transformer_blocks\.\d+\."
    r"(?:attn[12]\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2)|norm[123]\.linear_[12])"
    r"|patch_embed\.proj|norm_out\.linear_[12]|proj_out)$"
)


def kohya_anima_module_to_omni(kohya_module: str) -> str:
    """Map a flattened kohya DiT module name to the AnimaTransformer3DModel path."""
    original = kohya_module.replace("_", ".")
    for old, new in _KOHYA_ANIMA_KEY_REPLACEMENTS:
        original = original.replace(old, new)
    return diffusers_anima_module_to_omni(original)


def diffusers_anima_module_to_omni(original_module: str) -> str:
    """Map an original dotted Anima training module name to the AnimaTransformer3DModel path."""
    omni = original_module
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


def _get_matrix(checkpoint, key: str, format_name: str) -> torch.Tensor:
    tensor = checkpoint.get_tensor(key)
    if tensor.ndim != 2:
        raise ValueError(f"{format_name} Anima LoRA tensors must be matrices, got {key}={tuple(tensor.shape)}")
    return tensor


def _collect_kohya_entries(
    checkpoint,
    keys: list[str],
    lora_file: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float], bool]:
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
            side = down if weight_name.startswith("lora_down") else up
            if module_name in side:
                raise ValueError(f"Duplicate kohya Anima LoRA tensor for {module_name}.{weight_name}")
            side[module_name] = _get_matrix(checkpoint, key, "kohya")
        else:
            raise ValueError(f"Unsupported tensor {key!r} in kohya Anima LoRA {lora_file}")

    # kohya module names are still flattened; restore the omni names now so the
    # shared builder only ever sees AnimaTransformer3DModel paths.
    def remap(entries: dict[str, torch.Tensor] | dict[str, float]) -> dict:
        return {kohya_anima_module_to_omni(name): value for name, value in entries.items()}

    return remap(down), remap(up), remap(alphas), skipped_text_encoder


def _collect_diffusers_entries(
    checkpoint,
    keys: list[str],
    lora_file: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float], bool]:
    down: dict[str, torch.Tensor] = {}
    up: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    skipped_text_encoder = False
    for key in keys:
        if key.startswith(_DIFFUSERS_TEXT_ENCODER_PREFIX) or key.startswith(
            _DIFFUSERS_DIT_PREFIX + _DIFFUSERS_TEXT_CONDITIONER_PREFIX
        ):
            skipped_text_encoder = True
            continue
        if not key.startswith(_DIFFUSERS_DIT_PREFIX):
            raise ValueError(f"Unrecognized key {key!r} in diffusers Anima LoRA {lora_file}")
        tail = key[len(_DIFFUSERS_DIT_PREFIX) :]
        if tail.endswith(".lora_A.weight"):
            module_name, side = tail[: -len(".lora_A.weight")], down
        elif tail.endswith(".lora_B.weight"):
            module_name, side = tail[: -len(".lora_B.weight")], up
        elif tail.endswith(".alpha"):
            alpha = checkpoint.get_tensor(key)
            if alpha.numel() != 1:
                raise ValueError(f"diffusers Anima LoRA alpha must be a scalar, got {key}={tuple(alpha.shape)}")
            alphas[diffusers_anima_module_to_omni(tail[: -len(".alpha")])] = float(alpha.float())
            continue
        else:
            raise ValueError(f"Unsupported tensor {key!r} in diffusers Anima LoRA {lora_file}")
        module_name = diffusers_anima_module_to_omni(module_name)
        if module_name in side:
            raise ValueError(f"Duplicate diffusers Anima LoRA tensor for {key!r}")
        side[module_name] = _get_matrix(checkpoint, key, "diffusers")
    return down, up, alphas, skipped_text_encoder


def load_anima_lora(
    *,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
) -> tuple[LoRAModel, PEFTHelper] | None:
    """Load a kohya- or diffusers-format Anima DiT LoRA, or ``None`` if the artifact is neither."""
    lora_file = _select_lora_file(lora_path)
    if lora_file is None:
        return None

    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if any(key.startswith(_KOHYA_UNET_PREFIX) for key in keys):
            format_name = "kohya"
            down, up, alphas, skipped_text_encoder = _collect_kohya_entries(checkpoint, keys, lora_file)
        elif any(key.startswith(_DIFFUSERS_DIT_PREFIX) for key in keys):
            format_name = "diffusers"
            down, up, alphas, skipped_text_encoder = _collect_diffusers_entries(checkpoint, keys, lora_file)
        else:
            return None

    if skipped_text_encoder:
        logger.warning(
            "Ignoring text-encoder/text-conditioner entries in %s; only the Anima DiT adapter is served.",
            lora_file,
        )

    missing_up = sorted(set(down) - set(up))
    missing_down = sorted(set(up) - set(down))
    if missing_up or missing_down:
        raise ValueError(
            f"Incomplete {format_name} Anima LoRA pairs in {lora_file}: "
            f"missing lora_B={missing_up[:5]}, missing lora_A={missing_down[:5]}"
        )
    if not down:
        raise ValueError(f"{format_name} Anima LoRA {lora_file} contains no DiT pairs.")

    ranks = {name: tensor.shape[0] for name, tensor in down.items()}
    if len(set(ranks.values())) != 1:
        raise ValueError(f"{format_name} Anima LoRA {lora_file} mixes per-module ranks: {sorted(ranks.items())[:5]}")
    rank = next(iter(ranks.values()))

    tensors: dict[str, torch.Tensor] = {}
    for omni_module, down_tensor in down.items():
        up_tensor = up[omni_module]
        if up_tensor.shape[1] != rank:
            raise ValueError(
                f"{format_name} Anima LoRA pair rank mismatch for {omni_module}: "
                f"lora_A={tuple(down_tensor.shape)}, lora_B={tuple(up_tensor.shape)}"
            )
        if re.search(_ANIMA_LORA_TARGET_PATTERN, f"transformer.{omni_module}") is None:
            raise ValueError(f"Anima LoRA target {omni_module!r} is not a supported Anima module.")
        # A missing or zero alpha means alpha=rank, i.e. no scaling.
        alpha = alphas.get(omni_module) or rank
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
        "Loaded %s Anima LoRA %s: rank=%d, %d modules",
        format_name,
        lora_file,
        rank,
        len(lora_model.loras),
    )
    return lora_model, peft_helper
