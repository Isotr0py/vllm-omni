# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.anima.lora import (
    kohya_anima_module_to_omni,
    load_anima_lora,
)
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_RANK = 4
_IN_DIM = 6
_OUT_DIM = 8

# kohya module name -> omni module name, for every nn.Linear leaf a kohya
# Anima DiT LoRA can target.
_KOHYA_TO_OMNI = {
    "blocks_1_self_attn_q_proj": "transformer_blocks.1.attn1.to_q",
    "blocks_1_self_attn_k_proj": "transformer_blocks.1.attn1.to_k",
    "blocks_1_self_attn_v_proj": "transformer_blocks.1.attn1.to_v",
    "blocks_1_self_attn_output_proj": "transformer_blocks.1.attn1.to_out.0",
    "blocks_1_cross_attn_q_proj": "transformer_blocks.1.attn2.to_q",
    "blocks_1_cross_attn_k_proj": "transformer_blocks.1.attn2.to_k",
    "blocks_1_cross_attn_v_proj": "transformer_blocks.1.attn2.to_v",
    "blocks_1_cross_attn_output_proj": "transformer_blocks.1.attn2.to_out.0",
    "blocks_1_mlp_layer1": "transformer_blocks.1.ff.net.0.proj",
    "blocks_1_mlp_layer2": "transformer_blocks.1.ff.net.2",
}


def _request(path) -> LoRARequest:
    return LoRARequest(
        lora_name="kohya",
        lora_int_id=1,
        lora_path=str(path),
    )


def _write_kohya_lora(
    path,
    *,
    rank: int = _RANK,
    alpha: float | None = 2.0,
    modules: dict[str, int] | None = None,
    omit_alpha_for: str | None = None,
    include_text_encoder: bool = False,
) -> dict[str, torch.Tensor]:
    """Write a kohya-named toy LoRA; returns the raw lora_up tensors per module."""
    tensors = {}
    ups = {}
    for module, module_rank in (modules or {module: _RANK for module in _KOHYA_TO_OMNI}).items():
        down = torch.randn(module_rank, _IN_DIM)
        up = torch.randn(_OUT_DIM, module_rank)
        tensors[f"lora_unet_{module}.lora_down.weight"] = down
        tensors[f"lora_unet_{module}.lora_up.weight"] = up
        ups[module] = up
        if alpha is not None and module != omit_alpha_for:
            tensors[f"lora_unet_{module}.alpha"] = torch.tensor(alpha)
    if include_text_encoder:
        prefix = "lora_te_model_layers_0_self_attn_q_proj"
        tensors[f"{prefix}.lora_down.weight"] = torch.randn(_RANK, _IN_DIM)
        tensors[f"{prefix}.lora_up.weight"] = torch.randn(_OUT_DIM, _RANK)
        tensors[f"{prefix}.alpha"] = torch.tensor(1.0)
    save_file(tensors, str(path))
    return ups


@pytest.mark.parametrize(("kohya_module", "omni_module"), sorted(_KOHYA_TO_OMNI.items()))
def test_kohya_key_mapping_covers_all_targets(kohya_module, omni_module):
    assert kohya_anima_module_to_omni(kohya_module) == omni_module


def test_load_maps_keys_and_folds_alpha(tmp_path):
    path = tmp_path / "anima_kohya.safetensors"
    ups = _write_kohya_lora(path, alpha=2.0)

    loaded = load_anima_lora(
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    lora_model, peft_helper = loaded
    assert peft_helper.r == _RANK
    # alpha/rank is folded into lora_B, so the PEFT scaling must be 1.
    assert peft_helper.lora_alpha == _RANK
    assert set(lora_model.loras) == {f"transformer.{omni}" for omni in _KOHYA_TO_OMNI.values()}
    for module, up in ups.items():
        weights = lora_model.loras[f"transformer.{_KOHYA_TO_OMNI[module]}"]
        torch.testing.assert_close(weights.lora_b, up * (2.0 / _RANK))


def test_load_defaults_missing_alpha_to_no_scaling(tmp_path):
    path = tmp_path / "anima_kohya.safetensors"
    module = next(iter(_KOHYA_TO_OMNI))
    ups = _write_kohya_lora(path, alpha=2.0, omit_alpha_for=module)

    lora_model, _ = load_anima_lora(
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    torch.testing.assert_close(lora_model.loras[f"transformer.{_KOHYA_TO_OMNI[module]}"].lora_b, ups[module])


def test_load_treats_zero_alpha_as_no_scaling(tmp_path):
    path = tmp_path / "anima_kohya.safetensors"
    ups = _write_kohya_lora(path, alpha=0.0)

    lora_model, _ = load_anima_lora(
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    for module, up in ups.items():
        torch.testing.assert_close(lora_model.loras[f"transformer.{_KOHYA_TO_OMNI[module]}"].lora_b, up)


def test_load_ignores_text_encoder_entries(tmp_path, caplog):
    path = tmp_path / "anima_kohya.safetensors"
    _write_kohya_lora(path, include_text_encoder=True)

    with caplog.at_level(logging.WARNING, logger="vllm_omni.diffusion.models.anima.lora"):
        lora_model, _ = load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )

    assert set(lora_model.loras) == {f"transformer.{omni}" for omni in _KOHYA_TO_OMNI.values()}
    assert any("text-encoder" in record.message for record in caplog.records)


def test_load_rejects_mixed_ranks(tmp_path):
    path = tmp_path / "anima_kohya.safetensors"
    modules = {module: _RANK for module in _KOHYA_TO_OMNI}
    modules["blocks_1_mlp_layer1"] = _RANK * 2
    _write_kohya_lora(path, modules=modules)

    with pytest.raises(ValueError, match="rank"):
        load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_load_rejects_unsupported_target(tmp_path):
    path = tmp_path / "anima_kohya.safetensors"
    _write_kohya_lora(path, modules={"t_embedder_1": _RANK})

    with pytest.raises(ValueError, match="not a supported Anima module"):
        load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_load_rejects_incomplete_pairs(tmp_path):
    path = tmp_path / "anima_kohya.safetensors"
    save_file({"lora_unet_blocks_1_self_attn_q_proj.lora_down.weight": torch.randn(_RANK, _IN_DIM)}, str(path))

    with pytest.raises(ValueError, match="Incomplete kohya Anima LoRA pairs"):
        load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_load_returns_none_for_non_kohya_artifacts(tmp_path):
    diffusers_path = tmp_path / "diffusers.safetensors"
    save_file(
        {"transformer.transformer_blocks.1.attn1.to_q.lora_A.weight": torch.randn(_RANK, _IN_DIM)},
        str(diffusers_path),
    )
    assert (
        load_anima_lora(
            lora_request=_request(diffusers_path),
            lora_path=diffusers_path,
            dtype=torch.float32,
        )
        is None
    )

    peft_dir = tmp_path / "peft_adapter"
    peft_dir.mkdir()
    (peft_dir / "adapter_config.json").write_text("{}")
    save_file(
        {"lora_unet_blocks_1_self_attn_q_proj.lora_down.weight": torch.randn(_RANK, _IN_DIM)},
        str(peft_dir / "adapter_model.safetensors"),
    )
    assert (
        load_anima_lora(
            lora_request=_request(peft_dir),
            lora_path=peft_dir,
            dtype=torch.float32,
        )
        is None
    )

    assert (
        load_anima_lora(
            lora_request=_request(tmp_path / "missing.safetensors"),
            lora_path=tmp_path / "missing.safetensors",
            dtype=torch.float32,
        )
        is None
    )
