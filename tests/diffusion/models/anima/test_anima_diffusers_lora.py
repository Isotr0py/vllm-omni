# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.anima.lora import (
    diffusers_anima_module_to_omni,
    load_anima_lora,
)
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_RANK = 8
_IN_DIM = 6
_OUT_DIM = 12

# Original Anima training module name (as it appears under the
# ``diffusion_model.`` prefix) -> omni module name.
_DIFFUSERS_TO_OMNI = {
    "blocks.0.self_attn.q_proj": "transformer_blocks.0.attn1.to_q",
    "blocks.0.self_attn.k_proj": "transformer_blocks.0.attn1.to_k",
    "blocks.0.self_attn.v_proj": "transformer_blocks.0.attn1.to_v",
    "blocks.0.self_attn.output_proj": "transformer_blocks.0.attn1.to_out.0",
    "blocks.0.cross_attn.q_proj": "transformer_blocks.0.attn2.to_q",
    "blocks.0.cross_attn.k_proj": "transformer_blocks.0.attn2.to_k",
    "blocks.0.cross_attn.v_proj": "transformer_blocks.0.attn2.to_v",
    "blocks.0.cross_attn.output_proj": "transformer_blocks.0.attn2.to_out.0",
    "blocks.0.mlp.layer1": "transformer_blocks.0.ff.net.0.proj",
    "blocks.0.mlp.layer2": "transformer_blocks.0.ff.net.2",
    "blocks.0.adaln_modulation_self_attn.1": "transformer_blocks.0.norm1.linear_1",
    "blocks.0.adaln_modulation_self_attn.2": "transformer_blocks.0.norm1.linear_2",
    "blocks.0.adaln_modulation_cross_attn.1": "transformer_blocks.0.norm2.linear_1",
    "blocks.0.adaln_modulation_cross_attn.2": "transformer_blocks.0.norm2.linear_2",
    "blocks.0.adaln_modulation_mlp.1": "transformer_blocks.0.norm3.linear_1",
    "blocks.0.adaln_modulation_mlp.2": "transformer_blocks.0.norm3.linear_2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}


def _request(path, int_id: int = 1, name: str = "diffusers") -> LoRARequest:
    return LoRARequest(
        lora_name=name,
        lora_int_id=int_id,
        lora_path=str(path),
    )


def _write_diffusers_lora(
    path,
    *,
    rank: int = _RANK,
    modules: dict[str, int] | None = None,
    alphas: dict[str, float] | None = None,
    include_text_conditioner: bool = False,
) -> dict[str, torch.Tensor]:
    """Write a diffusers/ComfyUI-named toy LoRA; returns the raw lora_B tensors per module."""
    tensors = {}
    ups = {}
    for module, module_rank in (modules or {module: _RANK for module in _DIFFUSERS_TO_OMNI}).items():
        down = torch.randn(module_rank, _IN_DIM)
        up = torch.randn(_OUT_DIM, module_rank)
        tensors[f"diffusion_model.{module}.lora_A.weight"] = down
        tensors[f"diffusion_model.{module}.lora_B.weight"] = up
        ups[module] = up
        if alphas is not None and module in alphas:
            tensors[f"diffusion_model.{module}.alpha"] = torch.tensor(alphas[module])
    if include_text_conditioner:
        for prefix in (
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj",
            "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj",
        ):
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(_RANK, _IN_DIM)
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(_OUT_DIM, _RANK)
    save_file(tensors, str(path))
    return ups


@pytest.mark.parametrize(("original_module", "omni_module"), sorted(_DIFFUSERS_TO_OMNI.items()))
def test_diffusers_key_mapping_covers_all_targets(original_module, omni_module):
    assert diffusers_anima_module_to_omni(original_module) == omni_module


def test_load_diffusers_maps_keys_without_scaling(tmp_path):
    path = tmp_path / "anima_turbo.safetensors"
    ups = _write_diffusers_lora(path)

    loaded = load_anima_lora(
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    lora_model, peft_helper = loaded
    assert peft_helper.r == _RANK
    # No alpha entries: the runtime scaling must be 1.
    assert peft_helper.lora_alpha == _RANK
    assert set(lora_model.loras) == {f"transformer.{omni}" for omni in _DIFFUSERS_TO_OMNI.values()}
    for module, up in ups.items():
        torch.testing.assert_close(lora_model.loras[f"transformer.{_DIFFUSERS_TO_OMNI[module]}"].lora_b, up)


def test_load_diffusers_folds_alpha_when_present(tmp_path):
    path = tmp_path / "anima_turbo.safetensors"
    module = next(iter(_DIFFUSERS_TO_OMNI))
    ups = _write_diffusers_lora(path, alphas={module: 2.0})

    lora_model, _ = load_anima_lora(
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        lora_model.loras[f"transformer.{_DIFFUSERS_TO_OMNI[module]}"].lora_b,
        ups[module] * (2.0 / _RANK),
    )
    untouched = sorted(_DIFFUSERS_TO_OMNI)[1]
    torch.testing.assert_close(lora_model.loras[f"transformer.{_DIFFUSERS_TO_OMNI[untouched]}"].lora_b, ups[untouched])


def test_load_diffusers_ignores_text_conditioner_entries(tmp_path, caplog):
    path = tmp_path / "anima_turbo.safetensors"
    _write_diffusers_lora(path, include_text_conditioner=True)

    with caplog.at_level(logging.WARNING, logger="vllm_omni.diffusion.models.anima.lora"):
        lora_model, _ = load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )

    assert set(lora_model.loras) == {f"transformer.{omni}" for omni in _DIFFUSERS_TO_OMNI.values()}
    assert any("text-encoder/text-conditioner" in record.message for record in caplog.records)


def test_load_diffusers_rejects_unknown_module(tmp_path):
    path = tmp_path / "anima_turbo.safetensors"
    _write_diffusers_lora(path, modules={"blocks.0.t_embedder.1": _RANK})

    with pytest.raises(ValueError, match="not a supported Anima module"):
        load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_load_diffusers_rejects_foreign_prefix(tmp_path):
    path = tmp_path / "anima_turbo.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(_RANK, _IN_DIM),
            "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(_OUT_DIM, _RANK),
            "vae.encoder.conv_in.lora_A.weight": torch.randn(_RANK, _IN_DIM),
        },
        str(path),
    )

    with pytest.raises(ValueError, match="Unrecognized key"):
        load_anima_lora(
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def _compose(specs):
    manager = DiffusionLoRAManager.__new__(DiffusionLoRAManager)
    return manager._compose_lora_models(composed_id=999, specs=specs)


def test_compose_kohya_and_diffusers_adapters(tmp_path):
    shared_kohya = "blocks_1_self_attn_q_proj"
    shared_diffusers = "blocks.1.self_attn.q_proj"
    shared_omni = "transformer.transformer_blocks.1.attn1.to_q"

    kohya_path = tmp_path / "style.safetensors"
    kohya_rank = 4
    kohya_tensors = {
        f"lora_unet_{shared_kohya}.lora_down.weight": torch.randn(kohya_rank, _IN_DIM),
        f"lora_unet_{shared_kohya}.lora_up.weight": torch.randn(_OUT_DIM, kohya_rank),
        f"lora_unet_{shared_kohya}.alpha": torch.tensor(2.0),
        "lora_unet_blocks_1_mlp_layer1.lora_down.weight": torch.randn(kohya_rank, _IN_DIM),
        "lora_unet_blocks_1_mlp_layer1.lora_up.weight": torch.randn(_OUT_DIM, kohya_rank),
        "lora_unet_blocks_1_mlp_layer1.alpha": torch.tensor(2.0),
    }
    save_file(kohya_tensors, str(kohya_path))

    diffusers_path = tmp_path / "turbo.safetensors"
    _write_diffusers_lora(diffusers_path, modules={shared_diffusers: _RANK, "blocks.1.mlp.layer2": _RANK})

    kohya_model, _ = load_anima_lora(
        lora_request=_request(kohya_path, int_id=1, name="style"),
        lora_path=kohya_path,
        dtype=torch.float32,
    )
    turbo_model, _ = load_anima_lora(
        lora_request=_request(diffusers_path, int_id=2, name="turbo"),
        lora_path=diffusers_path,
        dtype=torch.float32,
    )

    kohya_scale = 0.7
    composed, peft_helper = _compose([(kohya_model, kohya_scale), (turbo_model, 1.0)])

    # Union of covered modules; the shared module composes to rank 4+8.
    assert set(composed.loras) == {
        shared_omni,
        "transformer.transformer_blocks.1.ff.net.0.proj",
        "transformer.transformer_blocks.1.ff.net.2",
    }
    assert peft_helper.r == kohya_rank + _RANK

    shared = composed.loras[shared_omni]
    assert shared.rank == kohya_rank + _RANK
    kohya_weights = kohya_model.loras[shared_omni]
    turbo_weights = turbo_model.loras[shared_omni]
    torch.testing.assert_close(shared.lora_a, torch.cat([kohya_weights.lora_a, turbo_weights.lora_a], dim=0))
    torch.testing.assert_close(
        shared.lora_b,
        torch.cat([kohya_weights.lora_b * kohya_scale, turbo_weights.lora_b], dim=1),
    )

    # The composed delta equals the sum of the individually scaled deltas.
    x = torch.randn(_IN_DIM, 5)
    expected = (kohya_weights.lora_b * kohya_scale) @ kohya_weights.lora_a @ x + turbo_weights.lora_b @ (
        turbo_weights.lora_a @ x
    )
    torch.testing.assert_close(shared.lora_b @ shared.lora_a @ x, expected)

    # kohya alpha was folded at load time: B already carries alpha/rank.
    torch.testing.assert_close(
        kohya_weights.lora_b,
        kohya_tensors[f"lora_unet_{shared_kohya}.lora_up.weight"] * (2.0 / kohya_rank),
    )

    # Modules covered by a single adapter keep that adapter's contribution.
    ff_only = composed.loras["transformer.transformer_blocks.1.ff.net.2"]
    assert ff_only.rank == _RANK
    torch.testing.assert_close(ff_only.lora_b, turbo_model.loras["transformer.transformer_blocks.1.ff.net.2"].lora_b)


def test_compose_rank_budget_error(tmp_path):
    path = tmp_path / "turbo.safetensors"
    module = "blocks.0.self_attn.q_proj"
    _write_diffusers_lora(path, modules={module: 64})

    specs = []
    for index in range(9):
        model, _ = load_anima_lora(
            lora_request=_request(path, int_id=index + 1, name=f"turbo{index}"),
            lora_path=path,
            dtype=torch.float32,
        )
        specs.append((model, 1.0))

    with pytest.raises(ValueError, match="exceeding the maximum supported rank"):
        _compose(specs)
