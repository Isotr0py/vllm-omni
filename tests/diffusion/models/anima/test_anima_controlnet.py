# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ControlNet-LLLite (v2) adapter tests for the Anima DiT.

Covers checkpoint parsing/loading, the FiLM correction math, the
conditioning-trunk token grid, injection into a tiny
AnimaTransformer3DModel (CPU, mocked TP like test_anima_tp.py), and the
pipeline-level control-image preprocessing and sigma-window helpers.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.attention import selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.models.anima import anima_transformer
from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel
from vllm_omni.diffusion.models.anima.controlnet import AnimaControlNetLLLite, LLLiteModule

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_TINY_LLLITE = {
    "num_blocks": 2,
    "hidden_dim": 64,
    "cond_emb_dim": 8,
    "mlp_dim": 8,
    "cond_dim": 16,
    "cond_resblocks": 1,
    "aspp_dilations": (1, 2),
}

_TINY_LLLITE_METADATA = {
    "lllite.version": "2",
    "lllite.cond_dim": "16",
    "lllite.cond_emb_dim": "8",
    "lllite.mlp_dim": "8",
    "lllite.cond_resblocks": "1",
    "lllite.use_aspp": "true",
    "lllite.aspp_dilations": "1,2",
    "lllite.cond_in_channels": "3",
}

_TINY_TRANSFORMER_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 1,
    "mlp_ratio": 1.0,
    "text_embed_dim": 4,
    "adaln_lora_dim": 3,
    "max_size": (1, 2, 2),
    "patch_size": (1, 1, 1),
    "rope_scale": (1.0, 1.0, 1.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}

_TRANSFORMER_HIDDEN = _TINY_TRANSFORMER_CONFIG["num_attention_heads"] * _TINY_TRANSFORMER_CONFIG["attention_head_dim"]

_REAL_CHECKPOINT = Path("/tmp/lllite/anima_regional_4335.safetensors")


def _write_lllite_checkpoint(
    path: Path,
    *,
    metadata: dict[str, str] | None = None,
    drop_saved_key: str | None = None,
    extra_tensors: dict[str, torch.Tensor] | None = None,
    **dims,
) -> AnimaControlNetLLLite:
    """Write a kohya-format (sd-scripts v2) LLLite checkpoint; returns the source net."""
    torch.manual_seed(0)
    net = AnimaControlNetLLLite(**{**_TINY_LLLITE, **dims})
    for param in net.parameters():
        nn.init.normal_(param)
    state = {}
    for name, tensor in net.state_dict().items():
        if name.startswith("conditioning1."):
            saved = f"lllite_{name}"
        else:
            match = re.fullmatch(r"block_modules\.(\d+)\.(.+)", name)
            module_idx, tail = int(match.group(1)), match.group(2)
            saved = f"lllite_dit_blocks_{module_idx // 3}_self_attn_{'qkv'[module_idx % 3]}_proj.{tail}"
        state[saved] = tensor.contiguous()
    if drop_saved_key is not None:
        del state[drop_saved_key]
    state.update(extra_tensors or {})
    save_file(state, str(path), metadata=_TINY_LLLITE_METADATA if metadata is None else metadata)
    return net


def _mock_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock a TP world of 1 (same approach as test_anima_tp.py)."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(anima_transformer, "get_tensor_model_parallel_world_size", lambda: 1)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda *_args, **_kwargs: default_unquantized_gemm)
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda tensor: tensor)


def _build_transformer(monkeypatch: pytest.MonkeyPatch) -> AnimaTransformer3DModel:
    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *_args, **_kwargs: SDPABackend)
    _mock_tp(monkeypatch)
    transformer = AnimaTransformer3DModel(**_TINY_TRANSFORMER_CONFIG).eval()
    # vLLM linear weights are uninitialized at construction (proj_out is exactly
    # zero), so randomize every parameter to make the forward sensitive to its input.
    for param in transformer.parameters():
        nn.init.normal_(param)
    return transformer


def _tiny_forward_inputs() -> dict:
    return {
        "hidden_states": torch.randn(1, 1, 1, 2, 2),
        "timestep": torch.tensor([0.5]),
        "encoder_hidden_states": torch.randn(1, 3, 4),
        "padding_mask": torch.zeros(1, 1, 2, 2),
    }


# ── Checkpoint parsing and loading ──


def test_from_single_file_parses_metadata_and_remaps_keys(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    source = _write_lllite_checkpoint(path)

    loaded = AnimaControlNetLLLite.from_single_file(str(path))

    assert loaded.num_blocks == _TINY_LLLITE["num_blocks"]
    assert loaded.hidden_dim == _TINY_LLLITE["hidden_dim"]
    assert len(loaded.block_modules) == 3 * _TINY_LLLITE["num_blocks"]
    source_state = source.state_dict()
    loaded_state = loaded.state_dict()
    assert set(loaded_state) == set(source_state)
    for name, tensor in source_state.items():
        assert torch.equal(loaded_state[name], tensor), name


def test_from_single_file_honors_dtype(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(path)

    loaded = AnimaControlNetLLLite.from_single_file(str(path), dtype=torch.bfloat16)

    assert loaded.conditioning1.conv1.weight.dtype == torch.bfloat16
    assert loaded.block_modules[0].down.weight.dtype == torch.bfloat16


def test_from_single_file_rejects_wrong_version(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(path, metadata={**_TINY_LLLITE_METADATA, "lllite.version": "1"})

    with pytest.raises(ValueError, match="not an LLLite v2 checkpoint"):
        AnimaControlNetLLLite.from_single_file(str(path))


def test_from_single_file_rejects_missing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(path, metadata={})

    with pytest.raises(ValueError, match="not an LLLite v2 checkpoint"):
        AnimaControlNetLLLite.from_single_file(str(path))

    malformed = {key: value for key, value in _TINY_LLLITE_METADATA.items() if key != "lllite.cond_dim"}
    _write_lllite_checkpoint(path, metadata=malformed)
    with pytest.raises(ValueError, match="malformed LLLite metadata"):
        AnimaControlNetLLLite.from_single_file(str(path))


def test_from_single_file_rejects_non_aspp_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(path, metadata={**_TINY_LLLITE_METADATA, "lllite.use_aspp": "false"})

    with pytest.raises(ValueError, match="without ASPP"):
        AnimaControlNetLLLite.from_single_file(str(path))


def test_from_single_file_rejects_unrecognized_key(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(
        path,
        extra_tensors={"lllite_dit_blocks_0_self_attn_w_proj.down.weight": torch.randn(8, 64)},
    )

    with pytest.raises(ValueError, match="Unrecognized key"):
        AnimaControlNetLLLite.from_single_file(str(path))


def test_from_single_file_loads_strictly(tmp_path: Path) -> None:
    path = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(path, drop_saved_key="lllite_dit_blocks_1_self_attn_v_proj.up.weight")

    with pytest.raises(RuntimeError, match="Missing key"):
        AnimaControlNetLLLite.from_single_file(str(path))


# ── LLLiteModule correction math ──


def test_lllite_module_zero_init_is_identity() -> None:
    torch.manual_seed(0)
    module = LLLiteModule(hidden_dim=16, cond_emb_dim=4, mlp_dim=8)
    x = torch.randn(2, 5, 16)
    cond_emb = torch.randn(2, 5, 4)

    correction = module.forward_correction(x, cond_emb, multiplier=1.0)

    assert correction is not None
    assert correction.shape == x.shape
    assert torch.equal(correction, torch.zeros_like(x))


def test_forward_correction_gating() -> None:
    torch.manual_seed(0)
    module = LLLiteModule(hidden_dim=16, cond_emb_dim=4, mlp_dim=8)
    for param in module.parameters():
        nn.init.normal_(param)
    x = torch.randn(1, 5, 16)
    cond_emb = torch.randn(1, 5, 4)

    assert module.forward_correction(x, cond_emb, multiplier=0.0) is None
    assert module.forward_correction(x, None, multiplier=1.0) is None
    # Sequence-length mismatch -> correction skipped.
    assert module.forward_correction(x, torch.randn(1, 3, 4), multiplier=1.0) is None

    correction = module.forward_correction(x, cond_emb, multiplier=1.0)
    assert correction is not None
    assert torch.any(correction != 0)
    # Multiplier scales the correction linearly.
    torch.testing.assert_close(
        module.forward_correction(x, cond_emb, multiplier=0.5),
        correction * 0.5,
    )


def test_forward_correction_repeats_cond_over_cfg_batch() -> None:
    torch.manual_seed(0)
    module = LLLiteModule(hidden_dim=16, cond_emb_dim=4, mlp_dim=8)
    for param in module.parameters():
        nn.init.normal_(param)
    cond_emb = torch.randn(1, 5, 4)
    x = torch.randn(2, 5, 16)

    correction = module.forward_correction(x, cond_emb, multiplier=1.0)

    assert correction is not None
    assert correction.shape == x.shape
    # Batch entries with identical x must get identical corrections.
    x_same = x[0:1].repeat(2, 1, 1)
    repeated = module.forward_correction(x_same, cond_emb, multiplier=1.0)
    torch.testing.assert_close(repeated[0], repeated[1])


# ── Conditioning trunk ──


def test_cond_image_token_grid_matches_stride_16() -> None:
    net = AnimaControlNetLLLite(**_TINY_LLLITE)

    net.set_cond_image(torch.rand(1, 3, 64, 96) * 2 - 1)

    assert net.cond_emb is not None
    # Total stride 16: 64/16=4 tokens high, 96/16=6 tokens wide.
    assert net.cond_emb.shape == (1, 4 * 6, _TINY_LLLITE["cond_emb_dim"])
    net.clear_cond_image()
    assert net.cond_emb is None


# ── Transformer integration ──


def test_apply_to_rejects_block_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _build_transformer(monkeypatch)
    net = AnimaControlNetLLLite(**_TINY_LLLITE)  # 2 blocks vs 1 transformer layer

    with pytest.raises(ValueError, match="adapter expects 2"):
        net.apply_to(transformer)


def test_lllite_correction_changes_and_restores_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch)
    inputs = _tiny_forward_inputs()
    checkpoint = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(checkpoint, num_blocks=1, hidden_dim=_TRANSFORMER_HIDDEN)
    net = AnimaControlNetLLLite.from_single_file(str(checkpoint))
    # 2x2 image tokens -> a 32x32 control image (stride 16).
    control_image = torch.rand(1, 3, 32, 32) * 2 - 1

    with torch.inference_mode():
        baseline = transformer(**inputs, return_dict=False)[0]

        net.apply_to(transformer)
        net.set_cond_image(control_image)
        controlled = transformer(**inputs, return_dict=False)[0]
        assert controlled.shape == baseline.shape
        assert torch.isfinite(controlled).all()
        assert not torch.allclose(controlled, baseline)

        # Zero multiplier disables the correction exactly.
        net.set_multiplier(0.0)
        assert torch.equal(transformer(**inputs, return_dict=False)[0], baseline)
        net.set_multiplier(1.0)

        # Clearing the conditioning image disables the correction exactly.
        net.clear_cond_image()
        assert torch.equal(transformer(**inputs, return_dict=False)[0], baseline)

        # Detaching the adapter restores the baseline exactly.
        net.set_cond_image(control_image)
        net.restore(transformer)
        assert torch.equal(transformer(**inputs, return_dict=False)[0], baseline)


def test_lllite_cond_emb_seq_len_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    transformer = _build_transformer(monkeypatch)
    inputs = _tiny_forward_inputs()
    checkpoint = tmp_path / "lllite.safetensors"
    _write_lllite_checkpoint(checkpoint, num_blocks=1, hidden_dim=_TRANSFORMER_HIDDEN)
    net = AnimaControlNetLLLite.from_single_file(str(checkpoint))
    net.apply_to(transformer)
    # 48x32 -> 3x2=6 tokens, but the latent grid has 2x2=4.
    net.set_cond_image(torch.rand(1, 3, 48, 32) * 2 - 1)

    with torch.inference_mode(), pytest.raises(ValueError, match="sequence length mismatch"):
        transformer(**inputs, return_dict=False)


# ── Pipeline-level helpers ──


def test_anima_pipeline_registration_and_image_input_support() -> None:
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline
    from vllm_omni.diffusion.registry import _DIFFUSION_PRE_PROCESS_FUNCS

    assert _DIFFUSION_PRE_PROCESS_FUNCS["AnimaPipeline"] == "get_anima_pre_process_func"
    # SupportImageInput is a Protocol with ClassVar members, so check the marker directly.
    assert AnimaPipeline.support_image_input is True


def _run_pre_process(prompt):
    from vllm_omni.diffusion.models.anima.pipeline_anima import get_anima_pre_process_func
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    request = OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="lllite-pre",
    )
    return get_anima_pre_process_func(None)(request)


def test_pre_process_func_moves_control_image_to_additional_information() -> None:
    image = PIL.Image.fromarray((np.random.rand(20, 12, 3) * 255).astype(np.uint8))

    processed = _run_pre_process(
        {
            "prompt": "a red cube",
            "multi_modal_data": {
                "image": image,
                "controlnet_strength": 0.5,
                "controlnet_start": 0.2,
                "controlnet_end": 0.8,
            },
        }
    )

    info = processed.prompt["additional_information"]
    assert isinstance(info["control_image"], PIL.Image.Image)
    assert info["control_image"].mode == "RGB"
    assert info["control_image"].size == image.size
    assert info["controlnet_strength"] == 0.5
    assert info["controlnet_start"] == 0.2
    assert info["controlnet_end"] == 0.8
    assert processed.batch_compatibility_key == ("anima_control_image", ("image", image.size))


def test_pre_process_func_loads_control_image_from_path(tmp_path: Path) -> None:
    image_path = tmp_path / "control.png"
    PIL.Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8), mode="RGB").save(image_path)

    processed = _run_pre_process({"prompt": "a red cube", "multi_modal_data": {"control_image": str(image_path)}})

    info = processed.prompt["additional_information"]
    assert isinstance(info["control_image"], PIL.Image.Image)
    assert info["control_image"].mode == "RGB"
    assert info["control_image"].size == (16, 16)
    # The compatibility key describes the loaded image, not the source path.
    assert processed.batch_compatibility_key == ("anima_control_image", ("image", (16, 16)))


def test_pre_process_func_without_multi_modal_data() -> None:
    processed = _run_pre_process("a red cube")

    info = processed.prompt["additional_information"]
    assert "control_image" not in info
    assert processed.batch_compatibility_key == ("anima_control_image", ("none",))


def _make_control_image_probe() -> object:
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.controlnet = AnimaControlNetLLLite(**_TINY_LLLITE)
    return pipeline


def test_prepare_control_image_resizes_and_normalizes() -> None:
    pipeline = _make_control_image_probe()
    image = PIL.Image.fromarray((np.random.rand(20, 12, 3) * 255).astype(np.uint8))

    tensor = pipeline._prepare_control_image(image, 32, 48)

    assert tensor.shape == (1, 3, 32, 48)
    assert tensor.dtype == pipeline.controlnet.conditioning1.conv1.weight.dtype
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_prepare_control_image_rejects_bad_inputs() -> None:
    pipeline = _make_control_image_probe()
    image = PIL.Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="multiples of 16"):
        pipeline._prepare_control_image(image, 30, 48)
    with pytest.raises(ValueError, match="path or PIL.Image"):
        pipeline._prepare_control_image(42, 32, 48)


def test_sigma_at_percent_interpolates_and_clamps() -> None:
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25])

    assert AnimaPipeline._sigma_at_percent(0.0, sigmas) == pytest.approx(1.0)
    assert AnimaPipeline._sigma_at_percent(1.0, sigmas) == pytest.approx(0.25)
    assert AnimaPipeline._sigma_at_percent(0.5, sigmas) == pytest.approx(0.625)
    assert AnimaPipeline._sigma_at_percent(-1.0, sigmas) == pytest.approx(1.0)
    assert AnimaPipeline._sigma_at_percent(2.0, sigmas) == pytest.approx(0.25)


def test_sigma_window_gates_multiplier_like_diffuse() -> None:
    """Mirrors the per-step activation predicate in AnimaPipeline.diffuse."""
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    sigma_start = AnimaPipeline._sigma_at_percent(0.25, sigmas)
    sigma_end = AnimaPipeline._sigma_at_percent(0.75, sigmas)

    active = [sigma_end <= float(sigma) <= sigma_start for sigma in sigmas]

    assert active == [False, True, True, True, False]


# ── Real checkpoint smoke test ──


@pytest.mark.skipif(not _REAL_CHECKPOINT.is_file(), reason="real LLLite checkpoint not available")
def test_real_toyxyz_checkpoint_loads_and_conditions() -> None:
    net = AnimaControlNetLLLite.from_single_file(str(_REAL_CHECKPOINT))

    assert net.num_blocks == 28
    assert net.hidden_dim == 2048
    assert len(net.block_modules) == 3 * 28
    net.set_cond_image(torch.rand(1, 3, 64, 64) * 2 - 1)
    assert net.cond_emb is not None
    assert net.cond_emb.shape == (1, 16, 64)
