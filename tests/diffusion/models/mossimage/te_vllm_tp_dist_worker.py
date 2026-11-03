# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Child-process worker for test_te_megatron_tp_forward_gloo_two_ranks.

Not a pytest module: spawned by the test with argv = [rank, repo_root] and
TE_TP_CHECKPOINT / TE_TP_STORE in the environment. Each rank loads the full
checkpoint as reference, then builds the loader-fed TP-sharded skeleton and
compares the forward outputs. Prints TE_MEGATRON_TP_DIST_OK on success.

When no real vllm is installed (local CPU dev box), the OMNI_STUB_SHIM env
var points at a shim file that installs faithful stub TP linears; CI runs
this worker against the real vllm classes.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_SHIM = os.environ.get("OMNI_STUB_SHIM")
if _SHIM:
    sys.path.insert(0, str(Path(_SHIM).parent))
    import omni_stub_env

    omni_stub_env.install()


def _load_module(repo_root: str, relative: str, name: str):
    path = Path(repo_root) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    text_cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
        max_position_embeddings=128,
    )
    vision_cfg = Qwen3_5VisionConfig(hidden_size=32, depth=2, num_heads=2, out_hidden_size=64)
    return Qwen3_5Config(text_config=text_cfg.to_dict(), vision_config=vision_cfg.to_dict())


def _iter_checkpoint(path: str, prefix: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            yield prefix + key, f.get_tensor(key)


def main() -> None:
    rank = int(sys.argv[1])
    repo_root = sys.argv[2]
    checkpoint = os.environ["TE_TP_CHECKPOINT"]
    dist.init_process_group("gloo", init_method=f"file://{os.environ['TE_TP_STORE']}", rank=rank, world_size=2)

    from transformers import AutoConfig, AutoModel
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm_omni.diffusion.models import utils as omni_model_utils

    encoder_vllm_tp = _load_module(
        repo_root, "vllm_omni/diffusion/models/mossimage/encoder_vllm_tp.py", "mossimage_encoder_vllm_tp"
    )

    input_ids = torch.tensor([[3, 17, 42, 99, 7, 200, 11, 5, 6, 8, 13], [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0]])
    attention_mask = torch.ones(2, 11, dtype=torch.long)
    attention_mask[1, 7:] = 0

    # Reference: full model, plain load.
    from safetensors.torch import load_file

    reference = Qwen3_5Model(_tiny_config()).eval()
    reference.load_state_dict(load_file(checkpoint))
    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    # Sharded: skeleton -> plan -> lazy feed, mirroring the pipeline.
    config = _tiny_config()
    with torch.device("cpu"):
        with omni_model_utils.init_on_device_without_buffers(torch.device("meta")):
            model = AutoModel.from_config(config, dtype=torch.float32)
        del model.visual
        encoder_vllm_tp.apply_text_encoder_tp_plan(model)
        omni_model_utils.init_parameters_preserving_vllm_linear(model, dtype=torch.float32, device=torch.device("cpu"))
    model.eval()

    root = torch.nn.Module()
    root.text_encoder = model
    AutoWeightsLoader(root, ignore_unexpected_prefixes=["text_encoder.visual."]).load_weights(
        _iter_checkpoint(checkpoint, "text_encoder.")
    )

    with torch.no_grad():
        actual = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    # Sharded GEMMs may differ from the full-width ones by float reassociation.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    dist.barrier()
    dist.destroy_process_group()
    print(f"TE_MEGATRON_TP_DIST_OK rank={rank}", flush=True)


if __name__ == "__main__":
    main()
