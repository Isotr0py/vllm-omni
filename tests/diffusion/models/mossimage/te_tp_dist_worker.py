# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Child-process worker for test_te_tp_forward_gloo_two_ranks.

Not a pytest module: spawned by the test with argv = [rank, repo_root].
Each rank builds the identical tiny Qwen3.5 (same seed), checks the sliced
TP forward against its own full-forward reference, and prints TE_TP_DIST_OK
on success.  encoder_tp.py is loaded by file path so the worker needs no
vllm/vllm_omni import machinery.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _load_encoder_tp(repo_root: str):
    path = Path(repo_root) / "vllm_omni/diffusion/models/mossimage/encoder_tp.py"
    spec = importlib.util.spec_from_file_location("mossimage_encoder_tp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rank = int(sys.argv[1])
    repo_root = sys.argv[2]
    dist.init_process_group("gloo", init_method=f"file://{os.environ['TE_TP_STORE']}", rank=rank, world_size=2)

    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    encoder_tp = _load_encoder_tp(repo_root)

    torch.manual_seed(0)
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
    model = Qwen3_5Model(Qwen3_5Config(text_config=text_cfg.to_dict(), vision_config=vision_cfg.to_dict())).eval()

    torch.manual_seed(1)
    input_ids = torch.randint(0, 256, (2, 11))
    attention_mask = torch.ones(2, 11, dtype=torch.long)
    attention_mask[1, 7:] = 0

    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    encoder_tp.slice_text_encoder_for_tp(model, rank, 2)
    with torch.no_grad():
        actual = encoder_tp.text_encoder_tp_forward(
            model.language_model,
            input_ids,
            attention_mask,
            rank=rank,
            tp_size=2,
            device_group=None,  # default (world) group
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dist.barrier()
    dist.destroy_process_group()
    print(f"TE_TP_DIST_OK rank={rank}", flush=True)


if __name__ == "__main__":
    main()
