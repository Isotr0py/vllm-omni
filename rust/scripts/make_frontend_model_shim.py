#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a text-only "shim" model directory for the vllm-omni-rs frontend.

The Rust frontend loads the tokenizer and model config itself, but omni
checkpoints (Qwen2.5-Omni, Qwen3-Omni, ...) use composite configs where
``vocab_size`` lives at ``thinker_config.text_config.vocab_size`` — two levels
deep, while the Rust loader only checks the top level or a single nested
``text_config``. This script writes a small directory containing:

- a synthesized ``config.json`` with the thinker's text config hoisted to the
  top level (``vocab_size``, ``eos_token_id`` etc.); ``model_type`` values the
  Rust side cannot recognize are mapped to their plain-text backbone type;
- symlinks to the tokenizer/chat-template files of the real checkpoint.

Only the Rust ``--model`` argument should point at the shim; the omni façade
keeps using the real model id.
"""

import argparse
import json
import os
from pathlib import Path

# model_type of the hoisted text config -> plain-text backbone the Rust
# frontend understands. Unknown types are passed through unchanged (the
# renderer auto-detection falls back to the generic HF path).
MODEL_TYPE_MAP = {
    "qwen2_5_omni_thinker": "qwen2",
    "qwen3_omni_moe_thinker": "qwen3_moe",
    "qwen3_omni_thinker": "qwen3",
}

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
]


def find_text_config(config: dict) -> dict:
    """Return the LLM text config from a (possibly composite) model config."""
    if "vocab_size" in config:
        return config
    thinker = config.get("thinker_config")
    if isinstance(thinker, dict) and isinstance(thinker.get("text_config"), dict):
        return thinker["text_config"]
    if isinstance(config.get("text_config"), dict):
        return config["text_config"]
    raise ValueError("no vocab_size found at top level, text_config, or thinker_config.text_config")


def make_shim(model: str, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    if os.path.isdir(model):
        snapshot = Path(model)
    else:
        snapshot = Path(snapshot_download(model, allow_patterns=["*.json", "*.txt", "*.jinja"]))
    src_config = json.loads((snapshot / "config.json").read_text())
    text_config = find_text_config(src_config)

    shim_config = dict(text_config)
    shim_config["model_type"] = MODEL_TYPE_MAP.get(text_config.get("model_type"), text_config.get("model_type"))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(shim_config, indent=2))
    for name in TOKENIZER_FILES:
        target = snapshot / name
        link = out_dir / name
        if target.exists() and not link.exists():
            link.symlink_to(target)
    if not (out_dir / "tokenizer.json").exists():
        # Checkpoint ships only the slow BPE tokenizer (vocab.json+merges.txt);
        # the Rust loader requires a fast tokenizer, so convert and save one.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
        tokenizer.save_pretrained(out_dir)
        assert (out_dir / "tokenizer.json").exists(), "fast tokenizer conversion failed"
    print(f"shim written to {out_dir} (model_type={shim_config['model_type']}, vocab_size={shim_config['vocab_size']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HF model id or local checkpoint path")
    parser.add_argument("out_dir", type=Path, help="output shim directory")
    args = parser.parse_args()
    make_shim(args.model, args.out_dir)
