# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ControlNet-LLLite (v2) adapter for the Anima DiT.

Implements the kohya sd-scripts LLLite v2 architecture used by the
toyxyz/Anima_ControlNet-LLLite checkpoints: a shared conditioning trunk
(``conditioning1``) turns a pixel-space control image into a per-token
embedding, and one tiny FiLM-modulated MLP per targeted attention projection
(LLLiteModule) computes an additive correction on the projection input.

Unlike the reference (which monkey-patches ``nn.Linear.forward``), the
correction is applied by the attention layer itself: ``apply_to`` stashes
``(q_mod, k_mod, v_mod, net)`` on each block's ``attn1.lllite`` and
``CosmosAttention`` folds the correction into the fused ``to_qkv`` output.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

logger = logging.getLogger(__name__)

_LLLITE_ARCH_VERSION = "2"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_MODULE_RE = re.compile(r"^lllite_dit_blocks_(\d+)_self_attn_([qkv])_proj\.(.+)$")

# Checkpoint self_attn component order within one block: q, k, v.
_COMPONENT_INDEX = {"q": 0, "k": 1, "v": 2}


def _gn(channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = _gn(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _ASPP(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...]):
        super().__init__()
        branches = []
        for dilation in dilations:
            if dilation == 1:
                conv = nn.Conv2d(channels, channels, kernel_size=1)
            else:
                conv = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
            branches.append(nn.Sequential(conv, _gn(channels), nn.SiLU()))
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), _gn(channels), nn.SiLU())
        self.proj = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, kernel_size=1),
            _gn(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        outs = [branch(x) for branch in self.branches]
        global_out = self.global_conv(self.global_pool(x))
        global_out = F.interpolate(global_out, size=(height, width), mode="bilinear", align_corners=False)
        outs.append(global_out)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    """Shared conditioning trunk: control image -> (B, S, cond_emb_dim) tokens.

    Total stride is 16 (conv1 stride 4, conv3 stride 4), so the output token
    grid matches the Anima DiT image-token grid for a (16*H_tok, 16*W_tok)
    control image.
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool,
        aspp_dilations: tuple[int, ...],
        cond_in_channels: int,
    ):
        super().__init__()
        if cond_dim % 2 != 0:
            raise ValueError(f"cond_dim must be even, got {cond_dim}")
        ch_half = cond_dim // 2

        self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for resblock in self.resblocks:
            h = resblock(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        return self.out_norm(h)


class LLLiteModule(nn.Module):
    """FiLM-modulated correction MLP for one attention q/k/v projection input."""

    def __init__(self, hidden_dim: int, cond_emb_dim: int, mlp_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.cond_emb_dim = cond_emb_dim

        self.down = nn.Linear(hidden_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        self.up = nn.Linear(mlp_dim, hidden_dim)
        # Zero-init so the correction is exactly zero at train start.
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.depth_embed = nn.Parameter(torch.zeros(cond_emb_dim))

    def forward_correction(
        self,
        x: torch.Tensor,
        cond_emb: torch.Tensor | None,
        multiplier: float,
    ) -> torch.Tensor | None:
        """Return the additive correction for the projection input ``x`` (B, S, hidden_dim)."""
        if multiplier == 0.0 or cond_emb is None:
            return None

        cx = cond_emb
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return None
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)
        if x.shape[1] != cx.shape[1]:
            return None

        # Run the mini-MLP in the LLLite parameter dtype; cast back to x.dtype on exit.
        param_dtype = self.down.weight.dtype
        x_proc = x.to(param_dtype) if x.dtype != param_dtype else x
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)
        depth_embed = self.depth_embed
        if depth_embed.device != x.device:
            depth_embed = depth_embed.to(x.device)

        cond_local = cx + depth_embed
        h = F.silu(self.down(x_proc))
        gamma, beta = self.cond_to_film(cond_local).chunk(2, dim=-1)
        t = self.mid(torch.cat([cond_local, h], dim=-1))
        t = t * (1 + gamma) + beta
        t = F.silu(t)
        out = self.up(t) * multiplier
        return out.to(x.dtype) if out.dtype != x.dtype else out


class AnimaControlNetLLLite(nn.Module):
    """ControlNet-LLLite adapter for AnimaTransformer3DModel (v2 checkpoints)."""

    def __init__(
        self,
        num_blocks: int = 28,
        hidden_dim: int = 2048,
        cond_in_channels: int = 3,
        cond_emb_dim: int = 64,
        cond_dim: int = 128,
        mlp_dim: int = 64,
        cond_resblocks: int = 4,
        aspp_dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.conditioning1 = _Conditioning1(
            cond_dim=cond_dim,
            cond_emb_dim=cond_emb_dim,
            n_resblocks=cond_resblocks,
            use_aspp=True,
            aspp_dilations=tuple(aspp_dilations),
            cond_in_channels=cond_in_channels,
        )
        self.block_modules = nn.ModuleList(
            LLLiteModule(hidden_dim, cond_emb_dim, mlp_dim) for _ in range(3 * num_blocks)
        )
        self.multiplier = 1.0
        self.cond_emb: torch.Tensor | None = None

    @classmethod
    def from_single_file(
        cls,
        path: str,
        dtype: torch.dtype | None = None,
        device=None,
    ) -> AnimaControlNetLLLite:
        """Build the adapter from a safetensors checkpoint's metadata and load weights strictly."""
        lllite_file = Path(path)
        with safe_open(lllite_file, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            version = metadata.get("lllite.version")
            if version != _LLLITE_ARCH_VERSION:
                raise ValueError(
                    f"{lllite_file} is not an LLLite v{_LLLITE_ARCH_VERSION} checkpoint (lllite.version={version!r})"
                )
            try:
                cond_dim = int(metadata["lllite.cond_dim"])
                cond_emb_dim = int(metadata["lllite.cond_emb_dim"])
                mlp_dim = int(metadata["lllite.mlp_dim"])
                cond_resblocks = int(metadata["lllite.cond_resblocks"])
                use_aspp = metadata["lllite.use_aspp"].lower() == "true"
                aspp_dilations = tuple(int(d) for d in metadata["lllite.aspp_dilations"].split(","))
                cond_in_channels = int(metadata["lllite.cond_in_channels"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{lllite_file} has malformed LLLite metadata: {metadata}") from exc
            if not use_aspp:
                raise ValueError(f"{lllite_file} was trained without ASPP; only use_aspp=true is supported.")

            # hidden_dim and num_blocks are not recorded in the metadata; infer
            # them from the module tensors (down.weight is (mlp_dim, hidden_dim)).
            num_blocks = 0
            hidden_dim: int | None = None
            for key in checkpoint.keys():
                if key.startswith(_SAVED_COND_PREFIX):
                    continue
                match = _SAVED_MODULE_RE.match(key)
                if match is None:
                    raise ValueError(f"Unrecognized key {key!r} in LLLite checkpoint {lllite_file}")
                num_blocks = max(num_blocks, int(match.group(1)) + 1)
                if match.group(3) == "down.weight" and hidden_dim is None:
                    hidden_dim = checkpoint.get_slice(key).get_shape()[1]
            if num_blocks == 0 or hidden_dim is None:
                raise ValueError(f"{lllite_file} contains no LLLite attention modules")

            net = cls(
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                cond_in_channels=cond_in_channels,
                cond_emb_dim=cond_emb_dim,
                cond_dim=cond_dim,
                mlp_dim=mlp_dim,
                cond_resblocks=cond_resblocks,
                aspp_dilations=aspp_dilations,
            )

            state_dict: dict[str, torch.Tensor] = {}
            for key in checkpoint.keys():
                tensor = checkpoint.get_tensor(key)
                if key.startswith(_SAVED_COND_PREFIX):
                    state_dict["conditioning1." + key[len(_SAVED_COND_PREFIX) :]] = tensor
                    continue
                module_idx = _parse_module_key(key, net.num_blocks)
                if module_idx is None:
                    raise ValueError(f"Unrecognized key {key!r} in LLLite checkpoint {lllite_file}")
                block_idx, component, tail = module_idx
                state_dict[f"block_modules.{block_idx * 3 + component}.{tail}"] = tensor

        net.load_state_dict(state_dict, strict=True)
        if dtype is not None or device is not None:
            net = net.to(dtype=dtype, device=device)
        logger.info(
            "Loaded Anima ControlNet-LLLite %s: %d modules, cond_dim=%d, cond_emb_dim=%d, "
            "mlp_dim=%d, cond_resblocks=%d, aspp_dilations=%s",
            lllite_file,
            len(net.block_modules),
            cond_dim,
            cond_emb_dim,
            mlp_dim,
            cond_resblocks,
            aspp_dilations,
        )
        return net

    def _cond_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = self.conditioning1.conv1.weight
        return param.device, param.dtype

    def set_cond_image(self, cond_image: torch.Tensor) -> None:
        """Compute and store the conditioning embedding for a (1, 3, H, W) image in [-1, 1]."""
        device, dtype = self._cond_device_dtype()
        with torch.no_grad():
            self.cond_emb = self.conditioning1(cond_image.to(device=device, dtype=dtype))

    def clear_cond_image(self) -> None:
        self.cond_emb = None

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)

    def apply_to(self, transformer) -> None:
        """Attach the per-block modules to ``transformer.transformer_blocks[i].attn1.lllite``."""
        blocks = transformer.transformer_blocks
        if len(blocks) != self.num_blocks:
            raise ValueError(f"Transformer has {len(blocks)} blocks, adapter expects {self.num_blocks}")
        for block_idx, block in enumerate(blocks):
            base = block_idx * 3
            block.attn1.lllite = (
                self.block_modules[base],
                self.block_modules[base + 1],
                self.block_modules[base + 2],
                self,
            )

    def restore(self, transformer) -> None:
        """Detach the adapter from the transformer."""
        for block in transformer.transformer_blocks:
            block.attn1.lllite = None


def _parse_module_key(key: str, num_blocks: int) -> tuple[int, int, str] | None:
    """Parse ``lllite_dit_blocks_{i}_self_attn_{q,k,v}_proj.{tail}`` -> (block_idx, component_idx, tail)."""
    match = _SAVED_MODULE_RE.match(key)
    if match is None:
        return None
    block_idx = int(match.group(1))
    if block_idx >= num_blocks:
        raise ValueError(f"LLLite checkpoint key {key!r} targets block {block_idx} >= num_blocks={num_blocks}")
    return block_idx, _COMPONENT_INDEX[match.group(2)], match.group(3)
