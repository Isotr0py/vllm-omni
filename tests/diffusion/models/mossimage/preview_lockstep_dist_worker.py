# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Child-process worker for test_preview_decode_participates_on_all_ranks.

Not a pytest module: spawned with argv = [rank, repo_root] and
PREVIEW_TP_STORE in the environment. Each rank builds a pipeline stub whose
decode_preview carries a TP-group collective (like the patch-parallel VAE
wrapper's broadcasts) and calls build_progress_output; both ranks must finish
(a primary-only decode would deadlock the collective). Prints
PREVIEW_LOCKSTEP_DIST_OK on success.
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


def main() -> None:
    rank = int(sys.argv[1])
    repo_root = sys.argv[2]
    dist.init_process_group("gloo", init_method=f"file://{os.environ['PREVIEW_TP_STORE']}", rank=rank, world_size=2)

    spec = importlib.util.spec_from_file_location("omni_progress", Path(repo_root) / "vllm_omni/diffusion/progress.py")
    progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress)

    class _Pipeline(torch.nn.Module):
        def decode_preview(self, latents, max_side=256):
            # Stand-in for the patch-parallel VAE wrapper: a collective that
            # hangs unless every rank calls decode_preview.
            t = latents.float().sum().reshape(1)
            dist.all_reduce(t)
            return "cHJldmlldw" if t.item() else None

    pipeline = _Pipeline()
    output = progress.build_progress_output(
        pipeline,
        extra_args={"stream_progress": True, "preview_interval": 1},
        step_index=1,
        total=4,
        latents=torch.ones(1, 4, 4, 4),
        request_id="req",
        is_primary=(rank == 0),
    )
    if rank == 0:
        assert output is not None and output.custom_output.get("preview_b64") == "cHJldmlldw"
    else:
        assert output is None  # decoded for the collective, result discarded
    dist.barrier()
    dist.destroy_process_group()
    print(f"PREVIEW_LOCKSTEP_DIST_OK rank={rank}", flush=True)


if __name__ == "__main__":
    main()
