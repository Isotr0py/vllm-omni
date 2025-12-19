import os

import pytest
import torch

from vllm_omni.utils.platform_utils import is_npu
from .conftest import OmniRunner

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


models = ["Tongyi-MAI/Z-Image-Turbo", "riverclouds/qwen_image_random"]

# NPU still can't run Tongyi-MAI/Z-Image-Turbo properly
# Modelscope can't find riverclouds/qwen_image_random
# TODO: When NPU support is ready, remove this branch.
if is_npu():
    models = ["Qwen/Qwen-Image"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(omni_runner: type[OmniRunner], model_name: str):
    with omni_runner(model_name, mode="diffusion") as m:
        # high resolution may cause OOM on L4
        height = 256
        width = 256
        images = m.generate_diffusion(
            "a photo of a cat sitting on a laptop keyboard",
            height=height,
            width=width,
            num_inference_steps=2,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
            num_outputs_per_prompt=2,
        )
        assert len(images) == 2
        # check image size
        assert images[0].width == width
        assert images[0].height == height
        images[0].save("image_output.png")
