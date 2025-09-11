import base64
import json
import os
import pickle
import requests
import socket
import subprocess
import time
import torch
from comfy.utils import ProgressBar
from torchvision.transforms import ToPILImage, ToTensor


from .globals import *
from .nodes_general import *
from ..modules.host_manager import *
from ..multigpu_diffusion.modules.utils import *


XDIT_CONFIGS = {
    "checkpoint":                   CHECKPOINT,
    "type":                         XDIT_MODEL_LIST,
    "nproc_per_node":               NPROC_PER_NODE,
    "pipefusion_parallel_degree":   PIPEFUSION_PARALLEL_DEGREE,
    "tensor_parallel_degree":       TENSOR_PARALLEL_DEGREE,
    "data_parallel_degree":         DATA_PARALLEL_DEGREE,
    "ulysses_degree":               ULYSSES_DEGREE,
    "ring_degree":                  RING_DEGREE,
    "use_cfg_parallel":             BOOLEAN_DEFAULT_FALSE,
}


XDIT_CONFIGS_OPTIONAL = {
    "gguf_model":                   MODEL_GGUF,
}


class xDiTConfig:
    @classmethod
    def INPUT_TYPES(s):
        global XDIT_CONFIGS, XDIT_CONFIGS_OPTIONAL
        return { "required": XDIT_CONFIGS, "optional": XDIT_CONFIGS_OPTIONAL }
    RETURN_TYPES    = XDIT_CONFIG
    FUNCTION        = "get_config"
    CATEGORY        = ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        return (kwargs,)


class xDiTSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host_config":      GENERIC_CONFIG,
                "xdit_config":      XDIT_CONFIG,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
                "clip_skip":        CLIP_SKIP,
            },
            "optional": {
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "positive_embeds":  CONDITIONING,
                "negative_embeds":  CONDITIONING,
                "ip_image":         IMAGE,
                "ip_image_scale":   SCALE_PERCENTAGE,
                "latent":           LATENT,
            }
        }

    RETURN_TYPES = IMAGE
    FUNCTION = "generate"
    CATEGORY = XDIT_CATEGORY

    def generate(
        self,
        host_config,
        xdit_config,
        seed,
        steps,
        guidance_scale,
        clip_skip,
        positive_prompt=None,
        negative_prompt=None,
        positive_embeds=None,
        negative_embeds=None,
        ip_image=None,
        ip_image_scale=None,
        latent=None,
    ):
        assert (len(positive_prompt) > 0 or positive_embeds is not None), "You must provide a prompt."
        if positive_prompt is not None and len(positive_prompt) > 0:
            assert (positive_embeds is None), "Provide a positive prompt or a positive embedding, but not both."
        if negative_prompt is not None and len(negative_prompt) > 0:
            assert (negative_embeds is None), "Provide a negative prompt or a negative embedding, but not both."

        bar = ProgressBar(100)
        host_config.update(xdit_config)
        launch_host(host_config, "xdit", bar)

        data = {
            "steps": steps,
            "seed": seed,
            "cfg": guidance_scale,
            "clip_skip": clip_skip,
        }

        if positive_prompt is not None: data["positive"] = positive_prompt
        if negative_prompt is not None: data["negative"] = negative_prompt
        if positive_embeds is not None: data["positive_embeds"] = pickle_and_encode_b64(positive_embeds)
        if negative_embeds is not None: data["negative_embeds"] = pickle_and_encode_b64(negative_embeds)
        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])

        if ip_image is not None and host_config.get("ip_adapter"):
            ip_image = ip_image.squeeze(0)  # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            data["ip_image_scale"] = ip_image_scale

        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            print("Successfully created media")
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated.\nCheck console for details."

