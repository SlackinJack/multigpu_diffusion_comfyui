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


DISTRIFUSER_CONFIGS = {
    "model":            MODEL,
    "type":             DISTRIFUSER_MODEL_LIST,
    "nproc_per_node":   NPROC_PER_NODE,
    "parallelism":      DISTRIFUSER_PARALLELISM_LIST,
    "sync_mode":        DISTRIFUSER_SYNC_MODE_LIST,
    "no_cuda_graph":    BOOLEAN_DEFAULT_FALSE,
    "no_split_batch":   BOOLEAN_DEFAULT_FALSE,
}


DISTRIFUSER_CONFIGS_OPTIONAL = {
}


class DistrifuserPipelineConfig:
    @classmethod
    def INPUT_TYPES(s):
        global DISTRIFUSER_CONFIGS, DISTRIFUSER_CONFIGS_OPTIONAL
        return { "required": DISTRIFUSER_CONFIGS, "optional": DISTRIFUSER_CONFIGS_OPTIONAL}
    RETURN_TYPES = DISTRIFUSER_CONFIG
    FUNCTION = "get_config"
    CATEGORY = ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        return (kwargs,)


class DistrifuserSDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           GENERIC_CONFIG,
                "distri_config":    DISTRIFUSER_CONFIG,
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
                "image":            IMAGE,
                "latent":           LATENT,
            }
        }

    RETURN_TYPES = IMAGE
    FUNCTION = "generate"
    CATEGORY = DISTRIFUSER_CATEGORY

    def generate(
        self,
        config,
        distri_config,
        seed,
        steps,
        guidance_scale,
        clip_skip,
        positive_prompt=None,
        negative_prompt=None,
        positive_embeds=None,
        negative_embeds=None,
        image=None,
        latent=None,
    ):
        assert (len(positive_prompt) > 0 or positive_embeds), "You must provide a prompt."
        if positive_prompt is not None and len(positive_prompt) > 0:
            assert (positive_embeds is None), "Provide a positive prompt or a positive embedding, but not both."
        if negative_prompt is not None and len(negative_prompt) > 0:
            assert (negative_embeds is None), "Provide a negative prompt or a negative embedding, but not both."
        if config.get("lora") is not None and distri_config["nproc_per_node"] > 2:
            assert False, "In order to use LoRAs, you must use nproc_per_node <= 2."

        bar = ProgressBar(100)
        config.update(distri_config)
        launch_host(config, "distrifuser", bar)

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

        if image is not None and config.get("ip_adapter"):
            image = image.squeeze(0)                # NHWC -> HWC
            data["image"] = convert_tensor_to_b64(image)

        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            print("Successfully created media")
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated.\nCheck console for details."

