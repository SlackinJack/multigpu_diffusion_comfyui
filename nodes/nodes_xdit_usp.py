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


XDIT_USP_CONFIGS = {
    "model":                        MODEL,
    "type":                         XDIT_USP_MODEL_LIST,
    "nproc_per_node":               NPROC_PER_NODE,
    "ulysses_degree":               ULYSSES_DEGREE,
    "ring_degree":                  RING_DEGREE,
}


XDIT_USP_CONFIGS_OPTIONAL = {
    "gguf_model":                   MODEL_GGUF,
}


class xDiTUSPConfig:
    @classmethod
    def INPUT_TYPES(s):
        global XDIT_USP_CONFIGS, XDIT_USP_CONFIGS_OPTIONAL
        return { "required": XDIT_USP_CONFIGS, "optional": XDIT_USP_CONFIGS_OPTIONAL }
    RETURN_TYPES    = XDIT_USP_CONFIG
    FUNCTION        = "get_config"
    CATEGORY        = ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        return (kwargs,)


class xDiTUSPImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           GENERIC_CONFIG,
                "xdit_usp_config":  XDIT_USP_CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
                "clip_skip":        CLIP_SKIP,
            },
            "optional": {
            }
        }

    RETURN_TYPES = IMAGE
    FUNCTION = "generate"
    CATEGORY = XDIT_CATEGORY

    def generate(
        self,
        config,
        xdit_usp_config,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        guidance_scale,
        clip_skip,
    ):
        assert (len(positive_prompt) > 0), "You must provide a prompt."

        bar = ProgressBar(100)
        config.update(xdit_usp_config)
        launch_host(config, "xdit_usp", bar)

        data = {
            "steps": steps,
            "seed": seed,
            "cfg": guidance_scale,
            "clip_skip": clip_skip,
        }

        if positive_prompt is not None: data["positive"] = positive_prompt
        if negative_prompt is not None: data["negative"] = negative_prompt

        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            print("Successfully created media")
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated.\nCheck console for details."


class xDiTUSPVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           GENERIC_CONFIG,
                "xdit_usp_config":  XDIT_USP_CONFIG,
                "positive_prompt":  PROMPT,
                "negative_prompt":  PROMPT,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
                "num_frames":       NUM_FRAMES,
            },
            "optional": {
                "image":            IMAGE,
            }
        }

    RETURN_TYPES = IMAGE
    FUNCTION = "generate"
    CATEGORY = XDIT_CATEGORY

    def generate(
        self,
        config,
        xdit_usp_config,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        guidance_scale,
        num_frames,
        image=None,
    ):
        assert (len(positive_prompt) > 0), "You must provide a prompt."

        bar = ProgressBar(100)
        config.update(xdit_usp_config)
        launch_host(config, "xdit_usp", bar)

        data = {
            "steps":    steps,
            "seed":     seed,
            "cfg":      guidance_scale,
            "frames":   num_frames,
        }

        if positive_prompt is not None: data["positive"] = positive_prompt
        if negative_prompt is not None: data["negative"] = negative_prompt

        if image is not None:
            image = image.squeeze(0)                # NHWC -> HWC
            data["image"] = convert_tensor_to_b64(image)

        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            print("Successfully created media")
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated.\nCheck console for details."

