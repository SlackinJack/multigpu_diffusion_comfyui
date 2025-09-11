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


ASYNCDIFF_CONFIGS = {
    "model":            MODEL,
    "type":             ASYNCDIFF_MODEL_LIST,
    "nproc_per_node":   NPROC_PER_NODE,
    "model_n":          MODEL_N,
    "stride":           STRIDE,
    "time_shift":       BOOLEAN_DEFAULT_FALSE,
}


ASYNCDIFF_CONFIGS_OPTIONAL = {
}


class AsyncDiffPipelineConfig:
    @classmethod
    def INPUT_TYPES(s):
        global ASYNCDIFF_CONFIGS
        return { "required": ASYNCDIFF_CONFIGS, "optional": ASYNCDIFF_CONFIGS_OPTIONAL }
    RETURN_TYPES    = ASYNCDIFF_CONFIG
    FUNCTION        = "get_config"
    CATEGORY        = ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs):
        return (kwargs,)


class AsyncDiffADSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":               GENERIC_CONFIG,
                "asyncdiff_config":     ASYNCDIFF_CONFIG,
                "positive_prompt":      PROMPT,
                "seed":                 SEED,
                "steps":                STEPS,
                "guidance_scale":       CFG,
                "num_frames":           NUM_FRAMES,
            },
            "optional": {
                "negative_prompt":      PROMPT,
                "ip_image":             IMAGE,
                "ip_image_scale":       SCALE_PERCENTAGE,
                "control_image":        IMAGE,
                "control_image_scale":  SCALE_PERCENTAGE,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = ASYNCDIFF_CATEGORY

    def generate(
        self,
        config,
        asyncdiff_config,
        positive_prompt,
        seed,
        steps,
        guidance_scale,
        num_frames,
        negative_prompt=None,
        ip_image=None,
        ip_image_scale=None,
        control_image=None,
        control_image_scale=None,
    ):
        assert (config.get("motion_module") is not None) or (config.get("motion_adapter") is not None), "Either a motion module or a motion adapter must be set."
        assert (config.get("motion_module") is None) or (config.get("motion_adapter") is None), "Only one motion module or motion adapter must be set."
        assert (len(positive_prompt) > 0), "You must provide a prompt."

        bar = ProgressBar(100)
        config.update(asyncdiff_config)
        launch_host(config, "asyncdiff", bar)

        data = {
            "positive": positive_prompt,
            "seed":     seed,
            "steps":    steps,
            "cfg":      guidance_scale,
            "frames":   num_frames,
        }

        if negative_prompt is not None: data["negative"] = negative_prompt

        if ip_image is not None and config.get("ip_adapter"):
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            data["ip_image_scale"] = ip_image_scale,

        if control_image is not None and config.get("control_net"):
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            data["control_image_scale"] = image_scale,

        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            images = decode_b64_and_unpickle(response)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            print("Successfully created media")
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            assert False, "No media generated.\nCheck console for details."


class AsyncDiffSDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":               GENERIC_CONFIG,
                "asyncdiff_config":     ASYNCDIFF_CONFIG,
                "seed":                 SEED,
                "steps":                STEPS,
                "guidance_scale":       CFG,
                "clip_skip":            CLIP_SKIP,
            },
            "optional": {
                "positive_prompt":      PROMPT,
                "negative_prompt":      PROMPT,
                "ip_image":             IMAGE,
                "ip_image_scale":       SCALE_PERCENTAGE,
                "control_image":        IMAGE,
                "control_image_scale":  SCALE_PERCENTAGE,
                "positive_embeds":      CONDITIONING,
                "negative_embeds":      CONDITIONING,
                "latent":               LATENT,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = ASYNCDIFF_CATEGORY

    def generate(
        self,
        config,
        asyncdiff_config,
        seed,
        steps,
        guidance_scale,
        clip_skip,
        positive_prompt=None,
        negative_prompt=None,
        ip_image=None,
        ip_image_scale=None,
        control_image=None,
        control_image_scale=None,
        positive_embeds=None,
        negative_embeds=None,
        latent=None
    ):
        assert (len(positive_prompt) > 0 or positive_embeds is not None), "You must provide a positive input."
        if positive_prompt is not None and len(positive_prompt) > 0:
            assert (positive_embeds is None), "Provide a positive prompt or a positive embedding, but not both."
        if negative_prompt is not None and len(negative_prompt) > 0:
            assert (negative_embeds is None), "Provide a negative prompt or a negative embedding, but not both."

        bar = ProgressBar(100)
        config.update(asyncdiff_config)
        launch_host(config, "asyncdiff", bar)

        data = {
            "seed":         seed,
            "steps":        steps,
            "cfg":          guidance_scale,
            "clip_skip":    clip_skip,
        }

        if positive_prompt is not None: data["positive"] = positive_prompt
        if negative_prompt is not None: data["negative"] = negative_prompt
        if positive_embeds is not None: data["positive_embeds"] = pickle_and_encode_b64(positive_embeds)
        if negative_embeds is not None: data["negative_embeds"] = pickle_and_encode_b64(negative_embeds)
        if latent is not None:          data["latent"] = pickle_and_encode_b64(latent["samples"])
        if ip_image is not None and config.get("ip_adapter") is not None:
            ip_image = ip_image.squeeze(0)              # NHWC -> HWC
            data["ip_image"] = convert_tensor_to_b64(ip_image)
            data["ip_image_scale"] = ip_image_scale
        if control_image is not None and config.get("control_net") is not None:
            control_image = control_image.squeeze(0)    # NHWC -> HWC
            data["control_image"] = convert_tensor_to_b64(control_image)
            data["control_image_scale"] = control_image_scale

        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            print("Successfully created media")
            return (convert_b64_to_nhwc_tensor(response),)
        else:
            assert False, "No media generated.\nCheck console for details."


class AsyncDiffSVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":               GENERIC_CONFIG,
                "asyncdiff_config":     ASYNCDIFF_CONFIG,
                "image":                IMAGE,
                "image_scale":          SCALE_PERCENTAGE,
                "seed":                 SEED,
                "steps":                STEPS,
                "decode_chunk_size":    DECODE_CHUNK_SIZE,
                "num_frames":           NUM_FRAMES,
                "motion_bucket_id":     MOTION_BUCKET_ID,
                "noise_aug_strength":   NOISE_AUG_STRENGTH,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = ASYNCDIFF_CATEGORY

    def generate(
        self,
        config,
        asyncdiff_config,
        image,
        image_scale,
        seed,
        steps,
        decode_chunk_size,
        num_frames,
        motion_bucket_id,
        noise_aug_strength
    ):
        assert (image is not None), "You must provide an image."

        bar = ProgressBar(100)
        config.update(asyncdiff_config)
        launch_host(config, "asyncdiff", bar)

        image = image.squeeze(0)                    # NHWC -> HWC
        b64_image = convert_tensor_to_b64(image)
        data = {
            "image":                b64_image,
            "image_scale":          image_scale,
            "seed":                 seed,
            "steps":                steps,
            "decode_chunk_size":    decode_chunk_size,
            "frames":               num_frames,
            "motion_bucket_id":     motion_bucket_id,
            "noise_aug_strength":   noise_aug_strength,
        }
        response = get_result(data, bar)
        if response is not None:
            bar.update_absolute(100)
            images = decode_b64_and_unpickle(response)
            tensors = []
            for i in images:
                tensors.append(convert_image_to_hwc_tensor(i))
            print("Successfully created media")
            return (torch.stack(tuple(tensors)),)   # HWC -> NHWC
        else:
            assert False, "No media generated.\nCheck console for details."


class AsyncDiffSDUpscaleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":           GENERIC_CONFIG,
                "asyncdiff_config": ASYNCDIFF_CONFIG,
                "image":            IMAGE,
                "image_scale":      SCALE_PERCENTAGE,
                "positive_prompt":  PROMPT,
                "seed":             SEED,
                "steps":            STEPS,
                "guidance_scale":   CFG,
            },
            "optional": {
                "negative_prompt":  PROMPT,
            }
        }

    RETURN_TYPES    = IMAGE
    FUNCTION        = "generate"
    CATEGORY        = ASYNCDIFF_CATEGORY

    def generate(
        self,
        config,
        asyncdiff_config,
        image,
        image_scale,
        positive_prompt,
        seed,
        steps,
        guidance_scale,
        negative_prompt=None
    ):
        assert (len(positive_prompt) > 0), "You must provide a prompt."

        bar = ProgressBar(100)
        config.update(asyncdiff_config)
        launch_host(config, "asyncdiff", bar)

        if image.size(0) > 1:   images = list(torch.unbind(image, 0))   # NHWC -> [HWC], len == N
        else:                   images = [image.squeeze(0)]             # NHWC -> [HWC], len == 1
        tensors = []
        i = 0
        for im in images:
            i += 1
            print(f"Upscaling image: {i}/{len(images)}")
            b64_image = convert_tensor_to_b64(im)
            data = {
                "image":        b64_image,
                "image_scale":  image_scale,
                "positive":     positive_prompt,
                "seed":         seed,
                "steps":        steps,
                "cfg":          guidance_scale,
            }

            if negative_prompt is not None: data["negative"] = negative_prompt

            try:
                response = get_result(data, bar)
                if response is not None:
                    print(f"Finished upscaling image: {i}/{len(images)}")
                    im2 = decode_b64_and_unpickle(response)
                    tensors.append(convert_image_to_hwc_tensor(im2))
                else:
                    if len(images) == 1:
                        print("No media generated")
                    else:
                        print(f"Error processing image: {i}/{len(images)}")
            except Exception as e:
                print("Error getting data from server.")
                print(str(e))
        bar.update_absolute(100)
        print("Successfully created media")
        return (torch.stack(tuple(tensors)),)       # HWC -> NHWC

